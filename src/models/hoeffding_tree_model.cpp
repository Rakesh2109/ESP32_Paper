#include "models/hoeffding_tree_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tree {

namespace {

float entropy_from_counts(const std::vector<std::uint32_t>& counts) {
    std::uint32_t total = 0;
    for (std::uint32_t count : counts) {
        total += count;
    }
    if (total == 0u) {
        return 0.0f;
    }

    const float inv_total = 1.0f / static_cast<float>(total);
    float entropy = 0.0f;
    for (std::uint32_t count : counts) {
        if (count == 0u) {
            continue;
        }
        const float probability = static_cast<float>(count) * inv_total;
        entropy -= probability * std::log2(probability);
    }
    return entropy;
}

std::uint32_t sum_counts(const std::vector<std::uint32_t>& counts) {
    std::uint32_t total = 0;
    for (std::uint32_t count : counts) {
        total += count;
    }
    return total;
}

std::size_t safe_bin_index(float value, float min_value, float max_value, std::size_t num_bins) {
    if (!(max_value > min_value)) {
        return 0u;
    }
    const float normalized = (value - min_value) / (max_value - min_value);
    const float clipped = std::max(0.0f, std::min(1.0f, normalized));
    std::size_t bin = static_cast<std::size_t>(clipped * static_cast<float>(num_bins));
    if (bin >= num_bins) {
        bin = num_bins - 1u;
    }
    return bin;
}

struct SplitCandidate {
    bool valid = false;
    std::size_t feature = 0u;
    std::size_t boundary_bin = 0u;
    float threshold = 0.0f;
    float gain = -std::numeric_limits<float>::infinity();
    std::vector<std::uint32_t> left_counts;
    std::vector<std::uint32_t> right_counts;
};

}  // namespace

struct HoeffdingTreeModel::Node {
    bool is_leaf = true;
    std::size_t split_feature = 0u;
    std::size_t split_boundary_bin = 0u;
    float split_threshold = 0.0f;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    std::vector<std::uint32_t> class_counts;
    std::vector<std::vector<std::vector<std::uint32_t>>> feature_bin_class_counts;
    std::vector<float> feature_min;
    std::vector<float> feature_max;

    std::size_t seen = 0u;
    std::size_t depth = 0u;
    std::size_t samples_since_eval = 0u;

    Node(std::size_t input_dim,
         std::size_t num_classes,
         std::size_t num_bins,
         std::size_t node_depth)
        : class_counts(num_classes, 0u),
          feature_bin_class_counts(
              input_dim,
              std::vector<std::vector<std::uint32_t>>(num_bins,
                                                      std::vector<std::uint32_t>(num_classes, 0u))),
          feature_min(input_dim, std::numeric_limits<float>::infinity()),
          feature_max(input_dim, -std::numeric_limits<float>::infinity()),
          depth(node_depth) {}
};

HoeffdingTreeModel::HoeffdingTreeModel(std::size_t input_dim, const HoeffdingTreeConfig& cfg)
    : input_dim_(input_dim), cfg_(cfg), root_(std::make_unique<Node>(input_dim,
                                                                      cfg.num_classes,
                                                                      cfg.num_threshold_bins,
                                                                      0u)) {
    if (input_dim_ == 0u) {
        throw std::runtime_error("HoeffdingTreeModel requires input_dim > 0");
    }
    if (cfg_.num_classes < 2u) {
        throw std::runtime_error("HoeffdingTreeModel requires num_classes >= 2");
    }
    if (cfg_.num_threshold_bins < 2u) {
        throw std::runtime_error("HoeffdingTreeModel requires num_threshold_bins >= 2");
    }
    if (cfg_.delta <= 0.0f || cfg_.delta >= 1.0f) {
        throw std::runtime_error("HoeffdingTreeModel delta must be in (0, 1)");
    }
}

HoeffdingTreeModel::~HoeffdingTreeModel() = default;

float HoeffdingTreeModel::hoeffding_bound(float range_val, float confidence, std::size_t n) {
    if (n == 0u) {
        return std::numeric_limits<float>::infinity();
    }
    return std::sqrt((range_val * range_val * std::log(1.0f / confidence)) / (2.0f * static_cast<float>(n)));
}

static SplitCandidate best_split_for_feature(const HoeffdingTreeModel::Node* node,
                                             std::size_t feature_idx,
                                             std::size_t num_bins,
                                             std::size_t num_classes) {
    SplitCandidate best;
    best.feature = feature_idx;
    best.left_counts.assign(num_classes, 0u);
    best.right_counts.assign(num_classes, 0u);

    std::vector<std::uint32_t> left(num_classes, 0u);
    std::vector<std::uint32_t> right(node->class_counts.begin(), node->class_counts.end());

    const float min_value = node->feature_min[feature_idx];
    const float max_value = node->feature_max[feature_idx];
    if (!(max_value > min_value)) {
        return best;
    }

    const float parent_entropy = entropy_from_counts(node->class_counts);
    const std::uint32_t parent_total = sum_counts(node->class_counts);
    if (parent_total == 0u) {
        return best;
    }

    for (std::size_t boundary = 0u; boundary + 1u < num_bins; ++boundary) {
        const auto& boundary_bin_counts = node->feature_bin_class_counts[feature_idx][boundary];
        for (std::size_t class_idx = 0u; class_idx < num_classes; ++class_idx) {
            left[class_idx] += boundary_bin_counts[class_idx];
            right[class_idx] -= boundary_bin_counts[class_idx];
        }

        const std::uint32_t left_total = sum_counts(left);
        const std::uint32_t right_total = sum_counts(right);
        if (left_total == 0u || right_total == 0u) {
            continue;
        }

        const float weighted_children =
            (static_cast<float>(left_total) / static_cast<float>(parent_total)) * entropy_from_counts(left) +
            (static_cast<float>(right_total) / static_cast<float>(parent_total)) * entropy_from_counts(right);
        const float gain = parent_entropy - weighted_children;

        if (!best.valid || gain > best.gain) {
            best.valid = true;
            best.gain = gain;
            best.boundary_bin = boundary;
            const float ratio = static_cast<float>(boundary + 1u) / static_cast<float>(num_bins);
            best.threshold = min_value + (max_value - min_value) * ratio;
            best.left_counts = left;
            best.right_counts = right;
        }
    }

    return best;
}

float HoeffdingTreeModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    if (sample.features.size() != input_dim_) {
        throw std::runtime_error("HoeffdingTreeModel::train_sample - feature size mismatch");
    }
    if (sample.label < 0 || static_cast<std::size_t>(sample.label) >= cfg_.num_classes) {
        throw std::runtime_error("HoeffdingTreeModel::train_sample - label out of range");
    }

    const int pred_before = predict(sample.features);
    const float loss = (pred_before == sample.label) ? 0.0f : 1.0f;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }

    const std::size_t label = static_cast<std::size_t>(sample.label);
    update_path(root_.get(), sample.features, label);

    samples_seen_ += 1u;
    if (cfg_.memory_estimate_period > 0u && (samples_seen_ % cfg_.memory_estimate_period) == 0u) {
        const std::size_t estimated_bytes = estimate_model_bytes(root_.get());
        if (estimated_bytes > static_cast<std::size_t>(cfg_.max_size_mib * (2.0f * 1024.0f * 1024.0f))) {
            if (cfg_.stop_mem_management) {
                growth_allowed_ = false;
            }
        }
    }

    return loss;
}

HoeffdingTreeModel::Node* HoeffdingTreeModel::update_path(Node* node,
                                                          const std::vector<float>& features,
                                                          std::size_t label) {
    Node* current = node;

    while (true) {
        current->seen += 1u;
        current->samples_since_eval += 1u;
        current->class_counts[label] += 1u;

        for (std::size_t i = 0u; i < input_dim_; ++i) {
            const float value = features[i];
            current->feature_min[i] = std::min(current->feature_min[i], value);
            current->feature_max[i] = std::max(current->feature_max[i], value);
            const std::size_t bin = safe_bin_index(value,
                                                   current->feature_min[i],
                                                   current->feature_max[i],
                                                   cfg_.num_threshold_bins);
            current->feature_bin_class_counts[i][bin][label] += 1u;
        }

        if (!current->is_leaf) {
            current = (features[current->split_feature] <= current->split_threshold)
                          ? current->left.get()
                          : current->right.get();
            continue;
        }

        if (!growth_allowed_) {
            return current;
        }

        if (current->depth >= cfg_.max_depth) {
            return current;
        }

        if (current->seen < cfg_.min_samples_split) {
            return current;
        }

        if (current->samples_since_eval < cfg_.grace_period) {
            return current;
        }

        current->samples_since_eval = 0u;

        SplitCandidate best;
        SplitCandidate second_best;
        for (std::size_t feature_idx = 0u; feature_idx < input_dim_; ++feature_idx) {
            SplitCandidate candidate = best_split_for_feature(current, feature_idx,
                                                              cfg_.num_threshold_bins,
                                                              cfg_.num_classes);
            if (!candidate.valid) {
                continue;
            }
            if (!best.valid || candidate.gain > best.gain) {
                second_best = best;
                best = std::move(candidate);
            } else if (!second_best.valid || candidate.gain > second_best.gain) {
                second_best = std::move(candidate);
            }
        }

        if (!best.valid) {
            return current;
        }

        const float range_val = std::log2(static_cast<float>(cfg_.num_classes));
        const float epsilon = hoeffding_bound(range_val, cfg_.delta, current->seen);
        const float gain_diff = best.gain - (second_best.valid ? second_best.gain : -std::numeric_limits<float>::infinity());

        if (cfg_.merit_preprune && best.gain <= 0.0f) {
            return current;
        }

        if (!(gain_diff > epsilon || epsilon < cfg_.tie_threshold)) {
            return current;
        }

        current->is_leaf = false;
        current->split_feature = best.feature;
        current->split_boundary_bin = best.boundary_bin;
        current->split_threshold = best.threshold;

        current->left = std::make_unique<Node>(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, current->depth + 1u);
        current->right = std::make_unique<Node>(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, current->depth + 1u);
        current->left->class_counts = best.left_counts;
        current->right->class_counts = best.right_counts;
        current->left->seen = sum_counts(best.left_counts);
        current->right->seen = sum_counts(best.right_counts);
        return current;
    }
}

int HoeffdingTreeModel::predict(const std::vector<float>& features) const {
    if (features.size() != input_dim_) {
        throw std::runtime_error("HoeffdingTreeModel::predict - feature size mismatch");
    }
    if (!root_) {
        return 0;
    }
    return predict_node(root_.get(), features);
}

int HoeffdingTreeModel::predict_node(const Node* node, const std::vector<float>& features) const {
    const Node* current = node;
    while (current && !current->is_leaf) {
        const float value = features[current->split_feature];
        current = (value <= current->split_threshold) ? current->left.get() : current->right.get();
    }
    if (!current) {
        return 0;
    }
    return static_cast<int>(argmax_class(current->class_counts));
}

std::size_t HoeffdingTreeModel::argmax_class(const std::vector<std::uint32_t>& counts) {
    std::size_t best_idx = 0u;
    std::uint32_t best_value = 0u;
    for (std::size_t idx = 0u; idx < counts.size(); ++idx) {
        if (counts[idx] > best_value) {
            best_value = counts[idx];
            best_idx = idx;
        }
    }
    return best_idx;
}

std::size_t HoeffdingTreeModel::node_count() const {
    return node_count_impl(root_.get());
}

std::size_t HoeffdingTreeModel::leaf_count() const {
    return leaf_count_impl(root_.get());
}

std::size_t HoeffdingTreeModel::node_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    std::size_t count = 1u;
    if (!node->is_leaf) {
        count += node_count_impl(node->left.get());
        count += node_count_impl(node->right.get());
    }
    return count;
}

std::size_t HoeffdingTreeModel::leaf_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    if (node->is_leaf) {
        return 1u;
    }
    return leaf_count_impl(node->left.get()) + leaf_count_impl(node->right.get());
}

std::size_t HoeffdingTreeModel::estimate_model_bytes(const Node* node) const {
    if (!node) {
        return 0u;
    }
    const std::size_t bins = cfg_.num_threshold_bins;
    const std::size_t classes = cfg_.num_classes;
    const std::size_t per_node = sizeof(Node)
        + input_dim_ * (bins * classes * sizeof(std::uint32_t) + 2u * sizeof(float))
        + classes * sizeof(std::uint32_t);

    std::size_t total = per_node;
    if (!node->is_leaf) {
        total += estimate_model_bytes(node->left.get());
        total += estimate_model_bytes(node->right.get());
    }
    return total;
}

}  // namespace tree
