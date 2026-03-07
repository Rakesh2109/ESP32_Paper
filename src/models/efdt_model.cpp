#include "models/efdt_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tree {

namespace {

float entropy_from_counts(const std::vector<std::uint32_t>& counts) {
    std::uint32_t total = 0u;
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
        const float p = static_cast<float>(count) * inv_total;
        entropy -= p * std::log2(p);
    }
    return entropy;
}

std::uint32_t sum_counts(const std::vector<std::uint32_t>& counts) {
    std::uint32_t total = 0u;
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

bool is_pure_distribution(const std::vector<std::uint32_t>& counts) {
    std::size_t non_zero = 0u;
    for (std::uint32_t value : counts) {
        if (value > 0u) {
            non_zero += 1u;
            if (non_zero > 1u) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

struct EfdtModel::Node {
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
    std::size_t samples_since_split_eval = 0u;
    std::size_t samples_since_reeval = 0u;

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

static SplitCandidate best_split_for_feature(const EfdtModel::Node* node,
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
        const auto& boundary_counts = node->feature_bin_class_counts[feature_idx][boundary];
        for (std::size_t class_idx = 0u; class_idx < num_classes; ++class_idx) {
            left[class_idx] += boundary_counts[class_idx];
            right[class_idx] -= boundary_counts[class_idx];
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

EfdtModel::EfdtModel(std::size_t input_dim, const EfdtConfig& cfg)
    : input_dim_(input_dim),
      cfg_(cfg),
      root_(std::make_unique<Node>(input_dim, cfg.num_classes, cfg.num_threshold_bins, 0u)) {
    if (input_dim_ == 0u) {
        throw std::runtime_error("EfdtModel requires input_dim > 0");
    }
    if (cfg_.num_classes < 2u) {
        throw std::runtime_error("EfdtModel requires num_classes >= 2");
    }
    if (cfg_.num_threshold_bins < 2u) {
        throw std::runtime_error("EfdtModel requires num_threshold_bins >= 2");
    }
    if (cfg_.delta <= 0.0f || cfg_.delta >= 1.0f) {
        throw std::runtime_error("EfdtModel delta must be in (0, 1)");
    }
    if (cfg_.reevaluate_period == 0u) {
        throw std::runtime_error("EfdtModel reevaluate_period must be > 0");
    }
}

EfdtModel::~EfdtModel() = default;

float EfdtModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    if (sample.features.size() != input_dim_) {
        throw std::runtime_error("EfdtModel::train_sample - feature size mismatch");
    }
    if (sample.label < 0 || static_cast<std::size_t>(sample.label) >= cfg_.num_classes) {
        throw std::runtime_error("EfdtModel::train_sample - label out of range");
    }

    const int pred_before = predict(sample.features);
    const float loss = (pred_before == sample.label) ? 0.0f : 1.0f;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }

    const std::size_t label = static_cast<std::size_t>(sample.label);
    update_path(root_.get(), sample.features, label, true);

    return loss;
}

EfdtModel::Node* EfdtModel::update_path(Node* node,
                                        const std::vector<float>& features,
                                        std::size_t label,
                                        bool allow_restructure) {
    if (!node) {
        return nullptr;
    }

    node->seen += 1u;
    node->class_counts[label] += 1u;

    for (std::size_t i = 0u; i < input_dim_; ++i) {
        const float value = features[i];
        node->feature_min[i] = std::min(node->feature_min[i], value);
        node->feature_max[i] = std::max(node->feature_max[i], value);
        const std::size_t bin = safe_bin_index(value,
                                               node->feature_min[i],
                                               node->feature_max[i],
                                               cfg_.num_threshold_bins);
        node->feature_bin_class_counts[i][bin][label] += 1u;
    }

    if (!node->is_leaf) {
        node->samples_since_reeval += 1u;

        if (allow_restructure && node->samples_since_reeval >= cfg_.reevaluate_period && !is_pure_distribution(node->class_counts)) {
            node->samples_since_reeval = 0u;

            SplitCandidate best;
            std::vector<SplitCandidate> by_feature;
            by_feature.reserve(input_dim_);

            for (std::size_t feature_idx = 0u; feature_idx < input_dim_; ++feature_idx) {
                SplitCandidate candidate = best_split_for_feature(node,
                                                                  feature_idx,
                                                                  cfg_.num_threshold_bins,
                                                                  cfg_.num_classes);
                if (!candidate.valid) {
                    continue;
                }
                if (!best.valid || candidate.gain > best.gain) {
                    best = candidate;
                }
                by_feature.push_back(std::move(candidate));
            }

            if (best.valid) {
                SplitCandidate current_best;
                for (const auto& candidate : by_feature) {
                    if (candidate.feature == node->split_feature) {
                        current_best = candidate;
                        break;
                    }
                }

                const float range_val = std::log2(static_cast<float>(cfg_.num_classes));
                const float epsilon = std::sqrt((range_val * range_val * std::log(1.0f / cfg_.delta)) /
                                                (2.0f * static_cast<float>(node->seen)));

                if (0.0f - best.gain > epsilon) {
                    const std::size_t keep_depth = node->depth;
                    Node replacement(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, keep_depth);
                    replacement.class_counts = node->class_counts;
                    replacement.seen = sum_counts(replacement.class_counts);
                    *node = std::move(replacement);
                } else if (current_best.valid) {
                    const float merit_gap = best.gain - current_best.gain;
                    const bool should_replace = (merit_gap > epsilon || epsilon < cfg_.tie_threshold);

                    if (should_replace && best.feature != node->split_feature) {
                        const std::size_t keep_depth = node->depth;
                        Node replacement(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, keep_depth);
                        replacement.is_leaf = false;
                        replacement.split_feature = best.feature;
                        replacement.split_boundary_bin = best.boundary_bin;
                        replacement.split_threshold = best.threshold;
                        replacement.class_counts = node->class_counts;
                        replacement.seen = sum_counts(replacement.class_counts);
                        replacement.samples_since_reeval = 0u;

                        replacement.left = std::make_unique<Node>(input_dim_,
                                                                  cfg_.num_classes,
                                                                  cfg_.num_threshold_bins,
                                                                  keep_depth + 1u);
                        replacement.right = std::make_unique<Node>(input_dim_,
                                                                   cfg_.num_classes,
                                                                   cfg_.num_threshold_bins,
                                                                   keep_depth + 1u);
                        replacement.left->class_counts = best.left_counts;
                        replacement.right->class_counts = best.right_counts;
                        replacement.left->seen = sum_counts(best.left_counts);
                        replacement.right->seen = sum_counts(best.right_counts);

                        *node = std::move(replacement);
                    } else if (should_replace && best.feature == node->split_feature) {
                        node->split_boundary_bin = best.boundary_bin;
                        node->split_threshold = best.threshold;
                    }
                }
            }
        }

        if (!node->is_leaf) {
            Node* child = (features[node->split_feature] <= node->split_threshold)
                              ? node->left.get()
                              : node->right.get();
            return update_path(child, features, label, allow_restructure);
        }
    }

    node->samples_since_split_eval += 1u;

    if (!allow_restructure || node->depth >= cfg_.max_depth || node->seen < cfg_.min_samples_split ||
        node->samples_since_split_eval < cfg_.grace_period || is_pure_distribution(node->class_counts)) {
        return node;
    }

    node->samples_since_split_eval = 0u;

    SplitCandidate best;
    for (std::size_t feature_idx = 0u; feature_idx < input_dim_; ++feature_idx) {
        SplitCandidate candidate = best_split_for_feature(node,
                                                          feature_idx,
                                                          cfg_.num_threshold_bins,
                                                          cfg_.num_classes);
        if (!candidate.valid) {
            continue;
        }
        if (!best.valid || candidate.gain > best.gain) {
            best = std::move(candidate);
        }
    }

    if (!best.valid) {
        return node;
    }

    const float range_val = std::log2(static_cast<float>(cfg_.num_classes));
    const float epsilon = std::sqrt((range_val * range_val * std::log(1.0f / cfg_.delta)) /
                                    (2.0f * static_cast<float>(node->seen)));

    if (!(best.gain > epsilon || epsilon < cfg_.tie_threshold)) {
        return node;
    }

    node->is_leaf = false;
    node->split_feature = best.feature;
    node->split_boundary_bin = best.boundary_bin;
    node->split_threshold = best.threshold;
    node->samples_since_reeval = 0u;

    node->left = std::make_unique<Node>(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, node->depth + 1u);
    node->right = std::make_unique<Node>(input_dim_, cfg_.num_classes, cfg_.num_threshold_bins, node->depth + 1u);
    node->left->class_counts = best.left_counts;
    node->right->class_counts = best.right_counts;
    node->left->seen = sum_counts(best.left_counts);
    node->right->seen = sum_counts(best.right_counts);

    return node;
}

int EfdtModel::predict(const std::vector<float>& features) const {
    if (features.size() != input_dim_) {
        throw std::runtime_error("EfdtModel::predict - feature size mismatch");
    }
    if (!root_) {
        return 0;
    }
    return predict_node(root_.get(), features);
}

int EfdtModel::predict_node(const Node* node, const std::vector<float>& features) const {
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

std::size_t EfdtModel::argmax_class(const std::vector<std::uint32_t>& counts) {
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

std::size_t EfdtModel::node_count() const {
    return node_count_impl(root_.get());
}

std::size_t EfdtModel::leaf_count() const {
    return leaf_count_impl(root_.get());
}

std::size_t EfdtModel::node_count_impl(const Node* node) const {
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

std::size_t EfdtModel::leaf_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    if (node->is_leaf) {
        return 1u;
    }
    return leaf_count_impl(node->left.get()) + leaf_count_impl(node->right.get());
}

}  // namespace tree
