#include "models/hat_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

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

struct SplitCandidate {
    bool valid = false;
    std::size_t feature = 0u;
    std::size_t boundary_bin = 0u;
    float threshold = 0.0f;
    float gain = -std::numeric_limits<float>::infinity();
    std::vector<std::uint32_t> left_counts;
    std::vector<std::uint32_t> right_counts;
};

struct DriftStats {
    std::size_t seen = 0u;
    std::size_t errors = 0u;

    void update(bool is_error) {
        seen += 1u;
        if (is_error) {
            errors += 1u;
        }
    }

    float error_rate() const {
        if (seen == 0u) {
            return 0.0f;
        }
        return static_cast<float>(errors) / static_cast<float>(seen);
    }
};

}  // namespace

struct HatModel::Node {
    bool is_leaf = true;
    std::size_t split_feature = 0u;
    std::size_t split_boundary_bin = 0u;
    float split_threshold = 0.0f;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    std::unique_ptr<Node> alternate;

    std::vector<std::uint32_t> class_counts;
    std::vector<std::vector<std::vector<std::uint32_t>>> feature_bin_class_counts;
    std::vector<float> feature_min;
    std::vector<float> feature_max;

    std::size_t seen = 0u;
    std::size_t depth = 0u;
    std::size_t samples_since_split_eval = 0u;

    DriftStats main_drift;
    DriftStats alternate_drift;

    std::mt19937 rng;

    Node(std::size_t input_dim,
         std::size_t num_classes,
         std::size_t num_bins,
         std::size_t node_depth,
         std::uint32_t seed)
        : class_counts(num_classes, 0u),
          feature_bin_class_counts(
              input_dim,
              std::vector<std::vector<std::uint32_t>>(num_bins,
                                                      std::vector<std::uint32_t>(num_classes, 0u))),
          feature_min(input_dim, std::numeric_limits<float>::infinity()),
          feature_max(input_dim, -std::numeric_limits<float>::infinity()),
          depth(node_depth),
          rng(seed) {}
};

static SplitCandidate best_split_for_feature(const HatModel::Node* node,
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

HatModel::HatModel(std::size_t input_dim, const HatConfig& cfg)
    : input_dim_(input_dim),
      cfg_(cfg),
      root_(std::make_unique<Node>(input_dim, cfg.num_classes, cfg.num_threshold_bins, 0u, cfg.seed)) {
    if (input_dim_ == 0u) {
        throw std::runtime_error("HatModel requires input_dim > 0");
    }
    if (cfg_.num_classes < 2u) {
        throw std::runtime_error("HatModel requires num_classes >= 2");
    }
    if (cfg_.num_threshold_bins < 2u) {
        throw std::runtime_error("HatModel requires num_threshold_bins >= 2");
    }
    if (cfg_.delta <= 0.0f || cfg_.delta >= 1.0f) {
        throw std::runtime_error("HatModel delta must be in (0, 1)");
    }
    if (cfg_.drift_window_threshold == 0u) {
        throw std::runtime_error("HatModel drift_window_threshold must be > 0");
    }
    if (cfg_.switch_significance <= 0.0f || cfg_.switch_significance >= 1.0f) {
        throw std::runtime_error("HatModel switch_significance must be in (0, 1)");
    }
}

HatModel::~HatModel() = default;

float HatModel::hoeffding_bound(float range_val, float confidence, std::size_t n) {
    if (n == 0u) {
        return std::numeric_limits<float>::infinity();
    }
    return std::sqrt((range_val * range_val * std::log(1.0f / confidence)) /
                     (2.0f * static_cast<float>(n)));
}

float HatModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    if (sample.features.size() != input_dim_) {
        throw std::runtime_error("HatModel::train_sample - feature size mismatch");
    }
    if (sample.label < 0 || static_cast<std::size_t>(sample.label) >= cfg_.num_classes) {
        throw std::runtime_error("HatModel::train_sample - label out of range");
    }

    const int pred_before = predict(sample.features);
    const float loss = (pred_before == sample.label) ? 0.0f : 1.0f;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }

    std::uint32_t weight = 1u;
    if (cfg_.bootstrap_sampling) {
        std::poisson_distribution<int> poisson(1.0);
        const int k = poisson(root_->rng);
        if (k <= 0) {
            return loss;
        }
        weight = static_cast<std::uint32_t>(k);
    }

    const std::size_t label = static_cast<std::size_t>(sample.label);
    update_path(root_, sample.features, label, weight, true);

    return loss;
}

HatModel::Node* HatModel::update_path(std::unique_ptr<Node>& node,
                                      const std::vector<float>& features,
                                      std::size_t label,
                                      std::uint32_t weight,
                                      bool allow_restructure) {
    if (!node) {
        return nullptr;
    }

    for (std::uint32_t repeat = 0u; repeat < weight; ++repeat) {
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
    }

    if (!node->is_leaf) {
        const Node* main_child = (features[node->split_feature] <= node->split_threshold)
                                     ? node->left.get()
                                     : node->right.get();
        const int pred_main = predict_node(main_child, features);
        node->main_drift.update(pred_main != static_cast<int>(label));

        if (node->alternate) {
            const int pred_alt = predict_node(node->alternate.get(), features);
            node->alternate_drift.update(pred_alt != static_cast<int>(label));
            update_path(node->alternate, features, label, weight, false);

            const std::size_t n_main = node->main_drift.seen;
            const std::size_t n_alt = node->alternate_drift.seen;
            const std::size_t min_seen = std::min(n_main, n_alt);

            if (min_seen >= cfg_.drift_window_threshold) {
                const float eps = hoeffding_bound(1.0f, cfg_.switch_significance, min_seen);
                const float err_main = node->main_drift.error_rate();
                const float err_alt = node->alternate_drift.error_rate();

                if (err_alt + eps < err_main) {
                    node = std::move(node->alternate);
                    switched_alternates_ += 1u;
                    return update_path(node, features, label, weight, allow_restructure);
                }

                if (err_main + eps < err_alt) {
                    node->alternate.reset();
                    node->alternate_drift = DriftStats{};
                    pruned_alternates_ += 1u;
                }
            }
        } else {
            const std::size_t n = node->main_drift.seen;
            if (n >= cfg_.drift_window_threshold) {
                const float err = node->main_drift.error_rate();
                const float eps = hoeffding_bound(1.0f, cfg_.switch_significance, n);
                if (err > 0.5f + eps) {
                    node->alternate = std::make_unique<Node>(input_dim_,
                                                             cfg_.num_classes,
                                                             cfg_.num_threshold_bins,
                                                             node->depth,
                                                             node->rng());
                    node->alternate_drift = DriftStats{};
                    alternate_trees_ += 1u;
                }
            }
        }

        std::unique_ptr<Node>& child = (features[node->split_feature] <= node->split_threshold)
                                           ? node->left
                                           : node->right;
        return update_path(child, features, label, weight, allow_restructure);
    }

    node->samples_since_split_eval += weight;

    if (!allow_restructure || node->depth >= cfg_.max_depth || node->seen < cfg_.min_samples_split ||
        node->samples_since_split_eval < cfg_.grace_period || is_pure_distribution(node->class_counts)) {
        return node.get();
    }

    node->samples_since_split_eval = 0u;

    SplitCandidate best;
    SplitCandidate second_best;
    for (std::size_t feature_idx = 0u; feature_idx < input_dim_; ++feature_idx) {
        SplitCandidate candidate = best_split_for_feature(node.get(),
                                                          feature_idx,
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
        return node.get();
    }

    const float range_val = std::log2(static_cast<float>(cfg_.num_classes));
    const float eps = hoeffding_bound(range_val, cfg_.delta, node->seen);
    const float gain_diff = best.gain - (second_best.valid ? second_best.gain : 0.0f);

    if (!(gain_diff > eps || eps < cfg_.tie_threshold)) {
        return node.get();
    }

    node->is_leaf = false;
    node->split_feature = best.feature;
    node->split_boundary_bin = best.boundary_bin;
    node->split_threshold = best.threshold;

    node->left = std::make_unique<Node>(input_dim_,
                                        cfg_.num_classes,
                                        cfg_.num_threshold_bins,
                                        node->depth + 1u,
                                        node->rng());
    node->right = std::make_unique<Node>(input_dim_,
                                         cfg_.num_classes,
                                         cfg_.num_threshold_bins,
                                         node->depth + 1u,
                                         node->rng());
    node->left->class_counts = best.left_counts;
    node->right->class_counts = best.right_counts;
    node->left->seen = sum_counts(best.left_counts);
    node->right->seen = sum_counts(best.right_counts);

    return node.get();
}

int HatModel::predict(const std::vector<float>& features) const {
    if (features.size() != input_dim_) {
        throw std::runtime_error("HatModel::predict - feature size mismatch");
    }
    if (!root_) {
        return 0;
    }
    return predict_node(root_.get(), features);
}

int HatModel::predict_node(const Node* node, const std::vector<float>& features) const {
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

std::size_t HatModel::argmax_class(const std::vector<std::uint32_t>& counts) {
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

std::size_t HatModel::node_count() const {
    return node_count_impl(root_.get());
}

std::size_t HatModel::leaf_count() const {
    return leaf_count_impl(root_.get());
}

std::size_t HatModel::alternate_tree_count() const {
    return alternate_trees_;
}

std::size_t HatModel::switched_alternate_count() const {
    return switched_alternates_;
}

std::size_t HatModel::pruned_alternate_count() const {
    return pruned_alternates_;
}

std::size_t HatModel::node_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    std::size_t count = 1u;
    if (!node->is_leaf) {
        count += node_count_impl(node->left.get());
        count += node_count_impl(node->right.get());
    }
    if (node->alternate) {
        count += node_count_impl(node->alternate.get());
    }
    return count;
}

std::size_t HatModel::leaf_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    if (node->is_leaf) {
        return 1u;
    }
    std::size_t leaves = leaf_count_impl(node->left.get()) + leaf_count_impl(node->right.get());
    if (node->alternate) {
        leaves += leaf_count_impl(node->alternate.get());
    }
    return leaves;
}

}  // namespace tree
