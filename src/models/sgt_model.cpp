#include "models/sgt_model.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tree {

namespace {

struct RunningStats {
    std::size_t count = 0u;
    float mean = 0.0f;
    float m2 = 0.0f;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();

    void update(float value) {
        count += 1u;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
        const float delta = value - mean;
        mean += delta / static_cast<float>(count);
        const float delta2 = value - mean;
        m2 += delta * delta2;
    }

    float variance() const {
        if (count < 2u) {
            return 0.0f;
        }
        return m2 / static_cast<float>(count - 1u);
    }
};

struct SplitCandidate {
    bool valid = false;
    std::size_t feature = 0u;
    std::size_t boundary_bin = 0u;
    float threshold = 0.0f;
    float gain = -std::numeric_limits<float>::infinity();
    float left_grad = 0.0f;
    float left_hess = 0.0f;
    float right_grad = 0.0f;
    float right_hess = 0.0f;
};

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

}  // namespace

struct SgtModel::Node {
    bool is_leaf = true;
    std::size_t split_feature = 0u;
    std::size_t split_boundary_bin = 0u;
    float split_threshold = 0.0f;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;

    float prediction = 0.0f;
    float grad_sum = 0.0f;
    float hess_sum = 0.0f;

    std::vector<std::vector<float>> feature_bin_grad_sums;
    std::vector<std::vector<float>> feature_bin_hess_sums;
    std::vector<RunningStats> feature_stats;

    std::size_t seen = 0u;
    std::size_t depth = 0u;
    std::size_t samples_since_split_eval = 0u;
    std::size_t samples_since_pred_update = 0u;

    Node(std::size_t input_dim, std::size_t num_bins, std::size_t node_depth, float init_pred)
        : prediction(init_pred),
          feature_bin_grad_sums(input_dim, std::vector<float>(num_bins, 0.0f)),
          feature_bin_hess_sums(input_dim, std::vector<float>(num_bins, 0.0f)),
          feature_stats(input_dim),
          depth(node_depth) {}
};

static SplitCandidate best_split_for_feature(const SgtModel::Node* node,
                                             std::size_t feature_idx,
                                             const SgtConfig& cfg) {
    SplitCandidate best;
    best.feature = feature_idx;

    const RunningStats& stats = node->feature_stats[feature_idx];
    if (stats.count < cfg.warm_start) {
        return best;
    }

    const float std = std::sqrt(std::max(0.0f, stats.variance()));
    const float span = cfg.std_prop * std;
    float min_value = stats.min_value;
    float max_value = stats.max_value;

    if (span > 0.0f) {
        const float left = stats.mean - span;
        const float right = stats.mean + span;
        min_value = std::min(min_value, left);
        max_value = std::max(max_value, right);
    }

    if (!(max_value > min_value)) {
        return best;
    }

    float left_grad = 0.0f;
    float left_hess = 0.0f;
    float right_grad = node->grad_sum;
    float right_hess = node->hess_sum;

    for (std::size_t boundary = 0u; boundary + 1u < cfg.num_threshold_bins; ++boundary) {
        left_grad += node->feature_bin_grad_sums[feature_idx][boundary];
        left_hess += node->feature_bin_hess_sums[feature_idx][boundary];
        right_grad -= node->feature_bin_grad_sums[feature_idx][boundary];
        right_hess -= node->feature_bin_hess_sums[feature_idx][boundary];

        if (left_hess <= 0.0f || right_hess <= 0.0f) {
            continue;
        }

        const float gain = 0.5f * ((left_grad * left_grad) / (left_hess + cfg.lambda_value) +
                                   (right_grad * right_grad) / (right_hess + cfg.lambda_value) -
                                   (node->grad_sum * node->grad_sum) / (node->hess_sum + cfg.lambda_value)) -
                           cfg.gamma;

        if (!best.valid || gain > best.gain) {
            best.valid = true;
            best.gain = gain;
            best.boundary_bin = boundary;
            const float ratio = static_cast<float>(boundary + 1u) / static_cast<float>(cfg.num_threshold_bins);
            best.threshold = min_value + (max_value - min_value) * ratio;
            best.left_grad = left_grad;
            best.left_hess = left_hess;
            best.right_grad = right_grad;
            best.right_hess = right_hess;
        }
    }

    return best;
}

SgtModel::SgtModel(std::size_t input_dim, const SgtConfig& cfg)
    : input_dim_(input_dim), cfg_(cfg), root_(std::make_unique<Node>(input_dim, cfg.num_threshold_bins, 0u, cfg.init_pred)) {
    if (input_dim_ == 0u) {
        throw std::runtime_error("SgtModel requires input_dim > 0");
    }
    if (cfg_.num_threshold_bins < 2u) {
        throw std::runtime_error("SgtModel requires num_threshold_bins >= 2");
    }
    if (cfg_.delta <= 0.0f || cfg_.delta >= 1.0f) {
        throw std::runtime_error("SgtModel delta must be in (0, 1)");
    }
    if (cfg_.lambda_value < 0.0f || cfg_.gamma < 0.0f) {
        throw std::runtime_error("SgtModel lambda_value and gamma must be >= 0");
    }
}

SgtModel::~SgtModel() = default;

float SgtModel::sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

float SgtModel::hoeffding_bound(float range_val, float confidence, std::size_t n) {
    if (n == 0u) {
        return std::numeric_limits<float>::infinity();
    }
    return std::sqrt((range_val * range_val * std::log(1.0f / confidence)) /
                     (2.0f * static_cast<float>(n)));
}

float SgtModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    if (sample.features.size() != input_dim_) {
        throw std::runtime_error("SgtModel::train_sample - feature size mismatch");
    }

    const int pred_before = predict(sample.features);
    const float loss = (pred_before == sample.label) ? 0.0f : 1.0f;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }

    const float label = (sample.label > 0) ? 1.0f : 0.0f;
    update_path(root_.get(), sample.features, label, 1.0f);

    return loss;
}

SgtModel::Node* SgtModel::update_path(Node* node,
                                      const std::vector<float>& features,
                                      float label,
                                      float weight) {
    if (!node) {
        return nullptr;
    }

    node->seen += 1u;
    node->samples_since_split_eval += 1u;
    node->samples_since_pred_update += 1u;

    const float pred = sigmoid(node->prediction);
    const float grad = (pred - label) * weight;
    const float hess = pred * (1.0f - pred) * weight;

    node->grad_sum += grad;
    node->hess_sum += hess;

    for (std::size_t i = 0u; i < input_dim_; ++i) {
        const float value = features[i];
        node->feature_stats[i].update(value);

        const RunningStats& stats = node->feature_stats[i];
        float min_value = stats.min_value;
        float max_value = stats.max_value;
        const float std = std::sqrt(std::max(0.0f, stats.variance()));
        const float span = cfg_.std_prop * std;
        if (span > 0.0f) {
            min_value = std::min(min_value, stats.mean - span);
            max_value = std::max(max_value, stats.mean + span);
        }

        const std::size_t bin = safe_bin_index(value, min_value, max_value, cfg_.num_threshold_bins);
        node->feature_bin_grad_sums[i][bin] += grad;
        node->feature_bin_hess_sums[i][bin] += hess;
    }

    if (!node->is_leaf) {
        Node* child = (features[node->split_feature] <= node->split_threshold)
                          ? node->left.get()
                          : node->right.get();
        return update_path(child, features, label, weight);
    }

    if (node->samples_since_pred_update >= cfg_.grace_period && node->hess_sum > 0.0f) {
        const float delta = -node->grad_sum / (node->hess_sum + cfg_.lambda_value);
        node->prediction += delta;
        node->grad_sum = 0.0f;
        node->hess_sum = 0.0f;
        node->samples_since_pred_update = 0u;
    }

    if (node->depth >= cfg_.max_depth || node->seen < cfg_.min_samples_split ||
        node->samples_since_split_eval < cfg_.grace_period) {
        return node;
    }

    node->samples_since_split_eval = 0u;

    SplitCandidate best;
    SplitCandidate second_best;
    for (std::size_t feature_idx = 0u; feature_idx < input_dim_; ++feature_idx) {
        SplitCandidate candidate = best_split_for_feature(node, feature_idx, cfg_);
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

    if (!best.valid || best.gain <= 0.0f) {
        return node;
    }

    const float range_val = 1.0f;
    const float epsilon = hoeffding_bound(range_val, cfg_.delta, node->seen);
    const float gain_diff = best.gain - (second_best.valid ? second_best.gain : 0.0f);

    if (!(gain_diff > epsilon || epsilon < cfg_.tie_threshold)) {
        return node;
    }

    node->is_leaf = false;
    node->split_feature = best.feature;
    node->split_boundary_bin = best.boundary_bin;
    node->split_threshold = best.threshold;

    node->left = std::make_unique<Node>(input_dim_, cfg_.num_threshold_bins, node->depth + 1u, node->prediction);
    node->right = std::make_unique<Node>(input_dim_, cfg_.num_threshold_bins, node->depth + 1u, node->prediction);

    node->left->grad_sum = best.left_grad;
    node->left->hess_sum = best.left_hess;
    node->right->grad_sum = best.right_grad;
    node->right->hess_sum = best.right_hess;

    return node;
}

int SgtModel::predict(const std::vector<float>& features) const {
    if (features.size() != input_dim_) {
        throw std::runtime_error("SgtModel::predict - feature size mismatch");
    }
    if (!root_) {
        return 0;
    }
    return predict_node(root_.get(), features);
}

int SgtModel::predict_node(const Node* node, const std::vector<float>& features) const {
    const Node* current = node;
    while (current && !current->is_leaf) {
        const float value = features[current->split_feature];
        current = (value <= current->split_threshold) ? current->left.get() : current->right.get();
    }
    if (!current) {
        return 0;
    }
    return (sigmoid(current->prediction) >= 0.5f) ? 1 : 0;
}

std::size_t SgtModel::node_count() const {
    return node_count_impl(root_.get());
}

std::size_t SgtModel::leaf_count() const {
    return leaf_count_impl(root_.get());
}

std::size_t SgtModel::node_count_impl(const Node* node) const {
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

std::size_t SgtModel::leaf_count_impl(const Node* node) const {
    if (!node) {
        return 0u;
    }
    if (node->is_leaf) {
        return 1u;
    }
    return leaf_count_impl(node->left.get()) + leaf_count_impl(node->right.get());
}

}  // namespace tree
