#pragma once

#include "models/bnn_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tree {

struct SgtConfig {
    std::size_t num_threshold_bins = 16;
    std::size_t num_classes = 2;  // Binary classifier
    float delta = 1e-7f;
    float tie_threshold = 0.05f;
    std::size_t grace_period = 200;  // Match Python River implementation
    std::size_t min_samples_split = 64;
    std::size_t max_depth = 16;

    float init_pred = 0.0f;
    float lambda_value = 0.1f;
    float gamma = 1.0f;

    float std_prop = 1.0f;
    std::size_t warm_start = 20;
};

class SgtModel {
public:
    struct Node;

    SgtModel(std::size_t input_dim, const SgtConfig& cfg = {});
    ~SgtModel();

    float train_sample(const bnn::Sample& sample, int* pred_before_out = nullptr);
    int predict(const std::vector<float>& features) const;

    std::size_t node_count() const;
    std::size_t leaf_count() const;

private:
    std::size_t input_dim_;
    SgtConfig cfg_;
    std::unique_ptr<Node> root_;

    Node* update_path(Node* node, const std::vector<float>& features, float label, float weight);
    int predict_node(const Node* node, const std::vector<float>& features) const;

    static float sigmoid(float x);
    static float hoeffding_bound(float range_val, float confidence, std::size_t n);

    std::size_t node_count_impl(const Node* node) const;
    std::size_t leaf_count_impl(const Node* node) const;
};

}  // namespace tree
