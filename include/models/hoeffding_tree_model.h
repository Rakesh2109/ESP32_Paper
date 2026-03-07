#pragma once

#include "models/bnn_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tree {

struct HoeffdingTreeConfig {
    std::size_t num_classes = 2;
    std::size_t num_threshold_bins = 16;
    float delta = 1e-7f;
    float tie_threshold = 0.05f;
    std::size_t grace_period = 32;
    std::size_t min_samples_split = 64;
    std::size_t max_depth = 16;
    bool binary_split = true;
    float max_size_mib = 10.0f;
    std::size_t memory_estimate_period = 100000;
    bool stop_mem_management = false;
    bool merit_preprune = true;
};

class HoeffdingTreeModel {
public:
    struct Node;

    HoeffdingTreeModel(std::size_t input_dim, const HoeffdingTreeConfig& cfg = {});
    ~HoeffdingTreeModel();

    float train_sample(const bnn::Sample& sample, int* pred_before_out = nullptr);
    int predict(const std::vector<float>& features) const;

    std::size_t node_count() const;
    std::size_t leaf_count() const;

private:
    std::size_t input_dim_;
    HoeffdingTreeConfig cfg_;
    std::unique_ptr<Node> root_;

    std::size_t samples_seen_ = 0u;
    bool growth_allowed_ = true;

    Node* update_path(Node* node, const std::vector<float>& features, std::size_t label);
    int predict_node(const Node* node, const std::vector<float>& features) const;

    static std::size_t argmax_class(const std::vector<std::uint32_t>& counts);
    static float hoeffding_bound(float range_val, float confidence, std::size_t n);

    std::size_t node_count_impl(const Node* node) const;
    std::size_t leaf_count_impl(const Node* node) const;
    std::size_t estimate_model_bytes(const Node* node) const;
};

}  // namespace tree
