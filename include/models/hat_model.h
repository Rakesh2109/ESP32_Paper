#pragma once

#include "models/bnn_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tree {

struct HatConfig {
    std::size_t num_classes = 2;
    std::size_t num_threshold_bins = 16;
    float delta = 1e-7f;
    float tie_threshold = 0.05f;
    std::size_t grace_period = 32;
    std::size_t min_samples_split = 64;
    std::size_t max_depth = 16;

    bool bootstrap_sampling = true;
    std::size_t drift_window_threshold = 300;
    float switch_significance = 0.05f;
    std::uint32_t seed = 0u;
};

class HatModel {
public:
    struct Node;

    HatModel(std::size_t input_dim, const HatConfig& cfg = {});
    ~HatModel();

    float train_sample(const bnn::Sample& sample, int* pred_before_out = nullptr);
    int predict(const std::vector<float>& features) const;

    std::size_t node_count() const;
    std::size_t leaf_count() const;
    std::size_t alternate_tree_count() const;
    std::size_t switched_alternate_count() const;
    std::size_t pruned_alternate_count() const;

private:
    std::size_t input_dim_;
    HatConfig cfg_;
    std::unique_ptr<Node> root_;

    std::size_t alternate_trees_ = 0u;
    std::size_t switched_alternates_ = 0u;
    std::size_t pruned_alternates_ = 0u;

    Node* update_path(std::unique_ptr<Node>& node,
                      const std::vector<float>& features,
                      std::size_t label,
                      std::uint32_t weight,
                      bool allow_restructure);
    int predict_node(const Node* node, const std::vector<float>& features) const;

    static std::size_t argmax_class(const std::vector<std::uint32_t>& counts);
    static float hoeffding_bound(float range_val, float confidence, std::size_t n);

    std::size_t node_count_impl(const Node* node) const;
    std::size_t leaf_count_impl(const Node* node) const;
};

}  // namespace tree
