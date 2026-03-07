#pragma once

#include "models/bnn_types.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tree {

struct EfdtConfig {
    std::size_t num_classes = 2;
    std::size_t num_threshold_bins = 16;
    float delta = 1e-7f;
    float tie_threshold = 0.05f;
    std::size_t grace_period = 32;
    std::size_t min_samples_split = 64;
    std::size_t max_depth = 16;
    std::size_t reevaluate_period = 64;
};

class EfdtModel {
public:
    struct Node;

    EfdtModel(std::size_t input_dim, const EfdtConfig& cfg = {});
    ~EfdtModel();

    float train_sample(const bnn::Sample& sample, int* pred_before_out = nullptr);
    int predict(const std::vector<float>& features) const;

    std::size_t node_count() const;
    std::size_t leaf_count() const;

private:
    std::size_t input_dim_;
    EfdtConfig cfg_;
    std::unique_ptr<Node> root_;

    Node* update_path(Node* node,
                      const std::vector<float>& features,
                      std::size_t label,
                      bool allow_restructure);
    int predict_node(const Node* node, const std::vector<float>& features) const;

    static std::size_t argmax_class(const std::vector<std::uint32_t>& counts);

    std::size_t node_count_impl(const Node* node) const;
    std::size_t leaf_count_impl(const Node* node) const;
};

}  // namespace tree
