#pragma once

#include "models/bnn_layer.h"

#include <cstdint>
#include <vector>

namespace bnn {

class BinaryNeuralNetwork {
public:
    BinaryNeuralNetwork(std::size_t input_dim,
                        const std::vector<std::size_t>& hidden_dims,
                        std::size_t num_classes,
                        const TrainingConfig& cfg);

    float train_sample(const Sample& sample);
    int predict(const std::vector<float>& features);

private:
    TrainingConfig cfg_;
    std::vector<ScaledBinaryLinear> hidden_layers_;
    std::vector<BinaryBatchNorm> hidden_batch_norms_;
    ScaledBinaryLinear output_layer_;
    std::vector<std::vector<float>> binary_activations_;
    std::vector<float> grad_logits_cache_;
    std::size_t grad_accumulated_ = 0u;

    const std::vector<float>& forward(const std::vector<float>& input, bool training);
    void backward(const std::vector<float>& grad_logits);
    void apply_pending_updates();

    static float softmax_cross_entropy(const std::vector<float>& logits, int label);
    static std::vector<float> softmax_gradient(const std::vector<float>& logits, int label);
    static float softmax_loss_and_gradient(const std::vector<float>& logits, int label,
                                           std::vector<float>& grad_out);
};

}  // namespace bnn
