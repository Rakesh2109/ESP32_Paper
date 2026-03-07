#include "models/bnn_model.h"
#include "config/bnn_config.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bnn {

BinaryNeuralNetwork::BinaryNeuralNetwork(std::size_t input_dim,
                                         const std::vector<std::size_t>& hidden_dims,
                                         std::size_t num_classes,
                                         const TrainingConfig& cfg)
    : cfg_(cfg),
      output_layer_(hidden_dims.empty() ? input_dim : hidden_dims.back(),
                    num_classes,
                    /*binarize_weights=*/false,
                    /*binarize_inputs=*/false,
                    /*use_bias=*/true,
                    cfg.seed + 1337u) {
    if (hidden_dims.empty()) {
        throw std::runtime_error("BinaryNeuralNetwork requires at least one hidden layer");
    }

    std::size_t prev_dim = input_dim;
    unsigned int layer_seed = cfg.seed;
    hidden_layers_.reserve(hidden_dims.size());
    hidden_batch_norms_.reserve(hidden_dims.size());

    for (std::size_t idx = 0; idx < hidden_dims.size(); ++idx) {
        const bool bin_weights = true;
        const bool bin_inputs = true;
        hidden_layers_.emplace_back(prev_dim,
                                    hidden_dims[idx],
                                    bin_weights,
                                    bin_inputs,
                                    /*use_bias=*/false,
                                    layer_seed);
#if BNN_USE_BATCHNORM
        hidden_batch_norms_.emplace_back(hidden_dims[idx]);
#endif
        prev_dim = hidden_dims[idx];
        layer_seed += 191u;
    }

    binary_activations_.resize(hidden_layers_.size());
    grad_logits_cache_.assign(num_classes, 0.0f);
}

float BinaryNeuralNetwork::train_sample(const Sample& sample) {
    const auto& logits = forward(sample.features, true);
    const float loss = softmax_loss_and_gradient(logits, sample.label, grad_logits_cache_);
    backward(grad_logits_cache_);

    grad_accumulated_ += 1u;
    const std::size_t mb = (cfg_.micro_batch_size == 0u) ? 1u : cfg_.micro_batch_size;
    if (grad_accumulated_ >= mb) {
        apply_pending_updates();
    }

    return loss;
}

int BinaryNeuralNetwork::predict(const std::vector<float>& features) {
    apply_pending_updates();
    const auto& logits = forward(features, false);
    return static_cast<int>(std::distance(logits.begin(),
                                          std::max_element(logits.begin(), logits.end())));
}

const std::vector<float>& BinaryNeuralNetwork::forward(const std::vector<float>& input,
                                                      bool training) {
    const std::vector<float>* current = &input;
    for (std::size_t idx = 0; idx < hidden_layers_.size(); ++idx) {
        const auto& pre = hidden_layers_[idx].forward(*current);
#if BNN_USE_BATCHNORM
        const auto& bn_out = hidden_batch_norms_[idx].forward(pre, training);
        const auto& src = bn_out;
#else
        (void)training;
        const auto& src = pre;
#endif
        auto& activations = binary_activations_[idx];
        activations.resize(src.size());

        for (std::size_t i = 0; i < src.size(); ++i) {
            activations[i] = (src[i] >= 0.0f) ? 1.0f : -1.0f;
        }
        current = &activations;
    }

    return output_layer_.forward(*current);
}

void BinaryNeuralNetwork::backward(const std::vector<float>& grad_logits) {
    std::vector<float> grad = output_layer_.backward(grad_logits);
    
    for (std::size_t idx = hidden_layers_.size(); idx-- > 0;) {
#if BNN_USE_BATCHNORM
#if BNN_USE_STE_CLIP
        const auto& bn_output = hidden_batch_norms_[idx].get_output();
        for (std::size_t i = 0; i < grad.size(); ++i) {
            if (std::fabs(bn_output[i]) > 1.0f) {
                grad[i] = 0.0f;
            }
        }
#endif
        grad = hidden_batch_norms_[idx].backward(grad);
#endif
        grad = hidden_layers_[idx].backward(grad);
    }
}

void BinaryNeuralNetwork::apply_pending_updates() {
    if (grad_accumulated_ == 0u) {
        return;
    }
    for (std::size_t idx = 0; idx < hidden_layers_.size(); ++idx) {
        hidden_layers_[idx].apply_gradient(cfg_);
#if BNN_USE_BATCHNORM
        hidden_batch_norms_[idx].apply_gradient(cfg_);
#endif
    }
    output_layer_.apply_gradient(cfg_);
    grad_accumulated_ = 0u;
}

float BinaryNeuralNetwork::softmax_cross_entropy(const std::vector<float>& logits, int label) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (float logit : logits) {
        sum += std::exp(logit - max_logit);
    }
    const float log_prob = logits[label] - max_logit - std::log(sum);
    return -log_prob;
}

std::vector<float> BinaryNeuralNetwork::softmax_gradient(const std::vector<float>& logits,
                                                         int label) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size(), 0.0f);
    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) {
        p /= sum;
    }
    probs[label] -= 1.0f;
    return probs;
}

float BinaryNeuralNetwork::softmax_loss_and_gradient(const std::vector<float>& logits,
                                                     int label,
                                                     std::vector<float>& grad_out) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    if (grad_out.size() != logits.size()) {
        grad_out.assign(logits.size(), 0.0f);
    }

    float sum = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float e = std::exp(logits[i] - max_logit);
        grad_out[i] = e;
        sum += e;
    }

    const float inv_sum = 1.0f / sum;
    for (std::size_t i = 0; i < grad_out.size(); ++i) {
        grad_out[i] *= inv_sum;
    }
    grad_out[label] -= 1.0f;

    const float log_prob = logits[label] - max_logit - std::log(sum);
    return -log_prob;
}

}  // namespace bnn
