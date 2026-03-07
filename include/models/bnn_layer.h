#pragma once

#include "models/bnn_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef BNN_USE_BATCHNORM
#define BNN_USE_BATCHNORM 1
#endif

#ifndef BNN_USE_STE_CLIP
#define BNN_USE_STE_CLIP 1
#endif

namespace bnn {

class ScaledBinaryLinear {
public:
    ScaledBinaryLinear(std::size_t in_dim,
                       std::size_t out_dim,
                       bool binarize_weights,
                       bool binarize_inputs,
                       bool use_bias,
                       unsigned int seed);

    const std::vector<float>& forward(const std::vector<float>& input);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void apply_gradient(const TrainingConfig& cfg);

private:
    std::size_t input_dim_;
    std::size_t output_dim_;
    bool binarize_weights_;
    bool binarize_inputs_;
    bool use_bias_;

    std::vector<float> weights_;
    std::vector<float> biases_;
    std::vector<float> grad_w_;
    std::vector<float> grad_b_;

    std::vector<float> bin_input_;
    std::vector<std::int8_t> bin_input_sign_;
    std::vector<std::int8_t> weight_sign_;
    std::vector<float> alpha_;
    std::vector<float> pre_activation_;
    std::vector<float> input_cache_;
    bool binary_cache_dirty_;

    void init_weights(unsigned int seed);
    void refresh_binary_cache();
};

class BinaryBatchNorm {
public:
    BinaryBatchNorm(std::size_t dim, float momentum = 0.1f, float eps = 1e-5f);

    const std::vector<float>& forward(const std::vector<float>& input, bool training);
    std::vector<float> backward(const std::vector<float>& grad_output);
    void apply_gradient(const TrainingConfig& cfg);
    void reset_state();
    
    const std::vector<float>& get_output() const { return output_; }

private:
    std::size_t dim_;
    float momentum_;
    float eps_;

    std::vector<float> gamma_;
    std::vector<float> beta_;
    std::vector<float> running_mean_;
    std::vector<float> running_var_;
    
    std::vector<float> grad_gamma_;
    std::vector<float> grad_beta_;
    
    std::vector<float> normalized_;
    std::vector<float> inv_std_;
    std::vector<float> output_;
    std::vector<float> input_cache_;
    
    std::vector<int> shift_amount_;
};

}  // namespace bnn
