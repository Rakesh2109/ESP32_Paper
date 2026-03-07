#include "models/bnn_layer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace bnn {

ScaledBinaryLinear::ScaledBinaryLinear(std::size_t in_dim,
                                       std::size_t out_dim,
                                       bool binarize_weights,
                                       bool binarize_inputs,
                                       bool use_bias,
                                       unsigned int seed)
    : input_dim_(in_dim),
      output_dim_(out_dim),
      binarize_weights_(binarize_weights),
      binarize_inputs_(binarize_inputs),
      use_bias_(use_bias),
      weights_(out_dim * in_dim),
      biases_(use_bias ? std::vector<float>(out_dim, 0.0f) : std::vector<float>{}),
      grad_w_(out_dim * in_dim, 0.0f),
      grad_b_(use_bias ? std::vector<float>(out_dim, 0.0f) : std::vector<float>{}),
      bin_input_(in_dim, 0.0f),
      bin_input_sign_(binarize_inputs ? std::vector<std::int8_t>(in_dim, 0) : std::vector<std::int8_t>{}),
      weight_sign_(binarize_weights ? std::vector<std::int8_t>(out_dim * in_dim, 0)
                                    : std::vector<std::int8_t>{}),
      alpha_(binarize_weights ? std::vector<float>(out_dim, 0.0f)
                              : std::vector<float>{}),
      pre_activation_(out_dim, 0.0f),
      input_cache_(in_dim, 0.0f),
      binary_cache_dirty_(binarize_weights) {
    init_weights(seed);
}

void ScaledBinaryLinear::init_weights(unsigned int seed) {
    std::mt19937 rng(seed);
    const float limit =
        std::sqrt(6.0f / static_cast<float>(input_dim_ + output_dim_));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (auto& w : weights_) {
        w = dist(rng);
    }
    if (use_bias_) {
        std::fill(biases_.begin(), biases_.end(), 0.0f);
    }
}

const std::vector<float>& ScaledBinaryLinear::forward(const std::vector<float>& input) {
    if (input.size() != input_dim_) {
        throw std::runtime_error("ScaledBinaryLinear::forward - input size mismatch");
    }

    input_cache_ = input;

    if (binarize_inputs_) {
        for (std::size_t i = 0; i < input_dim_; ++i) {
            bin_input_sign_[i] = (input[i] >= 0.0f) ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
        }
    } else {
        bin_input_ = input;
    }

    if (binarize_weights_ && binary_cache_dirty_) {
        refresh_binary_cache();
    }

    for (std::size_t o = 0; o < output_dim_; ++o) {
        const std::size_t offset = o * input_dim_;
        float acc = 0.0f;

        if (binarize_weights_) {
            if (binarize_inputs_) {
                int acc_i = 0;
                for (std::size_t i = 0; i < input_dim_; ++i) {
                    acc_i += static_cast<int>(weight_sign_[offset + i]) * static_cast<int>(bin_input_sign_[i]);
                }
                acc = static_cast<float>(acc_i);
            } else {
                for (std::size_t i = 0; i < input_dim_; ++i) {
                    acc += static_cast<float>(weight_sign_[offset + i]) * bin_input_[i];
                }
            }

            acc *= alpha_[o];
        } else {
            for (std::size_t i = 0; i < input_dim_; ++i) {
                acc += weights_[offset + i] * bin_input_[i];
            }
        }

        if (use_bias_) {
            acc += biases_[o];
        }
        
        pre_activation_[o] = acc;
    }

    return pre_activation_;
}

std::vector<float> ScaledBinaryLinear::backward(const std::vector<float>& grad_output) {
    if (grad_output.size() != output_dim_) {
        throw std::runtime_error("ScaledBinaryLinear::backward - grad size mismatch");
    }

    std::vector<float> grad_input(input_dim_, 0.0f);
    for (std::size_t o = 0; o < output_dim_; ++o) {
        const float go = grad_output[o];
        const std::size_t offset = o * input_dim_;

        if (binarize_weights_) {
            const float scaled_grad = go * alpha_[o];
            
            if (binarize_inputs_) {
                for (std::size_t i = 0; i < input_dim_; ++i) {
                    grad_w_[offset + i] += scaled_grad * static_cast<float>(bin_input_sign_[i]);
                    grad_input[i] += scaled_grad * static_cast<float>(weight_sign_[offset + i]);
                }
            } else {
                for (std::size_t i = 0; i < input_dim_; ++i) {
                    grad_w_[offset + i] += scaled_grad * bin_input_[i];
                    grad_input[i] += scaled_grad * static_cast<float>(weight_sign_[offset + i]);
                }
            }
        } else {
            for (std::size_t i = 0; i < input_dim_; ++i) {
                grad_w_[offset + i] += go * bin_input_[i];
                grad_input[i] += go * weights_[offset + i];
            }
        }

        if (use_bias_) {
            grad_b_[o] += go;
        }
    }

    if (binarize_inputs_) {
        for (std::size_t i = 0; i < input_dim_; ++i) {
            if (std::fabs(input_cache_[i]) > 1.0f) {
                grad_input[i] = 0.0f;
            }
        }
    }

    return grad_input;
}

void ScaledBinaryLinear::apply_gradient(const TrainingConfig& cfg) {
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        float grad = grad_w_[i];
        if (cfg.weight_decay > 0.0f) {
            grad += cfg.weight_decay * weights_[i];
        }
        weights_[i] -= cfg.learning_rate * grad;
        if (binarize_weights_) {
            weights_[i] = std::max(-1.0f, std::min(1.0f, weights_[i]));
        }
        grad_w_[i] = 0.0f;
    }

    if (binarize_weights_) {
        binary_cache_dirty_ = true;
    }

    if (use_bias_) {
        for (std::size_t i = 0; i < biases_.size(); ++i) {
            biases_[i] -= cfg.learning_rate * grad_b_[i];
            grad_b_[i] = 0.0f;
        }
    }
}

void ScaledBinaryLinear::refresh_binary_cache() {
    for (std::size_t o = 0; o < output_dim_; ++o) {
        const std::size_t offset = o * input_dim_;
        float alpha_sum = 0.0f;
        for (std::size_t i = 0; i < input_dim_; ++i) {
            const float w = weights_[offset + i];
            alpha_sum += std::fabs(w);
            weight_sign_[offset + i] = (w >= 0.0f) ? static_cast<std::int8_t>(1) : static_cast<std::int8_t>(-1);
        }
        alpha_[o] = alpha_sum / static_cast<float>(input_dim_);
    }
    binary_cache_dirty_ = false;
}

BinaryBatchNorm::BinaryBatchNorm(std::size_t dim, float momentum, float eps)
    : dim_(dim),
      momentum_(momentum),
      eps_(eps),
      gamma_(dim, 1.0f),
      beta_(dim, 0.0f),
      running_mean_(dim, 0.0f),
      running_var_(dim, 1.0f),
      grad_gamma_(dim, 0.0f),
      grad_beta_(dim, 0.0f),
      normalized_(dim, 0.0f),
      inv_std_(dim, 1.0f),
      output_(dim, 0.0f),
      input_cache_(dim, 0.0f),
      shift_amount_(dim, 0) {}

static inline float ap2_quantize(float x) {
    if (x == 0.0f) return 0.0f;
    int exp;
    std::frexp(x, &exp);
    return std::ldexp(1.0f, exp - 1);
}

const std::vector<float>& BinaryBatchNorm::forward(const std::vector<float>& input, bool training) {
    if (input.size() != dim_) {
        throw std::runtime_error("BinaryBatchNorm::forward - dim mismatch");
    }

    input_cache_ = input;

    if (training) {
        for (std::size_t i = 0; i < dim_; ++i) {
            const float x = input[i];
            const float old_mean = running_mean_[i];
            const float old_var = running_var_[i];

            const float diff = x - old_mean;
            const float new_mean = old_mean + momentum_ * diff;
            const float new_var = (1.0f - momentum_) * old_var + momentum_ * diff * diff;

            running_mean_[i] = new_mean;
            running_var_[i] = std::max(new_var, eps_);

            const float inv_std = 1.0f / std::sqrt(running_var_[i] + eps_);
            inv_std_[i] = inv_std;
            normalized_[i] = (x - new_mean) * inv_std;
            
            const float abs_gamma = std::fabs(gamma_[i]);
            if (abs_gamma > eps_) {
                int exp;
                std::frexp(ap2_quantize(gamma_[i]), &exp);
                shift_amount_[i] = exp - 1;
                
                const float sign = (gamma_[i] >= 0.0f) ? 1.0f : -1.0f;
                output_[i] = sign * std::ldexp(normalized_[i], shift_amount_[i]) + beta_[i];
            } else {
                shift_amount_[i] = 0;
                output_[i] = beta_[i];
            }
        }
    } else {
        for (std::size_t i = 0; i < dim_; ++i) {
            const float inv_std = 1.0f / std::sqrt(running_var_[i] + eps_);
            inv_std_[i] = inv_std;
            const float norm = (input[i] - running_mean_[i]) * inv_std;
            normalized_[i] = norm;
            
            const float abs_gamma = std::fabs(gamma_[i]);
            if (abs_gamma > eps_) {
                int exp;
                std::frexp(ap2_quantize(gamma_[i]), &exp);
                shift_amount_[i] = exp - 1;
                
                const float sign = (gamma_[i] >= 0.0f) ? 1.0f : -1.0f;
                output_[i] = sign * std::ldexp(norm, shift_amount_[i]) + beta_[i];
            } else {
                shift_amount_[i] = 0;
                output_[i] = beta_[i];
            }
        }
    }

    return output_;
}

std::vector<float> BinaryBatchNorm::backward(const std::vector<float>& grad_output) {
    if (grad_output.size() != dim_) {
        throw std::runtime_error("BinaryBatchNorm::backward - dim mismatch");
    }

    std::vector<float> grad_input(dim_, 0.0f);
    for (std::size_t i = 0; i < dim_; ++i) {
        grad_gamma_[i] += grad_output[i] * normalized_[i];
        grad_beta_[i] += grad_output[i];
        
        const float inv_std = inv_std_[i];
        grad_input[i] = grad_output[i] * gamma_[i] * inv_std;
    }

    return grad_input;
}

void BinaryBatchNorm::apply_gradient(const TrainingConfig& cfg) {
    for (std::size_t i = 0; i < dim_; ++i) {
        gamma_[i] -= cfg.learning_rate * grad_gamma_[i];
        gamma_[i] = std::max(eps_, gamma_[i]);
        
        beta_[i] -= cfg.learning_rate * grad_beta_[i];
        
        grad_gamma_[i] = 0.0f;
        grad_beta_[i] = 0.0f;
    }
}

void BinaryBatchNorm::reset_state() {
    std::fill(running_mean_.begin(), running_mean_.end(), 0.0f);
    std::fill(running_var_.begin(), running_var_.end(), 1.0f);
    std::fill(grad_gamma_.begin(), grad_gamma_.end(), 0.0f);
    std::fill(grad_beta_.begin(), grad_beta_.end(), 0.0f);
    std::fill(normalized_.begin(), normalized_.end(), 0.0f);
    std::fill(inv_std_.begin(), inv_std_.end(), 1.0f);
    std::fill(output_.begin(), output_.end(), 0.0f);
    std::fill(input_cache_.begin(), input_cache_.end(), 0.0f);
    std::fill(shift_amount_.begin(), shift_amount_.end(), 0);
}

}  // namespace bnn
