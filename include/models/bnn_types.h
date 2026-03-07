// Hubara/Courbariaux BinaryNet - common types

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bnn {

struct TrainingConfig {
    float learning_rate = 1e-3f;
    float weight_decay = 1e-4f;
    float clip_value = 1.0f;
    unsigned int seed = 0u;
    std::size_t micro_batch_size = 1u;
};

struct Sample {
    std::vector<float> features;
    std::int32_t label = 0;
};

}  // namespace bnn

