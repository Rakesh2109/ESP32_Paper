#pragma once

#include "models/bnn_types.h"
#include "config/tm_config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tm_model {

struct TmVanillaConfig {
    std::size_t num_classes = TM_NUM_CLASSES;
    std::size_t clauses = TM_C;
    int threshold = TM_T;
    int specificity = TM_S;
    std::uint32_t seed = TM_SEED;
    int init_literal_density_pct = TM_INIT_LITERAL_DENSITY_PCT;
};

class TmVanillaModel {
public:
    TmVanillaModel(std::size_t input_dim, const TmVanillaConfig& cfg = {});
    ~TmVanillaModel();

    float train_sample(const bnn::Sample& sample, int* pred_before_out = nullptr);
    int predict(const std::vector<float>& features);
    float train_packed_bits(const std::uint8_t* packed_bits, std::size_t nfeat, std::int32_t label,
                            int* pred_before_out = nullptr);
    int predict_packed_bits(const std::uint8_t* packed_bits, std::size_t nfeat);

private:
    void encode_features(const std::vector<float>& features);
    void pack_msb_bits_to_words(const std::uint8_t* packed_bits, std::size_t nfeat);

    struct TMVanillaOpaque;
    TMVanillaOpaque* core_;
    std::size_t input_dim_;
    TmVanillaConfig cfg_;
    std::vector<std::uint8_t> binary_input_;
    std::vector<std::uint32_t> packed_words_;
};

}  // namespace tm_model
