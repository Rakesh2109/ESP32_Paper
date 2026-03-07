#include "models/tm_vanilla_model.h"

#include "models/tm_bitpack_utils.h"
#include "models/tm_vanilla.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tm_model {

struct TmVanillaModel::TMVanillaOpaque {
    TsetlinMachine impl{};
};

TmVanillaModel::TmVanillaModel(std::size_t input_dim, const TmVanillaConfig& cfg)
    : core_(new TMVanillaOpaque()),
      input_dim_(input_dim),
      cfg_(cfg),
      binary_input_(input_dim, 0u),
      packed_words_((input_dim + 31u) >> 5, 0u) {
    if (input_dim_ == 0 || input_dim_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        delete core_;
        throw std::invalid_argument("TM_VANILLA: invalid input_dim");
    }
    const int features = static_cast<int>(input_dim_);
    const std::size_t clauses_req = std::max<std::size_t>(cfg_.clauses, 2u);
    const int clauses = static_cast<int>(std::min<std::size_t>(
        clauses_req, static_cast<std::size_t>(MAX_CLAUSES)));
    if (!tm_init(&core_->impl, features, clauses, cfg_.threshold, cfg_.seed)) {
        delete core_;
        throw std::runtime_error("TM_VANILLA: init failed");
    }
}

TmVanillaModel::~TmVanillaModel() {
    if (core_ != nullptr) {
        tm_free(&core_->impl);
        delete core_;
        core_ = nullptr;
    }
}

void TmVanillaModel::encode_features(const std::vector<float>& features) {
    const std::size_t n = std::min(features.size(), input_dim_);
    for (std::size_t i = 0; i < n; ++i) {
        binary_input_[i] = (features[i] > 0.0f) ? 1u : 0u;
    }
    for (std::size_t i = n; i < input_dim_; ++i) {
        binary_input_[i] = 0u;
    }
}

void TmVanillaModel::pack_msb_bits_to_words(const std::uint8_t* packed_bits, std::size_t nfeat) {
    detail::pack_msb_bits_to_words(packed_bits, nfeat, input_dim_, packed_words_);
}

float TmVanillaModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    encode_features(sample.features);
    const int target = (sample.label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    // tm_update() returns class_sum before automata updates.
    const int score_before = tm_update(&core_->impl, binary_input_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmVanillaModel::predict(const std::vector<float>& features) {
    encode_features(features);
    const int score = tm_score(&core_->impl, binary_input_.data());
    return (score >= 0) ? 1 : 0;
}

float TmVanillaModel::train_packed_bits(const std::uint8_t* packed_bits,
                                        std::size_t nfeat,
                                        std::int32_t label,
                                        int* pred_before_out) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int target = (label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    const int score_before = tm_update_words(&core_->impl, packed_words_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmVanillaModel::predict_packed_bits(const std::uint8_t* packed_bits, std::size_t nfeat) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int score = tm_score_words(&core_->impl, packed_words_.data());
    return (score >= 0) ? 1 : 0;
}

}  // namespace tm_model
