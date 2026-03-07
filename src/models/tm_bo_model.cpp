#include "models/tm_bo_model.h"

#include "models/tm_bitpack_utils.h"
#include "models/tm_bo.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tm_model {

struct TmBoModel::TMBoOpaque {
    TM_BO impl{};
};

TmBoModel::TmBoModel(std::size_t input_dim, const TmBoConfig& cfg)
    : core_(new TMBoOpaque()),
      input_dim_(input_dim),
      cfg_(cfg),
      binary_input_(input_dim, 0u),
      packed_words_((input_dim + 31u) >> 5, 0u) {
    if (input_dim_ == 0 || input_dim_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        delete core_;
        throw std::invalid_argument("TM_BO: invalid input_dim");
    }
    const int features = static_cast<int>(input_dim_);
    if (!tm_bo_init(&core_->impl, features, cfg_.threshold, cfg_.seed)) {
        delete core_;
        throw std::runtime_error("TM_BO: init failed");
    }
    tm_bo_set_init_literal_density(&core_->impl, cfg_.init_literal_density_pct);

    const std::size_t clause_count = std::max<std::size_t>(cfg_.clauses, 2u);
    for (std::size_t j = 0; j < clause_count; ++j) {
        const int8_t sign = (j & 1u) ? -1 : +1;
        if (!tm_bo_clause_add_hot(&core_->impl, sign)) {
            tm_bo_free(&core_->impl);
            delete core_;
            throw std::runtime_error("TM_BO: clause allocation failed");
        }
    }
}

TmBoModel::~TmBoModel() {
    if (core_ != nullptr) {
        tm_bo_free(&core_->impl);
        delete core_;
        core_ = nullptr;
    }
}

void TmBoModel::encode_features(const std::vector<float>& features) {
    const std::size_t n = std::min(features.size(), input_dim_);
    for (std::size_t i = 0; i < n; ++i) {
        binary_input_[i] = (features[i] > 0.0f) ? 1u : 0u;
    }
    for (std::size_t i = n; i < input_dim_; ++i) {
        binary_input_[i] = 0u;
    }
}

void TmBoModel::pack_msb_bits_to_words(const std::uint8_t* packed_bits, std::size_t nfeat) {
    detail::pack_msb_bits_to_words(packed_bits, nfeat, input_dim_, packed_words_);
}

float TmBoModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    encode_features(sample.features);
    const int target = (sample.label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    // tm_bo_update() returns class_sum before automata updates.
    const int score_before = tm_bo_update(&core_->impl, binary_input_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmBoModel::predict(const std::vector<float>& features) {
    encode_features(features);
    const int score = tm_bo_score(&core_->impl, binary_input_.data());
    return (score >= 0) ? 1 : 0;
}

float TmBoModel::train_packed_bits(const std::uint8_t* packed_bits,
                                   std::size_t nfeat,
                                   std::int32_t label,
                                   int* pred_before_out) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int target = (label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    const int score_before = tm_bo_update_words(&core_->impl, packed_words_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmBoModel::predict_packed_bits(const std::uint8_t* packed_bits, std::size_t nfeat) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int score = tm_bo_score_words(&core_->impl, packed_words_.data());
    return (score >= 0) ? 1 : 0;
}

}  // namespace tm_model
