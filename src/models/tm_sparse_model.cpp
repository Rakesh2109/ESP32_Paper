#include "models/tm_sparse_model.h"

#include "models/tm_bitpack_utils.h"
#include "models/tm_sparse.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tm_model {

struct TmSparseModel::TMSparseOpaque {
    TMSparse impl{};
};

TmSparseModel::TmSparseModel(std::size_t input_dim, const TmSparseConfig& cfg)
    : core_(new TMSparseOpaque()),
      input_dim_(input_dim),
      cfg_(cfg),
      binary_input_(input_dim, 0u),
      packed_words_((input_dim + 31u) >> 5, 0u) {
    if (input_dim_ == 0 || input_dim_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        delete core_;
        throw std::invalid_argument("TM_SPARSE: invalid input_dim");
    }
    const int features = static_cast<int>(input_dim_);
    if (!tm_sparse_init(&core_->impl, features, cfg_.threshold, cfg_.seed)) {
        delete core_;
        throw std::runtime_error("TM_SPARSE: init failed");
    }
    tm_sparse_set_init_literal_density(&core_->impl, cfg_.init_literal_density_pct);

    const std::size_t clause_count = std::max<std::size_t>(cfg_.clauses, 2u);
    for (std::size_t j = 0; j < clause_count; ++j) {
        const int8_t sign = (j & 1u) ? -1 : +1;
        if (!tm_sparse_clause_add_hot(&core_->impl, sign)) {
            tm_sparse_free(&core_->impl);
            delete core_;
            throw std::runtime_error("TM_SPARSE: clause allocation failed");
        }
    }
}

TmSparseModel::~TmSparseModel() {
    if (core_ != nullptr) {
        tm_sparse_free(&core_->impl);
        delete core_;
        core_ = nullptr;
    }
}

void TmSparseModel::encode_features(const std::vector<float>& features) {
    const std::size_t n = std::min(features.size(), input_dim_);
    for (std::size_t i = 0; i < n; ++i) {
        binary_input_[i] = (features[i] > 0.0f) ? 1u : 0u;
    }
    for (std::size_t i = n; i < input_dim_; ++i) {
        binary_input_[i] = 0u;
    }
}

void TmSparseModel::pack_msb_bits_to_words(const std::uint8_t* packed_bits, std::size_t nfeat) {
    detail::pack_msb_bits_to_words(packed_bits, nfeat, input_dim_, packed_words_);
}

float TmSparseModel::train_sample(const bnn::Sample& sample, int* pred_before_out) {
    encode_features(sample.features);
    const int target = (sample.label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    // tm_sparse_update() returns class_sum before automata updates.
    const int score_before = tm_sparse_update(&core_->impl, binary_input_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmSparseModel::predict(const std::vector<float>& features) {
    encode_features(features);
    const int score = tm_sparse_score(&core_->impl, binary_input_.data());
    return (score >= 0) ? 1 : 0;
}

float TmSparseModel::train_packed_bits(const std::uint8_t* packed_bits,
                                       std::size_t nfeat,
                                       std::int32_t label,
                                       int* pred_before_out) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int target = (label > 0) ? 1 : 0;
    const int s = std::max(cfg_.specificity, 2);
    const int score_before = tm_sparse_update_words(&core_->impl, packed_words_.data(), target, s);
    const int pred_before = (score_before >= 0) ? 1 : 0;
    if (pred_before_out) {
        *pred_before_out = pred_before;
    }
    return (pred_before == target) ? 0.0f : 1.0f;
}

int TmSparseModel::predict_packed_bits(const std::uint8_t* packed_bits, std::size_t nfeat) {
    pack_msb_bits_to_words(packed_bits, nfeat);
    const int score = tm_sparse_score_words(&core_->impl, packed_words_.data());
    return (score >= 0) ? 1 : 0;
}

}  // namespace tm_model
