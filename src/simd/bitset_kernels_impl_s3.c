// src/bitset_kernels_impl_s3.c
#include "simd/bitset_kernels.h"

// NOTE:
// All functions here are declared WEAK and simply forward to scalar.
// On ESP32-S3, you can provide strong vectorized versions (same signatures)
// in another compilation unit (e.g., bitset_kernels_impl_s3.S or .c) and
// they will override these.

__attribute__((weak)) uint32_t BITSET_HOT
bitset_clause_mismatch_u32_s3(const uint32_t* pos,
                              const uint32_t* neg,
                              const uint32_t* x,
                              size_t n_words)
{
    return bitset_clause_mismatch_u32_scalar(pos, neg, x, n_words);
}

__attribute__((weak)) int BITSET_HOT
bitset_clause_holds_u32_s3(const uint32_t* pos,
                           const uint32_t* neg,
                           const uint32_t* x,
                           size_t n_words)
{
    return bitset_clause_holds_u32_scalar(pos, neg, x, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_and_u32_s3(uint32_t* dst,
                  const uint32_t* a,
                  const uint32_t* b,
                  size_t n_words)
{
    bitset_and_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_or_u32_s3(uint32_t* dst,
                 const uint32_t* a,
                 const uint32_t* b,
                 size_t n_words)
{
    bitset_or_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_xor_u32_s3(uint32_t* dst,
                  const uint32_t* a,
                  const uint32_t* b,
                  size_t n_words)
{
    bitset_xor_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) uint32_t BITSET_HOT
bitset_popcount_u32_s3(const uint32_t* a, size_t n_words)
{
    return bitset_popcount_u32_scalar(a, n_words);
}

