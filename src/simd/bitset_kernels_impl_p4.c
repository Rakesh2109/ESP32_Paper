#include "simd/bitset_kernels.h"

// Weak P4 fallback: forwards to scalar backend.
// A vectorized RVV implementation can override these symbols.

__attribute__((weak)) uint32_t BITSET_HOT
bitset_clause_mismatch_u32_p4(const uint32_t* pos,
                              const uint32_t* neg,
                              const uint32_t* x,
                              size_t n_words)
{
    return bitset_clause_mismatch_u32_scalar(pos, neg, x, n_words);
}

__attribute__((weak)) int BITSET_HOT
bitset_clause_holds_u32_p4(const uint32_t* pos,
                           const uint32_t* neg,
                           const uint32_t* x,
                           size_t n_words)
{
    return bitset_clause_holds_u32_scalar(pos, neg, x, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_and_u32_p4(uint32_t* dst,
                  const uint32_t* a,
                  const uint32_t* b,
                  size_t n_words)
{
    bitset_and_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_or_u32_p4(uint32_t* dst,
                 const uint32_t* a,
                 const uint32_t* b,
                 size_t n_words)
{
    bitset_or_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) void BITSET_HOT
bitset_xor_u32_p4(uint32_t* dst,
                  const uint32_t* a,
                  const uint32_t* b,
                  size_t n_words)
{
    bitset_xor_u32_scalar(dst, a, b, n_words);
}

__attribute__((weak)) uint32_t BITSET_HOT
bitset_popcount_u32_p4(const uint32_t* a, size_t n_words)
{
    return bitset_popcount_u32_scalar(a, n_words);
}

