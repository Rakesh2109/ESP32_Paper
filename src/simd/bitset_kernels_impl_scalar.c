// src/bitset_kernels_impl_scalar.c
#include "simd/bitset_kernels.h"

#define UNROLL4 for (; i + 4 <= n_words; i += 4)

uint32_t BITSET_HOT
bitset_clause_mismatch_u32_scalar(const uint32_t* pos,
                                  const uint32_t* neg,
                                  const uint32_t* x,
                                  size_t n_words)
{
    uint32_t accum = 0u;
    size_t i = 0;
    UNROLL4 {
        uint32_t x0 = x[i+0], x1 = x[i+1], x2 = x[i+2], x3 = x[i+3];
        accum |= (pos[i+0] & ~x0) | (neg[i+0] & x0);
        accum |= (pos[i+1] & ~x1) | (neg[i+1] & x1);
        accum |= (pos[i+2] & ~x2) | (neg[i+2] & x2);
        accum |= (pos[i+3] & ~x3) | (neg[i+3] & x3);
    }
    for (; i < n_words; ++i) {
        uint32_t xi = x[i];
        accum |= (pos[i] & ~xi) | (neg[i] & xi);
    }
    return accum;
}

int BITSET_HOT
bitset_clause_holds_u32_scalar(const uint32_t* pos,
                               const uint32_t* neg,
                               const uint32_t* x,
                               size_t n_words)
{
    return bitset_clause_mismatch_u32_scalar(pos, neg, x, n_words) == 0u;
}

void BITSET_HOT
bitset_and_u32_scalar(uint32_t* dst,
                      const uint32_t* a,
                      const uint32_t* b,
                      size_t n_words)
{
    size_t i = 0;
    UNROLL4 {
        dst[i+0] = a[i+0] & b[i+0];
        dst[i+1] = a[i+1] & b[i+1];
        dst[i+2] = a[i+2] & b[i+2];
        dst[i+3] = a[i+3] & b[i+3];
    }
    for (; i < n_words; ++i) dst[i] = a[i] & b[i];
}

void BITSET_HOT
bitset_or_u32_scalar(uint32_t* dst,
                     const uint32_t* a,
                     const uint32_t* b,
                     size_t n_words)
{
    size_t i = 0;
    UNROLL4 {
        dst[i+0] = a[i+0] | b[i+0];
        dst[i+1] = a[i+1] | b[i+1];
        dst[i+2] = a[i+2] | b[i+2];
        dst[i+3] = a[i+3] | b[i+3];
    }
    for (; i < n_words; ++i) dst[i] = a[i] | b[i];
}

void BITSET_HOT
bitset_xor_u32_scalar(uint32_t* dst,
                      const uint32_t* a,
                      const uint32_t* b,
                      size_t n_words)
{
    size_t i = 0;
    UNROLL4 {
        dst[i+0] = a[i+0] ^ b[i+0];
        dst[i+1] = a[i+1] ^ b[i+1];
        dst[i+2] = a[i+2] ^ b[i+2];
        dst[i+3] = a[i+3] ^ b[i+3];
    }
    for (; i < n_words; ++i) dst[i] = a[i] ^ b[i];
}

uint32_t BITSET_HOT
bitset_popcount_u32_scalar(const uint32_t* a, size_t n_words)
{
    uint32_t sum = 0;
    size_t i = 0;
    UNROLL4 {
        sum += __builtin_popcount(a[i+0]);
        sum += __builtin_popcount(a[i+1]);
        sum += __builtin_popcount(a[i+2]);
        sum += __builtin_popcount(a[i+3]);
    }
    for (; i < n_words; ++i) sum += __builtin_popcount(a[i]);
    return sum;
}

#undef UNROLL4

