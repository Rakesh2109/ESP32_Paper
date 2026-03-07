// include/bitset_kernels.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*===========================================================================
  Configuration & target detection
===========================================================================*/
#if !defined(BITSET_IDF_HAS_CONFIG)
#  if defined(__has_include)
#    if __has_include("sdkconfig.h")
#      include "sdkconfig.h"
#      define BITSET_IDF_HAS_CONFIG 1
#    endif
#  endif
#endif

#if !defined(BITSET_IS_ESP_PLATFORM)
#  if defined(ESP_PLATFORM)
#    define BITSET_IS_ESP_PLATFORM 1
#  else
#    define BITSET_IS_ESP_PLATFORM 0
#  endif
#endif

#if BITSET_IS_ESP_PLATFORM
#  include <esp_attr.h>
#  ifndef BITSET_HOT
#    define BITSET_HOT IRAM_ATTR
#  endif
#else
#  ifndef BITSET_HOT
#    define BITSET_HOT
#  endif
#endif

#ifndef BITSET_ALWAYS_INLINE
#  define BITSET_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif

// Manual overrides
#ifndef BITSET_FORCE_SCALAR
#  define BITSET_FORCE_SCALAR 0
#endif
#ifndef BITSET_FORCE_S3_VEC
#  define BITSET_FORCE_S3_VEC 0
#endif
#ifndef BITSET_FORCE_P4_VEC
#  define BITSET_FORCE_P4_VEC 0
#endif

// Auto-select backends if not forced
#if BITSET_FORCE_SCALAR
#  define BITSET_ENABLE_SCALAR 1
#  define BITSET_ENABLE_S3_VEC 0
#  define BITSET_ENABLE_P4_VEC 0
#else
#  if BITSET_FORCE_S3_VEC
#    define BITSET_ENABLE_S3_VEC 1
#    define BITSET_ENABLE_P4_VEC 0
#  elif BITSET_FORCE_P4_VEC
#    define BITSET_ENABLE_S3_VEC 0
#    define BITSET_ENABLE_P4_VEC 1
#  else
#    if BITSET_IDF_HAS_CONFIG && defined(CONFIG_IDF_TARGET_ESP32S3)
#      define BITSET_ENABLE_S3_VEC 1
#      define BITSET_ENABLE_P4_VEC 0
#    elif BITSET_IDF_HAS_CONFIG && defined(CONFIG_IDF_TARGET_ESP32P4)
#      define BITSET_ENABLE_S3_VEC 0
#      define BITSET_ENABLE_P4_VEC 1
#    else
#      define BITSET_ENABLE_S3_VEC 0
#      define BITSET_ENABLE_P4_VEC 0
#    endif
#  endif
#endif

/*===========================================================================
  Public API (stable): clause mismatch + bulk ops + popcount
===========================================================================*/
typedef enum bitset_backend_e {
    BITSET_BACKEND_SCALAR = 0,
    BITSET_BACKEND_S3_VEC = 1,
    BITSET_BACKEND_P4_VEC = 2
} bitset_backend_t;

static BITSET_ALWAYS_INLINE bitset_backend_t bitset_selected_backend(void) {
#if BITSET_ENABLE_P4_VEC
    return BITSET_BACKEND_P4_VEC;
#elif BITSET_ENABLE_S3_VEC
    return BITSET_BACKEND_S3_VEC;
#else
    return BITSET_BACKEND_SCALAR;
#endif
}

static uint32_t bitset_clause_mismatch_u32(const uint32_t* pos,
                                           const uint32_t* neg,
                                           const uint32_t* x,
                                           size_t n_words);

static int bitset_clause_holds_u32(const uint32_t* pos,
                                   const uint32_t* neg,
                                   const uint32_t* x,
                                   size_t n_words);

static void bitset_and_u32(uint32_t* dst,
                           const uint32_t* a,
                           const uint32_t* b,
                           size_t n_words);

static void bitset_or_u32(uint32_t* dst,
                          const uint32_t* a,
                          const uint32_t* b,
                          size_t n_words);

static void bitset_xor_u32(uint32_t* dst,
                           const uint32_t* a,
                           const uint32_t* b,
                           size_t n_words);

static uint32_t bitset_popcount_u32(const uint32_t* a, size_t n_words);

/*===========================================================================
  Backend symbol declarations
===========================================================================*/
// Scalar
uint32_t bitset_clause_mismatch_u32_scalar(const uint32_t* pos,
                                           const uint32_t* neg,
                                           const uint32_t* x,
                                           size_t n_words);
int      bitset_clause_holds_u32_scalar  (const uint32_t* pos,
                                          const uint32_t* neg,
                                          const uint32_t* x,
                                          size_t n_words);
void     bitset_and_u32_scalar           (uint32_t* dst,
                                          const uint32_t* a,
                                          const uint32_t* b,
                                          size_t n_words);
void     bitset_or_u32_scalar            (uint32_t* dst,
                                          const uint32_t* a,
                                          const uint32_t* b,
                                          size_t n_words);
void     bitset_xor_u32_scalar           (uint32_t* dst,
                                          const uint32_t* a,
                                          const uint32_t* b,
                                          size_t n_words);
uint32_t bitset_popcount_u32_scalar      (const uint32_t* a, size_t n_words);

// ESP32-S3 (Xtensa) – weak defaults provided
uint32_t bitset_clause_mismatch_u32_s3(const uint32_t* pos,
                                       const uint32_t* neg,
                                       const uint32_t* x,
                                       size_t n_words);
int      bitset_clause_holds_u32_s3    (const uint32_t* pos,
                                        const uint32_t* neg,
                                        const uint32_t* x,
                                        size_t n_words);
void     bitset_and_u32_s3             (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
void     bitset_or_u32_s3              (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
void     bitset_xor_u32_s3             (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
uint32_t bitset_popcount_u32_s3        (const uint32_t* a, size_t n_words);

// ESP32-P4 (RISC-V) – RVV impl or weak fallback
uint32_t bitset_clause_mismatch_u32_p4(const uint32_t* pos,
                                       const uint32_t* neg,
                                       const uint32_t* x,
                                       size_t n_words);
int      bitset_clause_holds_u32_p4    (const uint32_t* pos,
                                        const uint32_t* neg,
                                        const uint32_t* x,
                                        size_t n_words);
void     bitset_and_u32_p4             (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
void     bitset_or_u32_p4              (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
void     bitset_xor_u32_p4             (uint32_t* dst,
                                        const uint32_t* a,
                                        const uint32_t* b,
                                        size_t n_words);
uint32_t bitset_popcount_u32_p4        (const uint32_t* a, size_t n_words);

/*===========================================================================
  Public inline dispatchers
===========================================================================*/
static BITSET_ALWAYS_INLINE BITSET_HOT
uint32_t bitset_clause_mismatch_u32(const uint32_t* pos,
                                    const uint32_t* neg,
                                    const uint32_t* x,
                                    size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    return bitset_clause_mismatch_u32_p4(pos, neg, x, n_words);
#elif BITSET_ENABLE_S3_VEC
    return bitset_clause_mismatch_u32_s3(pos, neg, x, n_words);
#else
    return bitset_clause_mismatch_u32_scalar(pos, neg, x, n_words);
#endif
}

static BITSET_ALWAYS_INLINE BITSET_HOT
int bitset_clause_holds_u32(const uint32_t* pos,
                            const uint32_t* neg,
                            const uint32_t* x,
                            size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    return bitset_clause_holds_u32_p4(pos, neg, x, n_words);
#elif BITSET_ENABLE_S3_VEC
    return bitset_clause_holds_u32_s3(pos, neg, x, n_words);
#else
    return bitset_clause_holds_u32_scalar(pos, neg, x, n_words);
#endif
}

static BITSET_ALWAYS_INLINE BITSET_HOT
void bitset_and_u32(uint32_t* dst, const uint32_t* a,
                    const uint32_t* b, size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    bitset_and_u32_p4(dst, a, b, n_words);
#elif BITSET_ENABLE_S3_VEC
    bitset_and_u32_s3(dst, a, b, n_words);
#else
    bitset_and_u32_scalar(dst, a, b, n_words);
#endif
}

static BITSET_ALWAYS_INLINE BITSET_HOT
void bitset_or_u32(uint32_t* dst, const uint32_t* a,
                   const uint32_t* b, size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    bitset_or_u32_p4(dst, a, b, n_words);
#elif BITSET_ENABLE_S3_VEC
    bitset_or_u32_s3(dst, a, b, n_words);
#else
    bitset_or_u32_scalar(dst, a, b, n_words);
#endif
}

static BITSET_ALWAYS_INLINE BITSET_HOT
void bitset_xor_u32(uint32_t* dst, const uint32_t* a,
                    const uint32_t* b, size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    bitset_xor_u32_p4(dst, a, b, n_words);
#elif BITSET_ENABLE_S3_VEC
    bitset_xor_u32_s3(dst, a, b, n_words);
#else
    bitset_xor_u32_scalar(dst, a, b, n_words);
#endif
}

static BITSET_ALWAYS_INLINE BITSET_HOT
uint32_t bitset_popcount_u32(const uint32_t* a, size_t n_words)
{
#if BITSET_ENABLE_P4_VEC
    return bitset_popcount_u32_p4(a, n_words);
#elif BITSET_ENABLE_S3_VEC
    return bitset_popcount_u32_s3(a, n_words);
#else
    return bitset_popcount_u32_scalar(a, n_words);
#endif
}

#ifdef __cplusplus
} // extern "C"
#endif

