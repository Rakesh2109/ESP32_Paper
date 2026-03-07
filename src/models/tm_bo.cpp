// src/tm_bo.c
#include "models/tm_bo.h"
#include "simd/bitset_kernels.h"
#include "debug/bounds_monitor.h"

#include <string.h>   // memcpy, memset
#include <stdlib.h>   // malloc/free
#include <stdint.h>

#if defined(ESP_PLATFORM) && !defined(NATIVE_BUILD)
  #include "esp_heap_caps.h"
  static inline void* tm_malloc_fast(size_t n){ return heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT); }
  static inline void* tm_calloc_fast(size_t n){ void* p = tm_malloc_fast(n); if (p) memset(p, 0, n); return p; }
  static inline void  tm_free_fast(void* p)   { if (p) heap_caps_free(p); }
  // Prefer SPIRAM for large slabs to increase usable capacity on ESP32 variants
  static inline void* tm_malloc_slab(size_t n){
    void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p;
  }
  static inline void* tm_calloc_slab(size_t n){ void* p = tm_malloc_slab(n); if (p) memset(p, 0, n); return p; }
  static inline void  tm_free_slab(void* p)   { if (p) heap_caps_free(p); }
#else
  static inline void* tm_malloc_fast(size_t n){ return malloc(n); }
  static inline void* tm_calloc_fast(size_t n){ void* p = malloc(n); if (p) memset(p, 0, n); return p; }
  static inline void  tm_free_fast(void* p)   { free(p); }
  static inline void* tm_malloc_slab(size_t n){ return malloc(n); }
  static inline void* tm_calloc_slab(size_t n){ void* p = malloc(n); if (p) memset(p, 0, n); return p; }
  static inline void  tm_free_slab(void* p)   { free(p); }
#endif

/* ============================= RNG ============================== */
static inline uint64_t tm_rand_u64_(uint64_t s[2]) {
  uint64_t s1 = s[0], s0 = s[1];
  s[0] = s0;
  s1 ^= s1 << 23;
  s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
  return s[1] + s0;
}
static inline uint32_t tm_rand_u24_(uint64_t s[2]) { return (uint32_t)(tm_rand_u64_(s) >> 40); }
static inline float    tm_rand_float_(uint64_t s[2]) {
  return (float)tm_rand_u24_(s) * (1.0f / 16777216.0f);
}

static inline uint32_t prob_to_u24_threshold_(int numer, int denom) {
  const uint32_t P = (1u << 24);
  if (denom <= 0 || numer <= 0) return 0u;
  if (numer >= denom) return P;
  return (uint32_t)(((uint64_t)numer * (uint64_t)P) / (uint64_t)denom);
}

// P4-tuned gates (overridable via build flags).
#ifndef TM_P4_FAST_BERNOULLI
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TM_P4_FAST_BERNOULLI 1
#else
#define TM_P4_FAST_BERNOULLI 0
#endif
#endif

#ifndef TM_P4_DUAL_RNG24
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TM_P4_DUAL_RNG24 1
#else
#define TM_P4_DUAL_RNG24 0
#endif
#endif

#ifndef TM_P4_CLAUSE_EARLY_EXIT
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TM_P4_CLAUSE_EARLY_EXIT 1
#else
#define TM_P4_CLAUSE_EARLY_EXIT 0
#endif
#endif

/* ============================ LUTs =============================== */
static void tm_build_luts_(TM_BO* tm){
  if (tm->lut.built) return;
  const int N = tm->num_states;
  for (int s = 0; s < 256; ++s) {
    int in  = s;
    int inc = in < (2*N) ? (in + 1) : (2*N);
    int dec = in > 1     ? (in - 1) : 1;
    tm->lut.INC[s]        = (uint8_t)inc;
    tm->lut.STATE_DEC[s]  = (uint8_t)dec;
    tm->lut.CROSS_UP[s]   = (in <= N && inc > N) ? 1u : 0u;
    tm->lut.CROSS_DOWN[s] = (in >  N && dec <= N) ? 1u : 0u;
  }
  tm->lut.built = 1;
}

/* =========================== Bit ops ============================ */
static inline void clause_set_mask_bit_(TMClause* c, uint32_t* mask, int k) {
  const int wi = k >> 5;
  const uint32_t before = c->pos_mask[wi] | c->neg_mask[wi];
  mask[(uint32_t)wi] |= (1u << (k & 31));
  if (before == 0u) {
    ++c->active_word_count;
  }
}

static inline void clause_clr_mask_bit_(TMClause* c, uint32_t* mask, int k) {
  const int wi = k >> 5;
  mask[(uint32_t)wi] &= ~(1u << (k & 31));
  if ((c->pos_mask[wi] | c->neg_mask[wi]) == 0u && c->active_word_count > 0u) {
    --c->active_word_count;
  }
}

static inline int xbit_(const uint32_t* Xw, int k) {
  return (int)((Xw[(uint32_t)k >> 5] >> (k & 31)) & 1u);
}

static inline uint8_t clause_holds_early_u32_(const uint32_t* pos,
                                              const uint32_t* neg,
                                              const uint32_t* xw,
                                              int n_words) {
  for (int i = 0; i < n_words; ++i) {
    uint32_t x = xw[i];
    if (((pos[i] & ~x) | (neg[i] & x)) != 0u) return 0u;
  }
  return 1u;
}

#if TM_INPUT_IS_MSB_PACKED
static inline uint8_t rev8_(uint8_t x) {
  x = (uint8_t)((x >> 4) | (x << 4));
  x = (uint8_t)(((x & 0xCCu) >> 2) | ((x & 0x33u) << 2));
  x = (uint8_t)(((x & 0xAAu) >> 1) | ((x & 0x55u) << 1));
  return x;
}
#endif

/* ======================== NZ list helper ========================= */
#if TM_USE_NZ
static inline void update_nz_list_(TM_BO* tm, TMClause* c, int w) {
  (void)tm;
  const uint32_t m = c->pos_mask[w] | c->neg_mask[w];
  const uint16_t old_pos1 = c->nz_pos[w];
  if (m != 0u) {
    if (old_pos1 == 0u) {
      const uint16_t pos = (uint16_t)c->nz_len;
      c->nz_idx[pos] = (uint16_t)w;
      c->nz_pos[w] = (uint16_t)(pos + 1u);
      c->nz_len++;
    }
  } else if (old_pos1 != 0u) {
    const uint16_t pos = (uint16_t)(old_pos1 - 1u);
    const uint16_t last_pos = (uint16_t)(c->nz_len - 1);
    const uint16_t moved_w = c->nz_idx[last_pos];
    c->nz_idx[pos] = moved_w;
    c->nz_pos[moved_w] = (uint16_t)(pos + 1u);
    c->nz_pos[w] = 0u;
    c->nz_len--;
  }
}
#endif

/* ======================== ID index helper ========================= */
static bool ensure_id_index_(TM_BO* tm, uint32_t id) {
  if ((int)id < tm->id_to_idx_cap) return true;
  int new_cap = (tm->id_to_idx_cap > 0) ? tm->id_to_idx_cap : 16;
  while (new_cap <= (int)id) {
    if (new_cap > (1 << 30)) return false;
    new_cap <<= 1;
  }
  int32_t* map_new = (int32_t*)tm_malloc_fast((size_t)new_cap * sizeof(int32_t));
  if (!map_new) return false;
  for (int i = 0; i < new_cap; ++i) map_new[i] = -1;
  if (tm->id_to_idx && tm->id_to_idx_cap > 0) {
    memcpy(map_new, tm->id_to_idx, (size_t)tm->id_to_idx_cap * sizeof(int32_t));
    tm_free_fast(tm->id_to_idx);
  }
  tm->id_to_idx = map_new;
  tm->id_to_idx_cap = new_cap;
  return true;
}

/* ===================== Pack X → 32-bit words ===================== */
static TM_HOT void tm_pack_X_to_words_(TM_BO* tm,
#if TM_INPUT_IS_MSB_PACKED
    const uint8_t* __restrict__ packed_msb
#else
    const uint8_t* __restrict__ X01
#endif
) {
  const int F = tm->num_features, W = tm->words_per_feat;
  if (tm->xwords_len != W) {
    tm_free_fast(tm->xwords);
    tm->xwords = (uint32_t*)tm_malloc_fast((size_t)W * sizeof(uint32_t));
    tm->xwords_len = W;
  }
#if TM_INPUT_IS_MSB_PACKED
  const int nbytes = (F + 7) >> 3;
  uint32_t* dst = tm->xwords;
  int wi = 0, i = 0;
  // process 4 bytes → 32 bits
  for (; i + 4 <= nbytes; i += 4) {
    uint32_t w =  (uint32_t)rev8_(packed_msb[i + 0])        |
                 ((uint32_t)rev8_(packed_msb[i + 1]) << 8 ) |
                 ((uint32_t)rev8_(packed_msb[i + 2]) << 16) |
                 ((uint32_t)rev8_(packed_msb[i + 3]) << 24);
    dst[wi++] = w;
  }
  if (i < nbytes) {
    uint32_t w = 0; int sh = 0;
    for (; i < nbytes; ++i, sh += 8) w |= ((uint32_t)rev8_(packed_msb[i]) << sh);
    dst[wi++] = w;
  }
  const int tail = F & 31;
  if (tail) {
    uint32_t mask = (tail == 32) ? 0xFFFFFFFFu : ((1u << tail) - 1u);
    dst[W - 1] &= mask;
  }
#else
  // 0/1 bytes → words
  for (int w = 0; w < W; ++w) {
    uint32_t v = 0u, base = (uint32_t)w << 5;
    for (int b = 0; b < 32; ++b) {
      int k = (int)base + b; if (k >= F) break;
      if (X01[k]) v |= (1u << b);
    }
    tm->xwords[w] = v;
  }
#endif
}

/* ================= Clause output & vote sum ======================= */
static inline uint8_t clause_has_any_literal_(const TMClause* c, int W) {
  (void)W;
  return (uint8_t)(c->active_word_count > 0u);
}

static inline uint8_t clause_eval_from_words_(const TMClause* c, const uint32_t* xw, int W) {
#if TM_USE_NZ
  // Only touch words that are non-zero in (pos|neg)
  for (int ii = 0; ii < c->nz_len; ++ii) {
    const int i = (int)c->nz_idx[ii];
    const uint32_t x = xw[i];
    if (((c->pos_mask[i] & ~x) | (c->neg_mask[i] & x)) != 0u) return 0u;
  }
  return 1u;
#else
#if TM_P4_CLAUSE_EARLY_EXIT
  return clause_holds_early_u32_(c->pos_mask, c->neg_mask, xw, W);
#else
  // Use backend bitset kernel across all words
  const uint32_t mis = bitset_clause_mismatch_u32(c->pos_mask, c->neg_mask, xw, (size_t)W);
  return (uint8_t)(mis == 0u);
#endif
#endif
}

static inline int tm_clip_class_sum_(const TM_BO* tm, int s) {
  if (s > tm->threshold) return tm->threshold;
  if (s < -tm->threshold) return -tm->threshold;
  return s;
}

static TM_HOT int tm_calculate_clause_output_update_(TM_BO* tm, const uint8_t* X) {
  tm_pack_X_to_words_(tm, X);
  const int W = tm->words_per_feat;
  const uint32_t* __restrict__ xw = tm->xwords;
  uint8_t* __restrict__ out = tm->clause_output;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  int s = 0;

  for (int j = 0; j < tm->num_active; ++j) {
    const uint8_t cj = clause_eval_from_words_(act[j], xw, W);
    out[j] = cj;
    s += (int)cj * (int)act[j]->sign;
  }
  return tm_clip_class_sum_(tm, s);
}

static TM_HOT int tm_calculate_clause_output_predict_(TM_BO* tm, const uint8_t* X) {
  tm_pack_X_to_words_(tm, X);
  const int W = tm->words_per_feat;
  const uint32_t* __restrict__ xw = tm->xwords;
  uint8_t* __restrict__ out = tm->clause_output;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  int s = 0;

  for (int j = 0; j < tm->num_active; ++j) {
    const TMClause* c = act[j];
    if (!clause_has_any_literal_(c, W)) {
      out[j] = 0;
      continue;
    }
    const uint8_t cj = clause_eval_from_words_(c, xw, W);
    out[j] = cj;
    s += (int)cj * (int)c->sign;
  }
  return tm_clip_class_sum_(tm, s);
}

/* ===================== Feedback (Type I / II) ===================== */
#if TM_P4_FAST_BERNOULLI
typedef uint32_t tm_s_thresh_t;
static inline tm_s_thresh_t tm_1_over_s_threshold_(int s) {
  return prob_to_u24_threshold_(1, s);
}
static inline void tm_rand2_u24_(TM_BO* tm, uint32_t* r0, uint32_t* r1) {
  const uint64_t r = tm_rand_u64_(tm->rng_state);
  *r0 = (uint32_t)(r >> 40);
  *r1 = (uint32_t)((r >> 16) & 0xFFFFFFu);
}
static inline bool bern_1_over_s_(TM_BO* tm, tm_s_thresh_t thr_1_div_s) {
  return tm_rand_u24_(tm->rng_state) <= thr_1_div_s;
}
static inline bool bern_s1_over_s_(TM_BO* tm, tm_s_thresh_t thr_1_div_s) {
  return tm_rand_u24_(tm->rng_state) > thr_1_div_s;
}
#else
typedef uint64_t tm_s_thresh_t;
static inline tm_s_thresh_t tm_1_over_s_threshold_(int s) {
  return (s > 0) ? (UINT64_MAX / (uint64_t)s) : UINT64_MAX;
}
static inline bool bern_1_over_s_(TM_BO* tm, tm_s_thresh_t thr_1_div_s) {
  return tm_rand_u64_(tm->rng_state) <= thr_1_div_s;
}
static inline bool bern_s1_over_s_(TM_BO* tm, tm_s_thresh_t thr_1_div_s) {
  return tm_rand_u64_(tm->rng_state) > thr_1_div_s;
}
#endif

static TM_HOT void tm_type_i_feedback_(TM_BO* tm, const uint32_t* Xw, TMClause* c,
                                       tm_s_thresh_t thr_1_div_s, uint8_t clause_out) {
  const int F = tm->num_features, N = tm->num_states, W = tm->words_per_feat;
  uint8_t* __restrict__ pos_state = c->pos_state;
  uint8_t* __restrict__ neg_state = c->neg_state;
  uint32_t* __restrict__ pos_mask = c->pos_mask;
  uint32_t* __restrict__ neg_mask = c->neg_mask;
  const uint8_t* __restrict__ inc_lut = tm->lut.INC;
  const uint8_t* __restrict__ dec_lut = tm->lut.STATE_DEC;
  const uint8_t* __restrict__ up_lut = tm->lut.CROSS_UP;
  const uint8_t* __restrict__ down_lut = tm->lut.CROSS_DOWN;

  if (!clause_out) {
    // Penalize with prob 1/s
    int k = 0;
    for (int wi = 0; wi < W && k < F; ++wi) {
#if TM_USE_NZ
      bool word_dirty = false;
#endif
      int lim = F - k;
      if (lim > 32) lim = 32;
      for (int b = 0; b < lim; ++b, ++k) {
#if TM_P4_FAST_BERNOULLI && TM_P4_DUAL_RNG24
        uint32_t r0 = 0u, r1 = 0u;
        tm_rand2_u24_(tm, &r0, &r1);
        if (pos_state[k] > 1 && r0 <= thr_1_div_s) {
          uint8_t old = pos_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
        if (neg_state[k] > 1 && r1 <= thr_1_div_s) {
          uint8_t old = neg_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
#else
        if (pos_state[k] > 1 && bern_1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = pos_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
        if (neg_state[k] > 1 && bern_1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = neg_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
#endif
      }
#if TM_USE_NZ
      if (word_dirty) update_nz_list_(tm, c, wi);
#endif
    }
    return;
  }

  // Reward matching literals
  int k = 0;
  for (int wi = 0; wi < W && k < F; ++wi) {
    uint32_t bits = Xw[wi];
#if TM_USE_NZ
    bool word_dirty = false;
#endif
    int lim = F - k;
    if (lim > 32) lim = 32;
    for (int b = 0; b < lim; ++b, ++k) {
      int xk = (int)(bits & 1u);
      bits >>= 1;
      if (xk) {
#if TM_P4_FAST_BERNOULLI && TM_P4_DUAL_RNG24
        uint32_t r0 = 0u, r1 = 0u;
        tm_rand2_u24_(tm, &r0, &r1);
        if (pos_state[k] < 2*N && r0 > thr_1_div_s) {
          uint8_t old = pos_state[k];
          uint8_t neu = inc_lut[old];
          if (up_lut[old]) {
            clause_set_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
        if (neg_state[k] > 1 && r1 <= thr_1_div_s) {
          uint8_t old = neg_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
#else
        if (pos_state[k] < 2*N && bern_s1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = pos_state[k];
          uint8_t neu = inc_lut[old];
          if (up_lut[old]) {
            clause_set_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
        if (neg_state[k] > 1 && bern_1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = neg_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
#endif
      } else {
#if TM_P4_FAST_BERNOULLI && TM_P4_DUAL_RNG24
        uint32_t r0 = 0u, r1 = 0u;
        tm_rand2_u24_(tm, &r0, &r1);
        if (neg_state[k] < 2*N && r0 > thr_1_div_s) {
          uint8_t old = neg_state[k];
          uint8_t neu = inc_lut[old];
          if (up_lut[old]) {
            clause_set_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
        if (pos_state[k] > 1 && r1 <= thr_1_div_s) {
          uint8_t old = pos_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
#else
        if (neg_state[k] < 2*N && bern_s1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = neg_state[k];
          uint8_t neu = inc_lut[old];
          if (up_lut[old]) {
            clause_set_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          neg_state[k] = neu;
        }
        if (pos_state[k] > 1 && bern_1_over_s_(tm, thr_1_div_s)) {
          uint8_t old = pos_state[k];
          uint8_t neu = dec_lut[old];
          if (down_lut[old]) {
            clause_clr_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
            word_dirty = true;
#endif
          }
          pos_state[k] = neu;
        }
#endif
      }
    }
#if TM_USE_NZ
    if (word_dirty) update_nz_list_(tm, c, wi);
#endif
  }
}

static TM_HOT void tm_type_ii_feedback_(TM_BO* tm, const uint32_t* Xw, TMClause* c) {
  const int F = tm->num_features, N = tm->num_states, W = tm->words_per_feat;
  uint8_t* __restrict__ pos_state = c->pos_state;
  uint8_t* __restrict__ neg_state = c->neg_state;
  uint32_t* __restrict__ pos_mask = c->pos_mask;
  uint32_t* __restrict__ neg_mask = c->neg_mask;
  const uint8_t* __restrict__ inc_lut = tm->lut.INC;
  const uint8_t* __restrict__ up_lut = tm->lut.CROSS_UP;

  int k = 0;
  for (int wi = 0; wi < W && k < F; ++wi) {
    uint32_t bits = Xw[wi];
#if TM_USE_NZ
    bool word_dirty = false;
#endif
    int lim = F - k;
    if (lim > 32) lim = 32;
    for (int b = 0; b < lim; ++b, ++k) {
      int xk = (int)(bits & 1u);
      bits >>= 1;

      if ((pos_state[k] <= N) && (xk == 0)) {
        uint8_t old = pos_state[k];
        uint8_t neu = inc_lut[old];
        if (up_lut[old]) {
          clause_set_mask_bit_(c, pos_mask, k);
#if TM_USE_NZ
          word_dirty = true;
#endif
        }
        pos_state[k] = neu;
      }
      if ((neg_state[k] <= N) && (xk != 0)) {
        uint8_t old = neg_state[k];
        uint8_t neu = inc_lut[old];
        if (up_lut[old]) {
          clause_set_mask_bit_(c, neg_mask, k);
#if TM_USE_NZ
          word_dirty = true;
#endif
        }
        neg_state[k] = neu;
      }
    }
#if TM_USE_NZ
    if (word_dirty) update_nz_list_(tm, c, wi);
#endif
  }
}

/* ==================== Active scratch ensure ====================== */
static bool ensure_active_scratch_(TM_BO* tm, int need) {
  if (tm->active_cap >= need) return true;
  int newcap = tm->active_cap ? (tm->active_cap * 2) : 8;
  while (newcap < need) newcap *= 2;

  TMClause** active_new = (TMClause**)tm_malloc_fast((size_t)newcap * sizeof(TMClause*));
  uint8_t*   out_new    = (uint8_t*) tm_malloc_fast((size_t)newcap);
  int8_t*    fb_new     = (int8_t*)  tm_malloc_fast((size_t)newcap);

  if (!active_new || !out_new || !fb_new) {
    if (active_new) tm_free_fast(active_new);
    if (out_new)    tm_free_fast(out_new);
    if (fb_new)     tm_free_fast(fb_new);
    return false;
  }
  if (tm->active)        memcpy(active_new, tm->active,        (size_t)tm->num_active * sizeof(TMClause*));
  if (tm->clause_output) memcpy(out_new,    tm->clause_output, (size_t)tm->num_active);
  if (tm->feedback_to_clauses) memcpy(fb_new, tm->feedback_to_clauses, (size_t)tm->num_active);

  tm_free_fast(tm->active);
  tm_free_fast(tm->clause_output);
  tm_free_fast(tm->feedback_to_clauses);

  tm->active = active_new;
  tm->clause_output = out_new;
  tm->feedback_to_clauses = fb_new;
  tm->active_cap = newcap;
  return true;
}

/* ============ Capacity reallocation for slabs/clauses ============ */
static bool realloc_capacity_(TM_BO* tm, int new_cap) {
  const int F = tm->num_features, W = tm->words_per_feat;
  if (new_cap < 1) new_cap = 1;
  if (new_cap == tm->capacity) return true;

  // Allocate large slabs from SPIRAM when available to reduce pressure on internal RAM
  uint8_t*  pos_state_new = (uint8_t*) tm_malloc_slab((size_t)new_cap * (size_t)F);
  uint8_t*  neg_state_new = (uint8_t*) tm_malloc_slab((size_t)new_cap * (size_t)F);
  // Keep hot masks in internal RAM for faster clause checks
  uint32_t* pos_mask_new  = (uint32_t*)tm_malloc_fast((size_t)new_cap * (size_t)W * sizeof(uint32_t));
  uint32_t* neg_mask_new  = (uint32_t*)tm_malloc_fast((size_t)new_cap * (size_t)W * sizeof(uint32_t));
#if TM_USE_NZ
  uint16_t* nz_idx_new    = (uint16_t*)tm_malloc_fast((size_t)new_cap * (size_t)W * sizeof(uint16_t));
  uint16_t* nz_pos_new    = (uint16_t*)tm_malloc_fast((size_t)new_cap * (size_t)W * sizeof(uint16_t));
#endif
  TMClause* clauses_new   = (TMClause*) tm_malloc_slab((size_t)new_cap * sizeof(TMClause));

  if (!pos_state_new || !neg_state_new || !pos_mask_new || !neg_mask_new
#if TM_USE_NZ
      || !nz_idx_new || !nz_pos_new
#endif
      || !clauses_new) {
    if (pos_state_new) tm_free_slab(pos_state_new);
    if (neg_state_new) tm_free_slab(neg_state_new);
    if (pos_mask_new)  tm_free_fast(pos_mask_new);
    if (neg_mask_new)  tm_free_fast(neg_mask_new);
#if TM_USE_NZ
    if (nz_idx_new)    tm_free_fast(nz_idx_new);
    if (nz_pos_new)    tm_free_fast(nz_pos_new);
#endif
    if (clauses_new)   tm_free_slab(clauses_new);
    return false;
  }

  // Copy old slabs
  const int copy_cnt = (tm->num_used < new_cap) ? tm->num_used : new_cap;
  if (tm->capacity > 0 && copy_cnt > 0) {
    memcpy(pos_state_new, tm->slab_pos_state, (size_t)copy_cnt * (size_t)F * sizeof(uint8_t));
    memcpy(neg_state_new, tm->slab_neg_state, (size_t)copy_cnt * (size_t)F * sizeof(uint8_t));
    memcpy(pos_mask_new,  tm->slab_pos_mask,  (size_t)copy_cnt * (size_t)W * sizeof(uint32_t));
    memcpy(neg_mask_new,  tm->slab_neg_mask,  (size_t)copy_cnt * (size_t)W * sizeof(uint32_t));
#if TM_USE_NZ
    memcpy(nz_idx_new,    tm->slab_nz_idx,    (size_t)copy_cnt * (size_t)W * sizeof(uint16_t));
    memcpy(nz_pos_new,    tm->slab_nz_pos,    (size_t)copy_cnt * (size_t)W * sizeof(uint16_t));
#endif
    memcpy(clauses_new,   tm->clauses,        (size_t)copy_cnt * sizeof(TMClause));
  } else {
    memset(pos_state_new, 0, (size_t)new_cap * (size_t)F * sizeof(uint8_t));
    memset(neg_state_new, 0, (size_t)new_cap * (size_t)F * sizeof(uint8_t));
    memset(pos_mask_new,  0, (size_t)new_cap * (size_t)W * sizeof(uint32_t));
    memset(neg_mask_new,  0, (size_t)new_cap * (size_t)W * sizeof(uint32_t));
#if TM_USE_NZ
    memset(nz_idx_new,    0, (size_t)new_cap * (size_t)W * sizeof(uint16_t));
    memset(nz_pos_new,    0, (size_t)new_cap * (size_t)W * sizeof(uint16_t));
#endif
    memset(clauses_new,   0, (size_t)new_cap * sizeof(TMClause));
  }

  // Fix clause slab pointers
  for (int i=0;i<copy_cnt;++i) {
    clauses_new[i].pos_state = pos_state_new + i*F;
    clauses_new[i].neg_state = neg_state_new + i*F;
    clauses_new[i].pos_mask  = pos_mask_new  + i*W;
    clauses_new[i].neg_mask  = neg_mask_new  + i*W;
#if TM_USE_NZ
    clauses_new[i].nz_idx    = nz_idx_new    + i*W;
    clauses_new[i].nz_pos    = nz_pos_new    + i*W;
#endif
  }

  // Update active pointers (they point at items of clauses[])
  if (tm->active) {
    for (int j=0;j<tm->num_active;++j) {
      if (tm->active[j]) {
        uintptr_t offset = (uintptr_t)tm->active[j] - (uintptr_t)tm->clauses;
        tm->active[j] = (TMClause*)((uintptr_t)clauses_new + offset);
      }
    }
  }

  // Swap in
  tm_free_slab(tm->slab_pos_state);
  tm_free_slab(tm->slab_neg_state);
  tm_free_fast(tm->slab_pos_mask);
  tm_free_fast(tm->slab_neg_mask);
#if TM_USE_NZ
  tm_free_fast(tm->slab_nz_idx);
  tm_free_fast(tm->slab_nz_pos);
#endif
  tm_free_slab(tm->clauses);

  tm->slab_pos_state = pos_state_new;
  tm->slab_neg_state = neg_state_new;
  tm->slab_pos_mask  = pos_mask_new;
  tm->slab_neg_mask  = neg_mask_new;
#if TM_USE_NZ
  tm->slab_nz_idx    = nz_idx_new;
  tm->slab_nz_pos    = nz_pos_new;
#endif
  tm->clauses        = clauses_new;
  tm->capacity       = new_cap;
  if (tm->num_used > new_cap) tm->num_used = new_cap;
  return true;
}

static void maybe_shrink_(TM_BO* tm) {
  if (tm->capacity <= 1) return;
  if (tm->num_used * 2 > tm->capacity) return;
  int new_cap = tm->num_used > 0 ? tm->num_used : 1;
  (void)realloc_capacity_(tm, new_cap);
}

/* ================= Clause lookup ========================= */
static int find_clause_idx_(TM_BO* tm, uint32_t id) {
  if ((int)id < tm->id_to_idx_cap) {
    const int idx = (int)tm->id_to_idx[id];
    if (idx >= 0 && idx < tm->num_used && tm->clauses[idx].clause_id == id) {
      return idx;
    }
  }
  for (int i = 0; i < tm->num_used; ++i) {
    if (tm->clauses[i].clause_id == id) {
      if ((int)id < tm->id_to_idx_cap) tm->id_to_idx[id] = (int32_t)i;
      return i;
    }
  }
  return -1;
}

/* =========================== Init/Free ============================ */
bool tm_bo_init(TM_BO* tm, int features, int threshold, uint32_t seed) {
  if (!tm || features <= 0) return false;
  memset(tm, 0, sizeof(*tm));
  tm->num_features   = features;
  tm->words_per_feat = (features + 31) >> 5;
  tm->threshold      = (threshold > 0) ? threshold : 1;
  tm->num_states     = NUMBER_OF_STATES;
  tm->inv_2T         = 1.0f / (2.0f * (float)tm->threshold);
  tm->init_literal_density_pct = (uint8_t)TM_INIT_LITERAL_DENSITY_PCT;

  // RNG seed
  tm->rng_state[0] = seed ^ 0x9E3779B97F4A7C15ULL;
  tm->rng_state[1] = (seed + 1) ^ 0x9E3779B97F4A7C15ULL;
  (void)tm_rand_u64_(tm->rng_state);
  (void)tm_rand_u64_(tm->rng_state);

  tm->capacity = 0;
  tm->num_used = 0;
  tm->num_active = 0;
  tm->active_cap = 0;
  tm->next_clause_id = 1u;
  tm->slab_pos_state = tm->slab_neg_state = NULL;
  tm->slab_pos_mask  = tm->slab_neg_mask  = NULL;
#if TM_USE_NZ
  tm->slab_nz_idx    = NULL;
  tm->slab_nz_pos    = NULL;
#endif
  tm->clauses = NULL;
  tm->active  = NULL;
  tm->clause_output = NULL;
  tm->feedback_to_clauses = NULL;
  tm->xwords = NULL; tm->xwords_len = 0;
  tm->id_to_idx = NULL;
  tm->id_to_idx_cap = 0;
  tm->lut.built = 0;
  tm_build_luts_(tm);
  return true;
}

void tm_bo_set_init_literal_density(TM_BO* tm, int density_pct) {
  if (!tm) return;
  if (density_pct < 0) density_pct = 0;
  if (density_pct > 100) density_pct = 100;
  tm->init_literal_density_pct = (uint8_t)density_pct;
}

void tm_bo_free(TM_BO* tm) {
  if (!tm) return;
  tm_free_slab(tm->slab_pos_state);
  tm_free_slab(tm->slab_neg_state);
  tm_free_fast(tm->slab_pos_mask);
  tm_free_fast(tm->slab_neg_mask);
#if TM_USE_NZ
  tm_free_fast(tm->slab_nz_idx);
  tm_free_fast(tm->slab_nz_pos);
#endif
  tm_free_slab(tm->clauses);
  tm_free_fast(tm->active);
  tm_free_fast(tm->clause_output);
  tm_free_fast(tm->feedback_to_clauses);
  tm_free_fast(tm->xwords);
  tm_free_fast(tm->id_to_idx);
  memset(tm, 0, sizeof(*tm));
}

/* ====================== Clause management ========================= */
static void init_clause_random_(TM_BO* tm, TMClause* c) {
  const int F = tm->num_features;
  const int N = tm->num_states;
  const int W = tm->words_per_feat;
  int density = (int)tm->init_literal_density_pct;
  if (density < 0) density = 0;
  if (density > 100) density = 100;

#if TM_USE_NZ
  c->nz_len = 0;
  memset(c->nz_pos, 0, (size_t)W * sizeof(uint16_t));
#endif
  for (int k=0;k<F;++k) {
    const bool include_lit = (density >= 100) || ((int)(tm_rand_u24_(tm->rng_state) % 100u) < density);
    if (!include_lit) {
      c->pos_state[k] = (uint8_t)N;
      c->neg_state[k] = (uint8_t)N;
      continue;
    }
    if (tm_rand_float_(tm->rng_state) <= 0.5f) {
      c->pos_state[k] = (uint8_t)N;
      c->neg_state[k] = (uint8_t)(N+1);
      clause_set_mask_bit_(c, c->neg_mask, k);
    } else {
      c->pos_state[k] = (uint8_t)(N+1);
      c->neg_state[k] = (uint8_t)N;
      clause_set_mask_bit_(c, c->pos_mask, k);
    }
  }
#if TM_USE_NZ
  for (int w = 0; w < W; ++w) {
    if ((c->pos_mask[w] | c->neg_mask[w]) != 0u) {
      const uint16_t pos = (uint16_t)c->nz_len;
      c->nz_idx[pos] = (uint16_t)w;
      c->nz_pos[w] = (uint16_t)(pos + 1u);
      c->nz_len++;
    }
  }
#endif
}

TMClause* tm_clause_add_hot(TM_BO* tm, int8_t sign) {
  if (!tm) return NULL;
  int need = (tm->num_used == 0) ? 1 : tm->num_used + 1;
  if (need > tm->capacity) {
    int new_cap = tm->capacity ? (tm->capacity * 2) : 4;
    while (new_cap < need) new_cap *= 2;
    if (!realloc_capacity_(tm, new_cap)) return NULL;
  }
  const int i = tm->num_used++;
  const int F = tm->num_features, W = tm->words_per_feat;

  TMClause* c = &tm->clauses[i];
  c->clause_id = tm->next_clause_id;
  if (!ensure_id_index_(tm, c->clause_id)) {
    tm->num_used--;
    return NULL;
  }
  tm->next_clause_id++;
  c->sign = (sign >= 0) ? 1 : -1;
  c->frozen = 0;
  c->is_active = 0;
  c->active_word_count = 0u;

  c->pos_state = tm->slab_pos_state + i*F;
  c->neg_state = tm->slab_neg_state + i*F;
  c->pos_mask  = tm->slab_pos_mask  + i*W;
  c->neg_mask  = tm->slab_neg_mask  + i*W;
  memset(c->pos_mask, 0, (size_t)W * sizeof(uint32_t));
  memset(c->neg_mask, 0, (size_t)W * sizeof(uint32_t));
#if TM_USE_NZ
  c->nz_idx = tm->slab_nz_idx + i*W;
  c->nz_pos = tm->slab_nz_pos + i*W;
  c->nz_len = 0;
  memset(c->nz_idx, 0, (size_t)W * sizeof(uint16_t));
  memset(c->nz_pos, 0, (size_t)W * sizeof(uint16_t));
#endif
  init_clause_random_(tm, c);
  tm->id_to_idx[c->clause_id] = (int32_t)i;

  // Activate
  (void)ensure_active_scratch_(tm, tm->num_active + 1);
  tm->active[tm->num_active++] = c;
  c->is_active = 1;
  return c;
}

static void remove_from_active_(TM_BO* tm, TMClause* c) {
  if (!c->is_active || tm->num_active == 0) return;
  for (int j=0;j<tm->num_active;++j) {
    if (tm->active[j] == c) {
      tm->active[j] = tm->active[tm->num_active - 1];
      tm->num_active--;
      c->is_active = 0;
      break;
    }
  }
}

bool tm_clause_remove(TM_BO* tm, uint32_t clause_id) {
  if (!tm || tm->num_used == 0) return false;
  int idx = find_clause_idx_(tm, clause_id);
  if (idx < 0) return false;

  TMClause* c = &tm->clauses[idx];
  remove_from_active_(tm, c);
  const uint32_t removed_id = c->clause_id;

  // Swap with last to keep arrays dense
  int last = tm->num_used - 1;
  if (idx != last) {
    const int F = tm->num_features, W = tm->words_per_feat;
    memcpy(tm->slab_pos_state + idx*F, tm->slab_pos_state + last*F, (size_t)F * sizeof(uint8_t));
    memcpy(tm->slab_neg_state + idx*F, tm->slab_neg_state + last*F, (size_t)F * sizeof(uint8_t));
    memcpy(tm->slab_pos_mask  + idx*W, tm->slab_pos_mask  + last*W, (size_t)W * sizeof(uint32_t));
    memcpy(tm->slab_neg_mask  + idx*W, tm->slab_neg_mask  + last*W, (size_t)W * sizeof(uint32_t));
#if TM_USE_NZ
    memcpy(tm->slab_nz_idx   + idx*W, tm->slab_nz_idx   + last*W, (size_t)W * sizeof(uint16_t));
    memcpy(tm->slab_nz_pos   + idx*W, tm->slab_nz_pos   + last*W, (size_t)W * sizeof(uint16_t));
#endif

    tm->clauses[idx] = tm->clauses[last];
    TMClause* moved = &tm->clauses[idx];
    moved->pos_state = tm->slab_pos_state + idx*F;
    moved->neg_state = tm->slab_neg_state + idx*F;
    moved->pos_mask  = tm->slab_pos_mask  + idx*W;
    moved->neg_mask  = tm->slab_neg_mask  + idx*W;
#if TM_USE_NZ
    moved->nz_idx    = tm->slab_nz_idx    + idx*W;
    moved->nz_pos    = tm->slab_nz_pos    + idx*W;
#endif

    for (int j=0;j<tm->num_active;++j) {
      if (tm->active[j] == &tm->clauses[last]) tm->active[j] = moved;
    }
    if ((int)moved->clause_id < tm->id_to_idx_cap) {
      tm->id_to_idx[moved->clause_id] = (int32_t)idx;
    }
  }
  if ((int)removed_id < tm->id_to_idx_cap) tm->id_to_idx[removed_id] = -1;
  tm->num_used--;
  maybe_shrink_(tm);
  return true;
}

bool tm_clause_freeze(TM_BO* tm, uint32_t clause_id) {
  int idx = find_clause_idx_(tm, clause_id);
  if (idx < 0) return false;
  TMClause* c = &tm->clauses[idx];
  c->frozen = 1;
  remove_from_active_(tm, c);
  return true;
}

bool tm_clause_thaw(TM_BO* tm, uint32_t clause_id) {
  int idx = find_clause_idx_(tm, clause_id);
  if (idx < 0) return false;
  tm->clauses[idx].frozen = 0;
  return true;
}

bool tm_clause_activate(TM_BO* tm, uint32_t clause_id) {
  int idx = find_clause_idx_(tm, clause_id);
  if (idx < 0) return false;
  TMClause* c = &tm->clauses[idx];
  if (c->frozen) return false;
  if (c->is_active) return true;
  if (!ensure_active_scratch_(tm, tm->num_active + 1)) return false;
  tm->active[tm->num_active++] = c;
  c->is_active = 1;
  return true;
}

bool tm_clause_deactivate(TM_BO* tm, uint32_t clause_id) {
  int idx = find_clause_idx_(tm, clause_id);
  if (idx < 0) return false;
  remove_from_active_(tm, &tm->clauses[idx]);
  return true;
}

/* ===================== Update & Score ============================ */
// Forward declarations for words-based clause evaluation (no TM_HOT here to avoid attribute mismatch)
static int tm_calculate_clause_output_from_words_update_(TM_BO* tm, const uint32_t* Xw);
static int tm_calculate_clause_output_from_words_predict_(TM_BO* tm, const uint32_t* Xw);
int tm_bo_update(TM_BO* tm, const uint8_t* X_bits, int target, int s) {
  if (!tm || tm->num_active == 0) return 0;
  const int class_sum = tm_calculate_clause_output_update_(tm, X_bits);

  const int y = 2*target - 1; // +1 or -1
  const int twoT = tm->threshold << 1;
  const uint32_t th_i  = prob_to_u24_threshold_(tm->threshold - y*class_sum, twoT);
  const uint32_t th_ii = prob_to_u24_threshold_(tm->threshold + y*class_sum, twoT);
  const tm_s_thresh_t th_1_div_s = tm_1_over_s_threshold_(s);
  const uint32_t* Xw = tm->xwords;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  const uint8_t* __restrict__ out = tm->clause_output;
  for (int j=0;j<tm->num_active;++j) {
    TMClause* c = act[j];
    if ((int)c->sign == y) {
      if (tm_rand_u24_(tm->rng_state) <= th_i) {
        tm_type_i_feedback_(tm, Xw, c, th_1_div_s, out[j]);
      }
    } else {
      if (tm_rand_u24_(tm->rng_state) <= th_ii && likely(out[j])) {
        tm_type_ii_feedback_(tm, Xw, c);
      }
    }
  }
  return class_sum;
}

int tm_bo_score(TM_BO* tm, const uint8_t* X_bits) {
  if (!tm || tm->num_active == 0) return 0;
  return tm_calculate_clause_output_predict_(tm, X_bits);
}

int tm_bo_update_words(TM_BO* tm, const uint32_t* xwords, int target, int s) {
  if (!tm || tm->num_active == 0) return 0;
  const int class_sum = tm_calculate_clause_output_from_words_update_(tm, xwords);

  const int y = 2*target - 1; // +1 or -1
  const int twoT = tm->threshold << 1;
  const uint32_t th_i  = prob_to_u24_threshold_(tm->threshold - y*class_sum, twoT);
  const uint32_t th_ii = prob_to_u24_threshold_(tm->threshold + y*class_sum, twoT);
  const tm_s_thresh_t th_1_div_s = tm_1_over_s_threshold_(s);
  const uint32_t* Xw = xwords;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  const uint8_t* __restrict__ out = tm->clause_output;
  for (int j=0;j<tm->num_active;++j) {
    TMClause* c = act[j];
    if ((int)c->sign == y) {
      if (tm_rand_u24_(tm->rng_state) <= th_i) {
        tm_type_i_feedback_(tm, Xw, c, th_1_div_s, out[j]);
      }
    } else {
      if (tm_rand_u24_(tm->rng_state) <= th_ii && likely(out[j])) {
        tm_type_ii_feedback_(tm, Xw, c);
      }
    }
  }
  return class_sum;
}

// Variant that takes pre-packed words Xw to avoid per-call packing
static TM_HOT int tm_calculate_clause_output_from_words_update_(TM_BO* tm, const uint32_t* Xw) {
  const int W = tm->words_per_feat;
  uint8_t* __restrict__ out = tm->clause_output;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  int s = 0;

  for (int j = 0; j < tm->num_active; ++j) {
    const uint8_t cj = clause_eval_from_words_(act[j], Xw, W);
    out[j] = cj;
    s += (int)cj * (int)act[j]->sign;
  }
  return tm_clip_class_sum_(tm, s);
}

static TM_HOT int tm_calculate_clause_output_from_words_predict_(TM_BO* tm, const uint32_t* Xw) {
  const int W = tm->words_per_feat;
  uint8_t* __restrict__ out = tm->clause_output;
  TMClause* const* __restrict__ act = (TMClause* const*)tm->active;
  int s = 0;

  for (int j = 0; j < tm->num_active; ++j) {
    const TMClause* c = act[j];
    if (!clause_has_any_literal_(c, W)) {
      out[j] = 0;
      continue;
    }
    const uint8_t cj = clause_eval_from_words_(c, Xw, W);
    out[j] = cj;
    s += (int)cj * (int)c->sign;
  }
  return tm_clip_class_sum_(tm, s);
}

int tm_bo_score_words(TM_BO* tm, const uint32_t* xwords) {
  if (!tm || tm->num_active == 0) return 0;
  return tm_calculate_clause_output_from_words_predict_(tm, xwords);
}

/* ========================== Utilities ============================ */
uint32_t tm_bo_memory_usage_active(const TM_BO* tm) {
  if (!tm) return 0;
  const int F = tm->num_features, W = tm->words_per_feat;
  size_t states = (size_t)tm->num_used * (size_t)F * 2u * sizeof(uint8_t);
  size_t masks  = (size_t)tm->num_used * (size_t)W * 2u * sizeof(uint32_t);
  size_t clauses= (size_t)tm->num_used * sizeof(TMClause);
  size_t active = (size_t)tm->num_active * sizeof(TMClause*);
  size_t scratch= (size_t)tm->num_active * (sizeof(uint8_t) + sizeof(int8_t));
  size_t xbuf   = (size_t)tm->xwords_len * sizeof(uint32_t);
  size_t idmap  = (size_t)tm->id_to_idx_cap * sizeof(int32_t);
#if TM_USE_NZ
  size_t nzmeta = (size_t)tm->num_used * (size_t)W * 2u * sizeof(uint16_t);
#else
  size_t nzmeta = 0u;
#endif
  size_t total  = states + masks + clauses + active + scratch + xbuf + idmap + nzmeta;
  if (total > 0xFFFFFFFFu) total = 0xFFFFFFFFu;
  return (uint32_t)total;
}

/* ====================== Legacy compatibility ===================== */
bool tm_init(TsetlinMachine* tm, int features, int clauses, int threshold, uint32_t seed) {
  if (!tm) return false;
  if (!tm_bo_init(&tm->core, features, threshold, seed)) return false;
  tm->legacy_features = features;
  tm->legacy_clauses  = clauses;

  // Preallocate full capacity in one shot to avoid fragmentation on ESP32
  int target = clauses;
  while (target >= 64) {
    if (realloc_capacity_(&tm->core, target)) break;
    // Reduce and retry (halve down to minimum of 64)
    target = (target > 128) ? (target >> 1) : (target - 32);
  }
  if (target < 64) return false;

  // Preallocate active scratch for all clauses
  if (!ensure_active_scratch_(&tm->core, target)) return false;

  // Create requested number of clauses (no further slab reallocations)
  for (int j=0;j<target;++j) {
    int8_t sign = (j & 1) ? -1 : +1;
    if (!tm_clause_add_hot(&tm->core, sign)) return false;
  }
  // If we created fewer than requested, update legacy_clauses
  tm->legacy_clauses = target;
  
  // Record memory breakdown for bounds monitoring (sparse model)
  // Calculate actual memory usage from the sparse structure
  const int F = tm->core.num_features;
  uint32_t automata_bytes = (uint32_t)tm->core.num_used * (uint32_t)F * 2u * sizeof(uint8_t);
  // Use active_cap for clause_output and feedback arrays (actual allocated size)
  uint32_t clause_bytes = (uint32_t)tm->core.active_cap * sizeof(uint8_t);  // clause_output
  uint32_t feedback_bytes = (uint32_t)tm->core.active_cap * sizeof(int8_t); // feedback_to_clauses
  bounds_record_memory_breakdown(automata_bytes, clause_bytes, feedback_bytes);
  
  return true;
}

void tm_free(TsetlinMachine* tm) {
  if (!tm) return;
  tm_bo_free(&tm->core);
  memset(tm, 0, sizeof(*tm));
}

int tm_update(TsetlinMachine* tm, const uint8_t* X_bits, int target, int s) {
  return tm_bo_update(&tm->core, X_bits, target, s);
}

int tm_score(TsetlinMachine* tm, const uint8_t* X_bits) {
  return tm_bo_score(&tm->core, X_bits);
}

int tm_update_words(TsetlinMachine* tm, const uint32_t* xwords, int target, int s) {
  return tm_bo_update_words(&tm->core, xwords, target, s);
}

int tm_score_words(TsetlinMachine* tm, const uint32_t* xwords) {
  return tm_bo_score_words(&tm->core, xwords);
}

void tm_reinit(TsetlinMachine* tm) {
  if (!tm) return;
  for (int i=0;i<tm->core.num_used;++i) {
    TMClause* c = &tm->core.clauses[i];
    const int F = tm->core.num_features, W = tm->core.words_per_feat;
    memset(c->pos_mask, 0, (size_t)W * sizeof(uint32_t));
    memset(c->neg_mask, 0, (size_t)W * sizeof(uint32_t));
    c->active_word_count = 0u;
#if TM_USE_NZ
    c->nz_len = 0;
    memset(c->nz_pos, 0, (size_t)W * sizeof(uint16_t));
#endif
    init_clause_random_(&tm->core, c);
  }
}

uint32_t tm_memory_usage(int features, int clauses) {
  uint32_t ta_state_size       = (uint32_t)clauses * (uint32_t)features * 2u; // states only
  uint32_t clause_output_size  = (uint32_t)clauses;
  uint32_t feedback_size       = (uint32_t)clauses;
  uint32_t struct_size         = (uint32_t)sizeof(TsetlinMachine);
  uint64_t total = (uint64_t)struct_size + ta_state_size + clause_output_size + feedback_size;
  if (total > 0xFFFFFFFFu) total = 0xFFFFFFFFu;
  return (uint32_t)total;
}

