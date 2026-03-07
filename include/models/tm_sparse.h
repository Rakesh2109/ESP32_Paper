// include/models/tm_sparse.h
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================= Build-time knobs ========================= */
// Number of TA states per literal side (N = NUMBER_OF_STATES)
#ifndef NUMBER_OF_STATES
#  define NUMBER_OF_STATES 100
#endif

// Maintain sparse NZ-word list per clause (speeds clause check on sparse masks)
#ifndef TM_USE_NZ
#  define TM_USE_NZ 0
#endif

// Initial literal density in percent [0..100].
// 100 keeps previous behavior (all literals initialized included).
#ifndef TM_INIT_LITERAL_DENSITY_PCT
#  define TM_INIT_LITERAL_DENSITY_PCT 100
#endif

// Input layout: 0/1 bytes (0) or MSB-first packed bits (1)
#ifndef TM_INPUT_IS_MSB_PACKED
#  define TM_INPUT_IS_MSB_PACKED 0
#endif

/* ========================= Hot annotation ========================== */
#if defined(ESP_PLATFORM)
#  include <esp_attr.h>
#  ifndef TM_HOT
#    define TM_HOT IRAM_ATTR
#  endif
#else
#  ifndef TM_HOT
#    define TM_HOT
#  endif
#endif

#ifndef likely
#  define likely(x)   __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/* ============================ Clause =============================== */
typedef struct TMClause {
  uint32_t  clause_id;    // stable id
  int8_t    sign;         // +1 or -1
  uint8_t   frozen;       // 1 => cannot activate
  uint8_t   is_active;    // 1 => in active list
  uint8_t   _pad0;
  uint32_t  active_word_count; // count of words where (pos_mask|neg_mask) != 0

  // Per-literal TA states (bytes) and masks (bitsets)
  uint8_t*  pos_state;    // [F]
  uint8_t*  neg_state;    // [F]
  uint32_t* pos_mask;     // [W]
  uint32_t* neg_mask;     // [W]

#if TM_USE_NZ
  uint16_t* nz_idx;       // indices of non-zero words in (pos_mask|neg_mask)
  uint16_t* nz_pos;       // reverse map: word index -> position+1 in nz_idx (0 means absent)
  int       nz_len;
#endif
} TMClause;

/* ============================ Core ================================ */
typedef struct TMSparse {
  // Model params
  int       num_features;     // F
  int       words_per_feat;   // W = ceil(F/32)
  int       threshold;        // T
  int       num_states;       // N = NUMBER_OF_STATES
  float     inv_2T;           // 1/(2T)
  uint8_t   init_literal_density_pct; // Initial literal density [0..100]

  // RNG state (xorshift128+)
  uint64_t  rng_state[2];

  // Slabs (dense-of-clauses storage)
  uint8_t*  slab_pos_state;   // [capacity][F]
  uint8_t*  slab_neg_state;   // [capacity][F]
  uint32_t* slab_pos_mask;    // [capacity][W]
  uint32_t* slab_neg_mask;    // [capacity][W]
#if TM_USE_NZ
  uint16_t* slab_nz_idx;      // [capacity][W]
  uint16_t* slab_nz_pos;      // [capacity][W]
#endif
  TMClause* clauses;          // [capacity]
  int       capacity;         // clause storage capacity
  int       num_used;         // number of allocated clauses

  // Active set + scratch buffers
  TMClause** active;              // [active_cap]
  int        num_active;
  int        active_cap;
  uint8_t*   clause_output;       // [active_cap] 0/1
  int8_t*    feedback_to_clauses; // [active_cap]

  // Packed input buffer X → bit-words
  uint32_t*  xwords;          // [W]
  int        xwords_len;

  // Id generator
  uint32_t   next_clause_id;
  int32_t*   id_to_idx;       // direct map: clause_id -> dense index, -1 if absent
  int        id_to_idx_cap;   // number of entries in id_to_idx

  // Small LUTs
  struct {
    uint8_t INC[256];
    uint8_t STATE_DEC[256];
    uint8_t CROSS_UP[256];
    uint8_t CROSS_DOWN[256];
    int     built;
  } lut;
} TMSparse;

// Avoid symbol collisions when multiple TM backends are linked together.
#define tm_clause_add_hot        tm_sparse_clause_add_hot
#define tm_clause_remove         tm_sparse_clause_remove
#define tm_clause_freeze         tm_sparse_clause_freeze
#define tm_clause_thaw           tm_sparse_clause_thaw
#define tm_clause_activate       tm_sparse_clause_activate
#define tm_clause_deactivate     tm_sparse_clause_deactivate
#define tm_init                  tm_sparse_legacy_init
#define tm_free                  tm_sparse_legacy_free
#define tm_update                tm_sparse_legacy_update
#define tm_score                 tm_sparse_legacy_score
#define tm_update_words          tm_sparse_legacy_update_words
#define tm_score_words           tm_sparse_legacy_score_words
#define tm_reinit                tm_sparse_legacy_reinit
#define tm_memory_usage          tm_sparse_legacy_memory_usage

/* ============================ Public API =========================== */
// --- Lifecycle ---
bool tm_sparse_init(TMSparse* tm, int features, int threshold, uint32_t seed);
void tm_sparse_free(TMSparse* tm);
void tm_sparse_set_init_literal_density(TMSparse* tm, int density_pct);

// --- Clause management ---
TMClause* tm_clause_add_hot(TMSparse* tm, int8_t sign);
bool       tm_clause_remove(TMSparse* tm, uint32_t clause_id);       // free RAM
bool       tm_clause_freeze(TMSparse* tm, uint32_t clause_id);       // HOT -> COLD
bool       tm_clause_thaw(TMSparse* tm, uint32_t clause_id);         // COLD -> HOT possible
bool       tm_clause_activate(TMSparse* tm, uint32_t clause_id);     // add to active list
bool       tm_clause_deactivate(TMSparse* tm, uint32_t clause_id);   // remove from active list

// --- Learning & inference (operate only on ACTIVE clauses) ---
int   tm_sparse_update(TMSparse* tm, const uint8_t* X_bits, int target, int s);
int   tm_sparse_score (TMSparse* tm, const uint8_t* X_bits);

// Fast-path variants that reuse pre-packed input words (avoid packing per model)
int   tm_sparse_update_words(TMSparse* tm, const uint32_t* xwords, int target, int s);
int   tm_sparse_score_words (TMSparse* tm, const uint32_t* xwords);

// --- Utilities ---
uint32_t tm_sparse_memory_usage_active(const TMSparse* tm);      // RAM in use for HOT

/* ========================= Legacy wrapper ========================== */
typedef struct TsetlinMachine {
  TMSparse core;
  int      legacy_features;
  int      legacy_clauses;
} TsetlinMachine;

// Keep your original calls working:
bool     tm_init   (TsetlinMachine* tm, int features, int clauses, int threshold, uint32_t seed);
void     tm_free   (TsetlinMachine* tm);
int      tm_update (TsetlinMachine* tm, const uint8_t* X_bits, int target, int s);
int      tm_score  (TsetlinMachine* tm, const uint8_t* X_bits);
int      tm_update_words (TsetlinMachine* tm, const uint32_t* xwords, int target, int s);
int      tm_score_words  (TsetlinMachine* tm, const uint32_t* xwords);
void     tm_reinit (TsetlinMachine* tm);
uint32_t tm_memory_usage(int features, int clauses);

#ifdef __cplusplus
} // extern "C"
#endif

