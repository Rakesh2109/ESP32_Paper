/*
 * ESP32-Optimized Tsetlin Machine Implementation
 */

#include "models/tm_vanilla.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "core/utils.h"
#include "debug/bounds_monitor.h"

#ifndef TM_VANILLA_BOUNDS_INNER
#define TM_VANILLA_BOUNDS_INNER 0
#endif

// ========== RNG Functions ==========

uint64_t tm_rand(uint64_t state[2]) {
    // Xorshift128+ algorithm - fast and high-quality PRNG
    uint64_t s1 = state[0];
    uint64_t s0 = state[1];
    state[0] = s0;
    s1 ^= s1 << 23;
    state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    uint64_t result = state[1] + s0;
    return result;
}

float tm_rand_float(uint64_t state[2]) {
    // Generate random float in [0, 1) using top 24 bits (fits float mantissa)
    // Faster than 53-bit path and avoids precision loss on single-precision FPU
    uint32_t r24 = (uint32_t)(tm_rand(state) >> 40); // take top 24 bits
    return (float)r24 * (1.0f / 16777216.0f);       // 1 / 2^24
}

static inline uint32_t tm_rand_u24(uint64_t state[2]) {
    return (uint32_t)(tm_rand(state) >> 40);
}

static inline uint32_t tm_prob_to_u24_threshold(int numer, int denom) {
    const uint32_t P = (1u << 24);
    if (denom <= 0 || numer <= 0) return 0u;
    if (numer >= denom) return P;
    return (uint32_t)(((uint64_t)numer * (uint64_t)P) / (uint64_t)denom);
}

int tm_rand_int(uint64_t state[2], int min, int max) {
    if (min > max) {
        int temp = min;
        min = max;
        max = temp;
    }
    uint64_t range = (uint64_t)(max - min + 1);
    int val = min + (int)(tm_rand(state) % range);
    return val;
}

// ========== Memory Management ==========

uint32_t tm_memory_usage(int features, int clauses) {
    uint32_t ta_state_size = clauses * features * 2 * sizeof(uint8_t);
    uint32_t clause_output_size = clauses * sizeof(uint8_t);
    uint32_t feedback_size = clauses * sizeof(int8_t);
    uint32_t struct_size = sizeof(TsetlinMachine);
    uint32_t total = struct_size + ta_state_size + clause_output_size + feedback_size;
    return total;
}

// ========== Single TM Implementation ==========

bool tm_init(TsetlinMachine* tm, int features, int clauses, int threshold, uint32_t seed) {
    PROFILER_START();
    if (features > MAX_FEATURES || clauses > MAX_CLAUSES) {
        return false;
    }
    
    tm->num_features = features;
    tm->num_clauses = clauses;
    tm->threshold = threshold;
    tm->num_states = NUMBER_OF_STATES;
    
    // Initialize Xorshift128+ state from seed
    // Use SplitMix64-like initialization to avoid zero state
    tm->rng_state[0] = seed ^ 0x9E3779B97F4A7C15ULL;
    tm->rng_state[1] = (seed + 1) ^ 0x9E3779B97F4A7C15ULL;
    
    // Warm up the generator
    tm_rand(tm->rng_state);
    tm_rand(tm->rng_state);
    
    // Allocate memory with tracking
    int ta_size = clauses * features * 2;
    tm->ta_state = (uint8_t*)utils_malloc(ta_size * sizeof(uint8_t));
    BOUNDS_RECORD_MALLOC(ta_size, tm->ta_state != NULL);
    
    tm->clause_output = (uint8_t*)utils_malloc(clauses * sizeof(uint8_t));
    BOUNDS_RECORD_MALLOC(clauses, tm->clause_output != NULL);
    
    tm->feedback_to_clauses = (int8_t*)utils_malloc(clauses * sizeof(int8_t));
    BOUNDS_RECORD_MALLOC(clauses, tm->feedback_to_clauses != NULL);
    
    if (!tm->ta_state || !tm->clause_output || !tm->feedback_to_clauses) {
        tm_free(tm);
        return false;
    }
    
    // Initialize automata states
    for (int j = 0; j < clauses; j++) {
        for (int k = 0; k < features; k++) {
            int idx = (j * features + k) * 2;
            
            if (tm_rand_float(tm->rng_state) <= 0.5f) {
                tm->ta_state[idx] = NUMBER_OF_STATES;       // Include
                tm->ta_state[idx + 1] = NUMBER_OF_STATES + 1; // Include negated
            } else {
                tm->ta_state[idx] = NUMBER_OF_STATES + 1;   // Include
                tm->ta_state[idx + 1] = NUMBER_OF_STATES;   // Include negated
            }
        }
    }
    PROFILER_END();
    return true;
}

void tm_free(TsetlinMachine* tm) {
    if (tm->ta_state) {
        utils_free(tm->ta_state);
        BOUNDS_RECORD_FREE();
        tm->ta_state = NULL;
    }
    if (tm->clause_output) {
        utils_free(tm->clause_output);
        BOUNDS_RECORD_FREE();
        tm->clause_output = NULL;
    }
    if (tm->feedback_to_clauses) {
        utils_free(tm->feedback_to_clauses);
        BOUNDS_RECORD_FREE();
        tm->feedback_to_clauses = NULL;
    }
}

void tm_reinit(TsetlinMachine* tm) {
    for (int j = 0; j < tm->num_clauses; j++) {
        for (int k = 0; k < tm->num_features; k++) {
            int idx = (j * tm->num_features + k) * 2;
            
            if (tm_rand_float(tm->rng_state) <= 0.5f) {
                tm->ta_state[idx] = NUMBER_OF_STATES;
                tm->ta_state[idx + 1] = NUMBER_OF_STATES + 1;
            } else {
                tm->ta_state[idx] = NUMBER_OF_STATES + 1;
                tm->ta_state[idx + 1] = NUMBER_OF_STATES;
            }
        }
    }
}

// Get action from state (include=1, exclude=0)
static inline int tm_action(uint8_t state) {
    return (state > NUMBER_OF_STATES) ? 1 : 0;
}

// Calculate clause outputs
static void tm_calculate_clause_output(TsetlinMachine* tm, const uint8_t* X, int mode) {
    const int nf = tm->num_features;
    const int stride = nf * 2;
    const uint8_t* ta_base = tm->ta_state;
    for (int j = 0; j < tm->num_clauses; j++, ta_base += stride) {
        uint8_t clause_ok = 1;
        uint8_t any_include = 0;
        const uint8_t* x = X;
        const uint8_t* ta = ta_base;

        for (int k = 0; k < nf; k++, ta += 2, ++x) {

#if TM_VANILLA_BOUNDS_INNER
            // Debug-only: hot-loop bounds monitoring is expensive on embedded targets.
            BOUNDS_RECORD_ACCESS(false);
#endif

            const int action_include = tm_action(ta[0]);
            const int action_include_negated = tm_action(ta[1]);
            any_include |= (uint8_t)(action_include | action_include_negated);

            if ((action_include == 1 && *x == 0) ||
                (action_include_negated == 1 && *x == 1)) {
                clause_ok = 0;
                break;
            }
        }

        if (mode == TM_MODE_PREDICT && any_include == 0) {
            clause_ok = 0;
        }
        tm->clause_output[j] = clause_ok;
    }
}

// Calculate clause outputs from pre-packed LSB words.
static void tm_calculate_clause_output_words(TsetlinMachine* tm, const uint32_t* Xw, int mode) {
    const int nf = tm->num_features;
    const int nw = (nf + 31) >> 5;
    const int stride = nf * 2;
    const uint8_t* ta_base = tm->ta_state;
    for (int j = 0; j < tm->num_clauses; j++, ta_base += stride) {
        uint8_t clause_ok = 1;
        uint8_t any_include = 0;
        const uint8_t* ta = ta_base;

        int k = 0;
        for (int wi = 0; wi < nw && k < nf; ++wi) {
            uint32_t bits = Xw[wi];
            int lim = nf - k;
            if (lim > 32) lim = 32;
            for (int b = 0; b < lim; ++b, ++k, ta += 2) {
#if TM_VANILLA_BOUNDS_INNER
                BOUNDS_RECORD_ACCESS(false);
#endif
                const uint8_t xk = (uint8_t)(bits & 1u);
                bits >>= 1;
                const int action_include = tm_action(ta[0]);
                const int action_include_negated = tm_action(ta[1]);
                any_include |= (uint8_t)(action_include | action_include_negated);

                if ((action_include == 1 && xk == 0u) ||
                    (action_include_negated == 1 && xk == 1u)) {
                    clause_ok = 0;
                    break;
                }
            }
            if (clause_ok == 0u) break;
        }

        if (mode == TM_MODE_PREDICT && any_include == 0) {
            clause_ok = 0;
        }
        tm->clause_output[j] = clause_ok;
    }
}

// Sum up class votes
static int tm_sum_votes(TsetlinMachine* tm) {
    int class_sum = 0;
    const uint8_t* out = tm->clause_output;
    int j = 0;
    for (; j + 1 < tm->num_clauses; j += 2) {
        class_sum += (int)out[j];
        class_sum -= (int)out[j + 1];
    }
    if (j < tm->num_clauses) {
        class_sum += (int)out[j];
    }
    
    // Clip to threshold
    if (class_sum > tm->threshold) class_sum = tm->threshold;
    if (class_sum < -tm->threshold) class_sum = -tm->threshold;
    
    return class_sum;
}

// Type I feedback
static void tm_type_i_feedback(TsetlinMachine* tm, const uint8_t* X, int j, int s) {
    const uint32_t P = (1u << 24);
    const uint32_t th_1_div_s = (s > 0) ? (P / (uint32_t)s) : P;
    const uint32_t th_s1_div_s = (th_1_div_s <= P) ? (P - th_1_div_s) : 0u;
    const int nf = tm->num_features;
    uint8_t* ta = tm->ta_state + (j * nf * 2);

    if (tm->clause_output[j] == 0) {
        // If clause output is 0, penalize all literals
        for (int k = 0; k < nf; k++, ta += 2) {
            if (ta[0] > 1 && tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                ta[0]--;
            }
            if (ta[1] > 1 && tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                ta[1]--;
            }
        }
    } else if (tm->clause_output[j] == 1) {
        // If clause output is 1, reward matching literals
        const uint8_t* x = X;
        for (int k = 0; k < nf; k++, ta += 2, ++x) {
            if (*x == 1) {
                if (ta[0] < NUMBER_OF_STATES * 2 &&
                    tm_rand_u24(tm->rng_state) <= th_s1_div_s) {
                    ta[0]++;
                }
                if (ta[1] > 1 &&
                    tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                    ta[1]--;
                }
            } else {
                if (ta[1] < NUMBER_OF_STATES * 2 &&
                    tm_rand_u24(tm->rng_state) <= th_s1_div_s) {
                    ta[1]++;
                }
                if (ta[0] > 1 &&
                    tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                    ta[0]--;
                }
            }
        }
    }
}

static void tm_type_i_feedback_words(TsetlinMachine* tm, const uint32_t* Xw, int j, int s) {
    const uint32_t P = (1u << 24);
    const uint32_t th_1_div_s = (s > 0) ? (P / (uint32_t)s) : P;
    const uint32_t th_s1_div_s = (th_1_div_s <= P) ? (P - th_1_div_s) : 0u;
    const int nf = tm->num_features;
    const int nw = (nf + 31) >> 5;
    uint8_t* ta = tm->ta_state + (j * nf * 2);

    if (tm->clause_output[j] == 0) {
        for (int k = 0; k < nf; k++, ta += 2) {
            if (ta[0] > 1 && tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                ta[0]--;
            }
            if (ta[1] > 1 && tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                ta[1]--;
            }
        }
    } else if (tm->clause_output[j] == 1) {
        int k = 0;
        for (int wi = 0; wi < nw && k < nf; ++wi) {
            uint32_t bits = Xw[wi];
            int lim = nf - k;
            if (lim > 32) lim = 32;
            for (int b = 0; b < lim; ++b, ++k, ta += 2) {
                const uint8_t xk = (uint8_t)(bits & 1u);
                bits >>= 1;
                if (xk == 1u) {
                    if (ta[0] < NUMBER_OF_STATES * 2 &&
                        tm_rand_u24(tm->rng_state) <= th_s1_div_s) {
                        ta[0]++;
                    }
                    if (ta[1] > 1 &&
                        tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                        ta[1]--;
                    }
                } else {
                    if (ta[1] < NUMBER_OF_STATES * 2 &&
                        tm_rand_u24(tm->rng_state) <= th_s1_div_s) {
                        ta[1]++;
                    }
                    if (ta[0] > 1 &&
                        tm_rand_u24(tm->rng_state) <= th_1_div_s) {
                        ta[0]--;
                    }
                }
            }
        }
    }
}

// Type II feedback
static void tm_type_ii_feedback(TsetlinMachine* tm, const uint8_t* X, int j) {
    if (tm->clause_output[j] == 1) {
        const int nf = tm->num_features;
        uint8_t* ta = tm->ta_state + (j * nf * 2);
        const uint8_t* x = X;
        for (int k = 0; k < nf; k++, ta += 2, ++x) {
            int action_include = tm_action(ta[0]);
            int action_include_negated = tm_action(ta[1]);

            if (action_include == 0 && ta[0] < NUMBER_OF_STATES * 2 && *x == 0) {
                ta[0]++;
            }
            if (action_include_negated == 0 && ta[1] < NUMBER_OF_STATES * 2 && *x == 1) {
                ta[1]++;
            }
        }
    }
}

static void tm_type_ii_feedback_words(TsetlinMachine* tm, const uint32_t* Xw, int j) {
    if (tm->clause_output[j] == 1) {
        const int nf = tm->num_features;
        const int nw = (nf + 31) >> 5;
        uint8_t* ta = tm->ta_state + (j * nf * 2);
        int k = 0;
        for (int wi = 0; wi < nw && k < nf; ++wi) {
            uint32_t bits = Xw[wi];
            int lim = nf - k;
            if (lim > 32) lim = 32;
            for (int b = 0; b < lim; ++b, ++k, ta += 2) {
                const uint8_t xk = (uint8_t)(bits & 1u);
                bits >>= 1;
                int action_include = tm_action(ta[0]);
                int action_include_negated = tm_action(ta[1]);

                if (action_include == 0 && ta[0] < NUMBER_OF_STATES * 2 && xk == 0u) {
                    ta[0]++;
                }
                if (action_include_negated == 0 && ta[1] < NUMBER_OF_STATES * 2 && xk == 1u) {
                    ta[1]++;
                }
            }
        }
    }
}

int tm_update(TsetlinMachine* tm, const uint8_t* X, int target, int s) {
 
    // Calculate clause outputs
    tm_calculate_clause_output(tm, X, TM_MODE_UPDATE);
    
    // Sum up votes
    int class_sum = tm_sum_votes(tm);
    
    // Decide and apply feedback in one pass (avoids storing per-clause decisions).
    const int y = 2 * target - 1;
    const int numer = tm->threshold - y * class_sum;
    const int denom = tm->threshold * 2;
    const uint32_t th = tm_prob_to_u24_threshold(numer, denom);
    for (int j = 0; j < tm->num_clauses; j++) {
        const int clause_sign = (j & 1) ? -1 : 1;
        if (tm_rand_u24(tm->rng_state) <= th) {
            if (clause_sign == y) {
                tm_type_i_feedback(tm, X, j, s);
            } else if (tm->clause_output[j]) {
                tm_type_ii_feedback(tm, X, j);
            }
        }
    }
 
    return class_sum;
}

int tm_score(TsetlinMachine* tm, const uint8_t* X) {

    tm_calculate_clause_output(tm, X, TM_MODE_PREDICT);
    int score = tm_sum_votes(tm);
 
    return score;
}

int tm_update_words(TsetlinMachine* tm, const uint32_t* Xw, int target, int s) {
    tm_calculate_clause_output_words(tm, Xw, TM_MODE_UPDATE);
    int class_sum = tm_sum_votes(tm);

    const int y = 2 * target - 1;
    const int numer = tm->threshold - y * class_sum;
    const int denom = tm->threshold * 2;
    const uint32_t th = tm_prob_to_u24_threshold(numer, denom);
    for (int j = 0; j < tm->num_clauses; j++) {
        const int clause_sign = (j & 1) ? -1 : 1;
        if (tm_rand_u24(tm->rng_state) <= th) {
            if (clause_sign == y) {
                tm_type_i_feedback_words(tm, Xw, j, s);
            } else if (tm->clause_output[j]) {
                tm_type_ii_feedback_words(tm, Xw, j);
            }
        }
    }
    return class_sum;
}

int tm_score_words(TsetlinMachine* tm, const uint32_t* Xw) {
    tm_calculate_clause_output_words(tm, Xw, TM_MODE_PREDICT);
    int score = tm_sum_votes(tm);
    return score;
}

void tm_fit(TsetlinMachine* tm, const uint8_t** X, const uint8_t* y,
           int n_samples, int epochs, int s) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < n_samples; i++) {
            // Shuffle by random selection
            int idx = tm_rand_int(tm->rng_state, 0, n_samples - 1);
            tm_update(tm, X[idx], y[idx], s);
        }
    }
}

float tm_evaluate(TsetlinMachine* tm, const uint8_t** X, const uint8_t* y,
                 int n_samples) {
    int errors = 0;
    for (int i = 0; i < n_samples; i++) {
        int prediction = (tm_score(tm, X[i]) > 0) ? 1 : 0;
        if (prediction != y[i]) {
            errors++;
        }
    }
    float acc = 1.0f - ((float)errors / (float)n_samples);
    return acc;
}

