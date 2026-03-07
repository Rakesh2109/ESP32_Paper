/*
 * ESP32-Optimized Tsetlin Machine Implementation
 * Memory-efficient with minimal dynamic allocation
 */

#ifndef TSETLIN_MACHINE_ESP32_H
#define TSETLIN_MACHINE_ESP32_H

#include <stdint.h>
#include <stdbool.h>

// Configuration limits for ESP32
#ifndef MAX_FEATURES
#define MAX_FEATURES 128
#endif
#ifndef MAX_CLAUSES
#define MAX_CLAUSES 1000
#endif
#ifndef NUMBER_OF_STATES
#define NUMBER_OF_STATES 100
#endif

// Mode flags
#define TM_MODE_PREDICT 1
#define TM_MODE_UPDATE  0

// Tsetlin Machine structure (single class)
typedef struct {
    int num_features;
    int num_clauses;
    int threshold;
    int num_states;
    
    // Automata states: [clause][feature][type] where type: 0=include, 1=include_negated
    // Using 8-bit integers to save memory (max 255 states)
    uint8_t* ta_state;  // Flattened: [clause * feature * 2]
    
    // Clause outputs (computed during forward pass)
    uint8_t* clause_output;
    
    // Feedback to clauses
    int8_t* feedback_to_clauses;
    
    // RNG state (Xorshift128+)
    uint64_t rng_state[2];
} TsetlinMachine;

// ========== Single TM Functions ==========

// Initialize Tsetlin Machine
bool tm_init(TsetlinMachine* tm, int features, int clauses, int threshold, uint32_t seed);

// Free Tsetlin Machine memory
void tm_free(TsetlinMachine* tm);

// Reinitialize automata states
void tm_reinit(TsetlinMachine* tm);

// Update with single training example; returns class sum (pre-clipped)
int tm_update(TsetlinMachine* tm, const uint8_t* X, int target, int s);
int tm_update_words(TsetlinMachine* tm, const uint32_t* Xw, int target, int s);

// Predict/score for single example
int tm_score(TsetlinMachine* tm, const uint8_t* X);
int tm_score_words(TsetlinMachine* tm, const uint32_t* Xw);

// Batch training
void tm_fit(TsetlinMachine* tm, const uint8_t** X, const uint8_t* y, 
           int n_samples, int epochs, int s);

// Evaluate accuracy
float tm_evaluate(TsetlinMachine* tm, const uint8_t** X, const uint8_t* y, 
                 int n_samples);

// ========== Helper Functions ==========

// Fast RNG (Xorshift128+)
uint64_t tm_rand(uint64_t state[2]);

// Random float [0, 1)
float tm_rand_float(uint64_t state[2]);

// Random int in range [min, max]
int tm_rand_int(uint64_t state[2], int min, int max);

// Get memory usage estimate
uint32_t tm_memory_usage(int features, int clauses);

#endif // TSETLIN_MACHINE_ESP32_H


