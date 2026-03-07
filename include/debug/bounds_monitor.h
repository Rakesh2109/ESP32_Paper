/*
 * Experimental Bounds Monitoring System
 * Collects real-time data for bounds analysis graphs
 */

#ifndef BOUNDS_MONITOR_H
#define BOUNDS_MONITOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>  // For size_t

// Enable experimental bounds monitoring
#ifndef ENABLE_BOUNDS_MONITORING
#define ENABLE_BOUNDS_MONITORING 1
#endif

// Bounds monitoring statistics
typedef struct {
    // Memory bounds
    uint32_t heap_samples;
    uint32_t heap_min;
    uint32_t heap_max;
    uint64_t heap_sum;
    uint32_t heap_critical_events;  // < 50KB free
    
    // Array access tracking
    uint64_t array_accesses;
    uint32_t bounds_checks_passed;
    uint32_t bounds_violations_detected;
    
    // Stack monitoring
    uint32_t stack_samples;
    uint32_t stack_min_free;
    uint32_t stack_max_used;
    uint32_t stack_overflow_warnings;
    
    // Protocol validation
    uint32_t frames_received;
    uint32_t frames_valid;
    uint32_t frames_rejected_size;
    uint32_t frames_rejected_checksum;
    uint32_t frames_rejected_format;
    
    // Memory allocations
    uint32_t malloc_calls;
    uint32_t malloc_failures;
    uint32_t free_calls;
    uint32_t peak_allocation_size;
    
    // Performance impact
    uint64_t bounds_check_time_us;
    uint32_t bounds_check_calls;
    
    // Time series data (last 50 samples - reduced for memory efficiency)
    uint32_t heap_history[50];
    uint32_t stack_history[50];
    uint32_t history_index;
    
    // MEMORY BREAKDOWN
    uint32_t tm_automata_size;
    uint32_t tm_clause_size;
    uint32_t tm_feedback_size;
    uint32_t queue_size;
    uint32_t protocol_buffers_size;
    uint32_t total_allocated_by_tm;
    
    // THROUGHPUT TRACKING
    uint64_t training_start_us;
    uint64_t training_end_us;
    uint32_t training_samples;
    uint64_t testing_start_us;
    uint64_t testing_end_us;
    uint32_t testing_samples;
    uint32_t throughput_samples[25];  // Store last 25 throughput measurements - reduced for memory efficiency
    uint32_t throughput_index;
    
    // CLASSIFICATION METRICS
    uint32_t true_positives;
    uint32_t true_negatives;
    uint32_t false_positives;
    uint32_t false_negatives;
    
    // MODEL SIZE TRACKING
    uint32_t model_size_after_init;
    uint32_t model_size_after_training;
    uint32_t model_size_before_testing;
    
    // MEMORY CALL FREQUENCY
    uint32_t malloc_calls_during_init;
    uint32_t malloc_calls_during_training;
    uint32_t malloc_calls_during_testing;
    uint64_t total_bytes_allocated;
    uint64_t total_bytes_freed;
    
} BoundsMonitor;

// Global bounds monitor instance
extern BoundsMonitor g_bounds_monitor;

// Initialization
void bounds_monitor_init();
void bounds_monitor_reset();

// Recording functions
void bounds_record_heap_sample();
void bounds_record_stack_sample();
void bounds_record_array_access(bool violation);
void bounds_record_frame_validation(bool valid, uint8_t error_type);
void bounds_record_malloc(size_t size, bool success);
void bounds_record_free();

// NEW: Detailed tracking functions
void bounds_record_memory_breakdown(uint32_t automata, uint32_t clause, uint32_t feedback);
void bounds_record_model_size(uint32_t size, uint8_t stage);  // stage: 0=init, 1=after_train, 2=before_test
void bounds_record_training_start();
void bounds_record_training_end(uint32_t samples);
void bounds_record_testing_start();
void bounds_record_testing_end(uint32_t samples);
void bounds_record_classification(uint8_t predicted, uint8_t actual);
void bounds_record_throughput_sample(uint32_t samples_per_sec);

// Query functions
float bounds_get_heap_avg();
float bounds_get_stack_usage_percent();
float bounds_get_frame_rejection_rate();
float bounds_get_bounds_check_overhead_us();

// Export functions for graphing
void bounds_export_csv();
void bounds_export_json();
void bounds_print_summary();
void bounds_print_memory_breakdown();
void bounds_print_classification_report();
void bounds_print_scaling_analysis();
void bounds_print_performance_bounds();

// Macros for easy integration
#if ENABLE_BOUNDS_MONITORING
#define BOUNDS_RECORD_HEAP() bounds_record_heap_sample()
#define BOUNDS_RECORD_STACK() bounds_record_stack_sample()
#define BOUNDS_RECORD_ACCESS(v) bounds_record_array_access(v)
#define BOUNDS_RECORD_FRAME(valid, err) bounds_record_frame_validation(valid, err)
#define BOUNDS_RECORD_MALLOC(sz, ok) bounds_record_malloc(sz, ok)
#define BOUNDS_RECORD_FREE() bounds_record_free()
#else
#define BOUNDS_RECORD_HEAP() ((void)0)
#define BOUNDS_RECORD_STACK() ((void)0)
#define BOUNDS_RECORD_ACCESS(v) ((void)0)
#define BOUNDS_RECORD_FRAME(valid, err) ((void)0)
#define BOUNDS_RECORD_MALLOC(sz, ok) ((void)0)
#define BOUNDS_RECORD_FREE() ((void)0)
#endif

#endif // BOUNDS_MONITOR_H

