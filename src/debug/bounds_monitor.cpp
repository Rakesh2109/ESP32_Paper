/*
 * Experimental Bounds Monitoring Implementation
 */

#include "debug/bounds_monitor.h"
#include <Arduino.h>
#include <string.h>
#include <stdio.h>
#include "core/utils.h"
#if defined(ESP32)
#include "esp_timer.h"
#endif
#if !defined(ARDUINO)
#include <chrono>
#endif

// LOG_SERIAL is now a global BinaryLogSerial instance from utils.h
// All output automatically routes through binary LOG frames

#if defined(ESP32)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#endif

#if ENABLE_BOUNDS_MONITORING

static inline uint64_t monotonic_time_us() {
#if defined(ESP32)
    return static_cast<uint64_t>(esp_timer_get_time());
#elif defined(ARDUINO)
    return static_cast<uint64_t>(micros());
#else
    using namespace std::chrono;
    static const auto boot_time = steady_clock::now();
    return duration_cast<microseconds>(steady_clock::now() - boot_time).count();
#endif
}

BoundsMonitor g_bounds_monitor;

void bounds_monitor_init() {
    memset(&g_bounds_monitor, 0, sizeof(BoundsMonitor));
    g_bounds_monitor.heap_min = 0xFFFFFFFF;
    g_bounds_monitor.stack_min_free = 0xFFFFFFFF;
    g_bounds_monitor.training_start_us = 0;
    g_bounds_monitor.testing_start_us = 0;
}

void bounds_monitor_reset() {
    bounds_monitor_init();
}

void bounds_record_heap_sample() {
    uint32_t free_heap = ESP.getFreeHeap();
    
    g_bounds_monitor.heap_samples++;
    g_bounds_monitor.heap_sum += free_heap;
    
    if (free_heap < g_bounds_monitor.heap_min) {
        g_bounds_monitor.heap_min = free_heap;
    }
    if (free_heap > g_bounds_monitor.heap_max) {
        g_bounds_monitor.heap_max = free_heap;
    }
    
    // Critical threshold: less than 50KB free
    if (free_heap < 51200) {
        g_bounds_monitor.heap_critical_events++;
    }
    
    // Store in history (circular buffer)
    uint32_t idx = g_bounds_monitor.history_index % 50;
    g_bounds_monitor.heap_history[idx] = free_heap;
    g_bounds_monitor.history_index++;
}

void bounds_record_stack_sample() {
#if defined(ESP32)
    UBaseType_t free_stack = uxTaskGetStackHighWaterMark(NULL);
    uint32_t used_stack = 8192 - (free_stack * 4); // Assuming 8KB stack, 4 bytes per word
    
    // Clamp/validate used_stack
    if ((int32_t)used_stack < 0) used_stack = 0;
    if (used_stack > 8192) {
        used_stack = 8192;
        LOG_SERIAL.println("[WARN] Stack usage measurement exceeded max stack size. Clamped to 8192 bytes.");
    }
    
    g_bounds_monitor.stack_samples++;
    
    if (free_stack < g_bounds_monitor.stack_min_free) {
        g_bounds_monitor.stack_min_free = free_stack;
    }
    if (used_stack > g_bounds_monitor.stack_max_used) {
        g_bounds_monitor.stack_max_used = used_stack;
    }
    
    // Warning threshold: less than 1KB free
    if (free_stack < 256) {  // 256 words = 1KB
        g_bounds_monitor.stack_overflow_warnings++;
    }
    
    // Store in history
    uint32_t idx = g_bounds_monitor.history_index % 50;
    g_bounds_monitor.stack_history[idx] = used_stack;
#endif
}

void bounds_record_array_access(bool violation) {
    g_bounds_monitor.array_accesses++;
    
    if (violation) {
        g_bounds_monitor.bounds_violations_detected++;
    } else {
        g_bounds_monitor.bounds_checks_passed++;
    }
}

void bounds_record_frame_validation(bool valid, uint8_t error_type) {
    g_bounds_monitor.frames_received++;
    
    if (valid) {
        g_bounds_monitor.frames_valid++;
    } else {
        // error_type: 1=size, 2=checksum, 3=format
        switch (error_type) {
            case 1: g_bounds_monitor.frames_rejected_size++; break;
            case 2: g_bounds_monitor.frames_rejected_checksum++; break;
            case 3: g_bounds_monitor.frames_rejected_format++; break;
        }
    }
}

void bounds_record_malloc(size_t size, bool success) {
    g_bounds_monitor.malloc_calls++;
    
    if (success) {
        if (size > g_bounds_monitor.peak_allocation_size) {
            g_bounds_monitor.peak_allocation_size = size;
        }
    } else {
        g_bounds_monitor.malloc_failures++;
    }
}

void bounds_record_free() {
    g_bounds_monitor.free_calls++;
}

// NEW: Detailed tracking implementations
void bounds_record_memory_breakdown(uint32_t automata, uint32_t clause, uint32_t feedback) {
    g_bounds_monitor.tm_automata_size = automata;
    g_bounds_monitor.tm_clause_size = clause;
    g_bounds_monitor.tm_feedback_size = feedback;
    g_bounds_monitor.total_allocated_by_tm = automata + clause + feedback;
}

void bounds_record_model_size(uint32_t size, uint8_t stage) {
    switch(stage) {
        case 0: g_bounds_monitor.model_size_after_init = size; break;
        case 1: g_bounds_monitor.model_size_after_training = size; break;
        case 2: g_bounds_monitor.model_size_before_testing = size; break;
    }
}

void bounds_record_training_start() {
    g_bounds_monitor.training_start_us = monotonic_time_us();
    g_bounds_monitor.malloc_calls_during_training = g_bounds_monitor.malloc_calls;
}

void bounds_record_training_end(uint32_t samples) {
    g_bounds_monitor.training_end_us = monotonic_time_us();
    g_bounds_monitor.training_samples = samples;
    g_bounds_monitor.malloc_calls_during_training =
        g_bounds_monitor.malloc_calls - g_bounds_monitor.malloc_calls_during_training;
}

void bounds_record_testing_start() {
    g_bounds_monitor.testing_start_us = monotonic_time_us();
    g_bounds_monitor.malloc_calls_during_testing = g_bounds_monitor.malloc_calls;
}

void bounds_record_testing_end(uint32_t samples) {
    g_bounds_monitor.testing_end_us = monotonic_time_us();
    g_bounds_monitor.testing_samples = samples;
    g_bounds_monitor.malloc_calls_during_testing =
        g_bounds_monitor.malloc_calls - g_bounds_monitor.malloc_calls_during_testing;
}

void bounds_record_classification(uint8_t predicted, uint8_t actual) {
    if (predicted == 1 && actual == 1) {
        g_bounds_monitor.true_positives++;
    } else if (predicted == 0 && actual == 0) {
        g_bounds_monitor.true_negatives++;
    } else if (predicted == 1 && actual == 0) {
        g_bounds_monitor.false_positives++;
    } else if (predicted == 0 && actual == 1) {
        g_bounds_monitor.false_negatives++;
    }
}

void bounds_record_throughput_sample(uint32_t samples_per_sec) {
    uint32_t idx = g_bounds_monitor.throughput_index % 25;
    g_bounds_monitor.throughput_samples[idx] = samples_per_sec;
    g_bounds_monitor.throughput_index++;
}

float bounds_get_heap_avg() {
    if (g_bounds_monitor.heap_samples == 0) return 0.0f;
    return (float)g_bounds_monitor.heap_sum / g_bounds_monitor.heap_samples;
}

float bounds_get_stack_usage_percent() {
    if (g_bounds_monitor.stack_max_used == 0) return 0.0f;
    return (g_bounds_monitor.stack_max_used / 8192.0f) * 100.0f;
}

float bounds_get_frame_rejection_rate() {
    if (g_bounds_monitor.frames_received == 0) return 0.0f;
    uint32_t rejected = g_bounds_monitor.frames_rejected_size + 
                        g_bounds_monitor.frames_rejected_checksum + 
                        g_bounds_monitor.frames_rejected_format;
    return (rejected / (float)g_bounds_monitor.frames_received) * 100.0f;
}

float bounds_get_bounds_check_overhead_us() {
    if (g_bounds_monitor.bounds_check_calls == 0) return 0.0f;
    return (float)g_bounds_monitor.bounds_check_time_us / g_bounds_monitor.bounds_check_calls;
}

void bounds_export_csv() {
    LOG_SERIAL.println("\n=== BOUNDS_DATA_CSV_START ===");
    
    // Header
    LOG_SERIAL.println("metric,value");
    LOG_SERIAL.printf("heap_samples,%u\n", g_bounds_monitor.heap_samples);
    LOG_SERIAL.printf("heap_min_bytes,%u\n", g_bounds_monitor.heap_min);
    LOG_SERIAL.printf("heap_max_bytes,%u\n", g_bounds_monitor.heap_max);
    LOG_SERIAL.printf("heap_avg_bytes,%.2f\n", bounds_get_heap_avg());
    LOG_SERIAL.printf("heap_critical_events,%u\n", g_bounds_monitor.heap_critical_events);
    
    LOG_SERIAL.printf("array_accesses,%llu\n", g_bounds_monitor.array_accesses);
    LOG_SERIAL.printf("bounds_checks_passed,%u\n", g_bounds_monitor.bounds_checks_passed);
    LOG_SERIAL.printf("bounds_violations,%u\n", g_bounds_monitor.bounds_violations_detected);
    
    LOG_SERIAL.printf("stack_samples,%u\n", g_bounds_monitor.stack_samples);
    LOG_SERIAL.printf("stack_min_free_bytes,%u\n", g_bounds_monitor.stack_min_free * 4);
    LOG_SERIAL.printf("stack_max_used_bytes,%u\n", g_bounds_monitor.stack_max_used);
    LOG_SERIAL.printf("stack_usage_percent,%.2f\n", bounds_get_stack_usage_percent());
    LOG_SERIAL.printf("stack_overflow_warnings,%u\n", g_bounds_monitor.stack_overflow_warnings);
    
    LOG_SERIAL.printf("frames_received,%u\n", g_bounds_monitor.frames_received);
    LOG_SERIAL.printf("frames_valid,%u\n", g_bounds_monitor.frames_valid);
    LOG_SERIAL.printf("frames_rejected_size,%u\n", g_bounds_monitor.frames_rejected_size);
    LOG_SERIAL.printf("frames_rejected_checksum,%u\n", g_bounds_monitor.frames_rejected_checksum);
    LOG_SERIAL.printf("frames_rejected_format,%u\n", g_bounds_monitor.frames_rejected_format);
    LOG_SERIAL.printf("frame_rejection_rate_percent,%.2f\n", bounds_get_frame_rejection_rate());
    
    LOG_SERIAL.printf("malloc_calls,%u\n", g_bounds_monitor.malloc_calls);
    LOG_SERIAL.printf("malloc_failures,%u\n", g_bounds_monitor.malloc_failures);
    LOG_SERIAL.printf("free_calls,%u\n", g_bounds_monitor.free_calls);
    LOG_SERIAL.printf("peak_allocation_bytes,%u\n", g_bounds_monitor.peak_allocation_size);
    
    // Time series data
    LOG_SERIAL.println("\n=== HEAP_HISTORY_CSV ===");
    LOG_SERIAL.println("sample,heap_free_bytes");
    for (int i = 0; i < 50 && i < g_bounds_monitor.history_index; i++) {
        LOG_SERIAL.printf("%d,%u\n", i, g_bounds_monitor.heap_history[i]);
    }
    
    LOG_SERIAL.println("\n=== STACK_HISTORY_CSV ===");
    LOG_SERIAL.println("sample,stack_used_bytes");
    for (int i = 0; i < 50 && i < g_bounds_monitor.history_index; i++) {
        LOG_SERIAL.printf("%d,%u\n", i, g_bounds_monitor.stack_history[i]);
    }
    
    LOG_SERIAL.println("\n=== BOUNDS_DATA_CSV_END ===");
}

void bounds_export_json() {
    LOG_SERIAL.println("\n=== BOUNDS_DATA_JSON_START ===");
    LOG_SERIAL.println("{");
    LOG_SERIAL.println("  \"bounds_monitoring\": {");
    
    LOG_SERIAL.println("    \"memory\": {");
    LOG_SERIAL.printf("      \"heap_samples\": %u,\n", g_bounds_monitor.heap_samples);
    LOG_SERIAL.printf("      \"heap_min_bytes\": %u,\n", g_bounds_monitor.heap_min);
    LOG_SERIAL.printf("      \"heap_max_bytes\": %u,\n", g_bounds_monitor.heap_max);
    LOG_SERIAL.printf("      \"heap_avg_bytes\": %.2f,\n", bounds_get_heap_avg());
    LOG_SERIAL.printf("      \"heap_critical_events\": %u\n", g_bounds_monitor.heap_critical_events);
    LOG_SERIAL.println("    },");
    
    LOG_SERIAL.println("    \"array_bounds\": {");
    LOG_SERIAL.printf("      \"total_accesses\": %llu,\n", g_bounds_monitor.array_accesses);
    LOG_SERIAL.printf("      \"checks_passed\": %u,\n", g_bounds_monitor.bounds_checks_passed);
    LOG_SERIAL.printf("      \"violations_detected\": %u\n", g_bounds_monitor.bounds_violations_detected);
    LOG_SERIAL.println("    },");
    
    LOG_SERIAL.println("    \"stack\": {");
    LOG_SERIAL.printf("      \"samples\": %u,\n", g_bounds_monitor.stack_samples);
    LOG_SERIAL.printf("      \"min_free_bytes\": %u,\n", g_bounds_monitor.stack_min_free * 4);
    LOG_SERIAL.printf("      \"max_used_bytes\": %u,\n", g_bounds_monitor.stack_max_used);
    LOG_SERIAL.printf("      \"usage_percent\": %.2f,\n", bounds_get_stack_usage_percent());
    LOG_SERIAL.printf("      \"overflow_warnings\": %u\n", g_bounds_monitor.stack_overflow_warnings);
    LOG_SERIAL.println("    },");
    
    LOG_SERIAL.println("    \"protocol\": {");
    LOG_SERIAL.printf("      \"frames_received\": %u,\n", g_bounds_monitor.frames_received);
    LOG_SERIAL.printf("      \"frames_valid\": %u,\n", g_bounds_monitor.frames_valid);
    LOG_SERIAL.printf("      \"rejected_size\": %u,\n", g_bounds_monitor.frames_rejected_size);
    LOG_SERIAL.printf("      \"rejected_checksum\": %u,\n", g_bounds_monitor.frames_rejected_checksum);
    LOG_SERIAL.printf("      \"rejected_format\": %u,\n", g_bounds_monitor.frames_rejected_format);
    LOG_SERIAL.printf("      \"rejection_rate_percent\": %.2f\n", bounds_get_frame_rejection_rate());
    LOG_SERIAL.println("    },");
    
    LOG_SERIAL.println("    \"allocations\": {");
    LOG_SERIAL.printf("      \"malloc_calls\": %u,\n", g_bounds_monitor.malloc_calls);
    LOG_SERIAL.printf("      \"malloc_failures\": %u,\n", g_bounds_monitor.malloc_failures);
    LOG_SERIAL.printf("      \"free_calls\": %u,\n", g_bounds_monitor.free_calls);
    LOG_SERIAL.printf("      \"peak_allocation_bytes\": %u\n", g_bounds_monitor.peak_allocation_size);
    LOG_SERIAL.println("    }");
    
    LOG_SERIAL.println("  }");
    LOG_SERIAL.println("}");
    LOG_SERIAL.println("=== BOUNDS_DATA_JSON_END ===");
}

void bounds_print_summary() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║         EXPERIMENTAL BOUNDS MONITORING SUMMARY          ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    LOG_SERIAL.println("\n📊 MEMORY BOUNDS:");
    LOG_SERIAL.printf("  Heap Min:           %u bytes (%.1f KB)\n", 
                  g_bounds_monitor.heap_min, g_bounds_monitor.heap_min / 1024.0f);
    LOG_SERIAL.printf("  Heap Max:           %u bytes (%.1f KB)\n", 
                  g_bounds_monitor.heap_max, g_bounds_monitor.heap_max / 1024.0f);
    LOG_SERIAL.printf("  Heap Average:       %.1f bytes (%.1f KB)\n", 
                  bounds_get_heap_avg(), bounds_get_heap_avg() / 1024.0f);
    LOG_SERIAL.printf("  Critical Events:    %u (heap < 50KB)\n", 
                  g_bounds_monitor.heap_critical_events);
    
    LOG_SERIAL.println("\n📐 ARRAY BOUNDS:");
    LOG_SERIAL.printf("  Total Accesses:     %llu\n", g_bounds_monitor.array_accesses);
    LOG_SERIAL.printf("  Checks Passed:      %u\n", g_bounds_monitor.bounds_checks_passed);
    LOG_SERIAL.printf("  Violations Found:   %u\n", g_bounds_monitor.bounds_violations_detected);
    if (g_bounds_monitor.array_accesses > 0) {
        float violation_rate = (g_bounds_monitor.bounds_violations_detected / 
                               (float)g_bounds_monitor.array_accesses) * 100.0f;
        LOG_SERIAL.printf("  Violation Rate:     %.6f%%\n", violation_rate);
    }
    
    LOG_SERIAL.println("\n📚 STACK USAGE:");
    LOG_SERIAL.printf("  Min Free:           %u bytes\n", g_bounds_monitor.stack_min_free * 4);
    LOG_SERIAL.printf("  Max Used:           %u bytes\n", g_bounds_monitor.stack_max_used);
    LOG_SERIAL.printf("  Usage Percent:      %.1f%%\n", bounds_get_stack_usage_percent());
    LOG_SERIAL.printf("  Overflow Warnings:  %u\n", g_bounds_monitor.stack_overflow_warnings);
    
    LOG_SERIAL.println("\n📡 PROTOCOL VALIDATION:");
    LOG_SERIAL.printf("  Frames Received:    %u\n", g_bounds_monitor.frames_received);
    LOG_SERIAL.printf("  Frames Valid:       %u\n", g_bounds_monitor.frames_valid);
    LOG_SERIAL.printf("  Rejected (Size):    %u\n", g_bounds_monitor.frames_rejected_size);
    LOG_SERIAL.printf("  Rejected (Checksum):%u\n", g_bounds_monitor.frames_rejected_checksum);
    LOG_SERIAL.printf("  Rejected (Format):  %u\n", g_bounds_monitor.frames_rejected_format);
    LOG_SERIAL.printf("  Rejection Rate:     %.2f%%\n", bounds_get_frame_rejection_rate());
    
    LOG_SERIAL.println("\n💾 MEMORY ALLOCATIONS:");
    LOG_SERIAL.printf("  Malloc Calls:       %u\n", g_bounds_monitor.malloc_calls);
    LOG_SERIAL.printf("  Malloc Failures:    %u\n", g_bounds_monitor.malloc_failures);
    LOG_SERIAL.printf("  Free Calls:         %u\n", g_bounds_monitor.free_calls);
    LOG_SERIAL.printf("  Peak Allocation:    %u bytes (%.1f KB)\n", 
                  g_bounds_monitor.peak_allocation_size, 
                  g_bounds_monitor.peak_allocation_size / 1024.0f);
    
    LOG_SERIAL.println("\n════════════════════════════════════════════════════════════");
}

void bounds_print_memory_breakdown() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║              MEMORY BREAKDOWN ANALYSIS                   ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    LOG_SERIAL.println("\n📦 BNN MODEL MEMORY ALLOCATION:");
    LOG_SERIAL.printf("  Model Weights:      %u bytes (%.1f KB) [%.1f%%]\n", 
                  g_bounds_monitor.tm_automata_size,
                  g_bounds_monitor.tm_automata_size / 1024.0f,
                  100.0f);
    LOG_SERIAL.printf("  Feature Buffers:    %u bytes (%.1f KB) [%.1f%%]\n",
                  g_bounds_monitor.tm_clause_size,
                  g_bounds_monitor.tm_clause_size / 1024.0f,
                  0.0f);
    LOG_SERIAL.printf("  Activation Buffers: %u bytes (%.1f KB) [%.1f%%]\n",
                  g_bounds_monitor.tm_feedback_size,
                  g_bounds_monitor.tm_feedback_size / 1024.0f,
                  0.0f);
    LOG_SERIAL.printf("  ────────────────────────────────────────\n");
    LOG_SERIAL.printf("  TOTAL MODEL SIZE:   %u bytes (%.1f KB)\n\n",
                  g_bounds_monitor.total_allocated_by_tm,
                  g_bounds_monitor.total_allocated_by_tm / 1024.0f);
    
    LOG_SERIAL.println("🎯 HARDWARE CONTEXT (ESP32-S3):");
    LOG_SERIAL.printf("  Total SRAM:         %u bytes (%.1f MB)\n",
                  ESP.getHeapSize(), ESP.getHeapSize() / 1048576.0f);
    LOG_SERIAL.printf("  Free Heap:          %u bytes (%.1f KB)\n",
                  ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0f);
    LOG_SERIAL.printf("  Largest Free Block: %u bytes (%.1f KB)\n",
                  ESP.getMaxAllocHeap(), ESP.getMaxAllocHeap() / 1024.0f);
    
    LOG_SERIAL.println("\n💾 MODEL SIZE EVOLUTION:");
    LOG_SERIAL.printf("  After Initialization:   %u bytes (%.1f KB)\n",
                  g_bounds_monitor.model_size_after_init,
                  g_bounds_monitor.model_size_after_init / 1024.0f);
    LOG_SERIAL.printf("  After Training:         %u bytes (%.1f KB)\n",
                  g_bounds_monitor.model_size_after_training,
                  g_bounds_monitor.model_size_after_training / 1024.0f);
    LOG_SERIAL.printf("  Before Testing:         %u bytes (%.1f KB)\n",
                  g_bounds_monitor.model_size_before_testing,
                  g_bounds_monitor.model_size_before_testing / 1024.0f);
    
    uint32_t growth = g_bounds_monitor.model_size_after_training - g_bounds_monitor.model_size_after_init;
    LOG_SERIAL.printf("  Growth During Training: %u bytes (%.2f%%)\n\n",
                  growth,
                  g_bounds_monitor.model_size_after_init > 0 ? growth * 100.0f / g_bounds_monitor.model_size_after_init : 0.0f);
    
    LOG_SERIAL.println("🔄 MEMORY CALL FREQUENCY:");
    LOG_SERIAL.printf("  During Initialization:  %u malloc calls\n", 
                  g_bounds_monitor.malloc_calls_during_init);
    LOG_SERIAL.printf("  During Training:        %u malloc calls\n",
                  g_bounds_monitor.malloc_calls_during_training);
    LOG_SERIAL.printf("  During Testing:         %u malloc calls\n",
                  g_bounds_monitor.malloc_calls_during_testing);
    LOG_SERIAL.printf("  Total malloc calls:     %u\n",
                  g_bounds_monitor.malloc_calls);
    LOG_SERIAL.printf("  Total free calls:       %u\n",
                  g_bounds_monitor.free_calls);
    LOG_SERIAL.printf("  Memory leaks:           %d allocations\n",
                  (int)g_bounds_monitor.malloc_calls - (int)g_bounds_monitor.free_calls);
}

void bounds_print_classification_report() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║           CLASSIFICATION REPORT (Per-Class)              ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    uint32_t tp = g_bounds_monitor.true_positives;
    uint32_t tn = g_bounds_monitor.true_negatives;
    uint32_t fp = g_bounds_monitor.false_positives;
    uint32_t fn = g_bounds_monitor.false_negatives;
    uint32_t total = tp + tn + fp + fn;
    
    LOG_SERIAL.println("\n📊 CONFUSION MATRIX:");
    LOG_SERIAL.println("                 Predicted");
    LOG_SERIAL.println("               Class 0  Class 1");
    LOG_SERIAL.println("  Actual  0    " + String(tn) + "      " + String(fp));
    LOG_SERIAL.println("          1    " + String(fn) + "      " + String(tp));
    LOG_SERIAL.println("");
    
    // Calculate metrics
    float accuracy = (tp + tn) / (float)total;
    float precision_class1 = tp / (float)(tp + fp);
    float recall_class1 = tp / (float)(tp + fn);
    float f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1);
    
    float precision_class0 = tn / (float)(tn + fn);
    float recall_class0 = tn / (float)(tn + fp);
    float f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0);
    
    LOG_SERIAL.println("📈 CLASS 0 (Normal Traffic):");
    LOG_SERIAL.printf("  Precision:  %.4f  (%.2f%%)\n", precision_class0, precision_class0 * 100);
    LOG_SERIAL.printf("  Recall:     %.4f  (%.2f%%)\n", recall_class0, recall_class0 * 100);
    LOG_SERIAL.printf("  F1-Score:   %.4f\n", f1_class0);
    LOG_SERIAL.printf("  Support:    %u samples\n\n", tn + fp);
    
    LOG_SERIAL.println("🚨 CLASS 1 (Attack Traffic):");
    LOG_SERIAL.printf("  Precision:  %.4f  (%.2f%%)\n", precision_class1, precision_class1 * 100);
    LOG_SERIAL.printf("  Recall:     %.4f  (%.2f%%)\n", recall_class1, recall_class1 * 100);
    LOG_SERIAL.printf("  F1-Score:   %.4f\n", f1_class1);
    LOG_SERIAL.printf("  Support:    %u samples\n\n", tp + fn);
    
    LOG_SERIAL.println("🎯 OVERALL METRICS:");
    LOG_SERIAL.printf("  Accuracy:   %.4f  (%.2f%%)\n", accuracy, accuracy * 100);
    LOG_SERIAL.printf("  Macro Avg:  %.4f\n", (f1_class0 + f1_class1) / 2.0f);
    LOG_SERIAL.printf("  Weighted Avg: %.4f\n", 
                  (f1_class0 * (tn+fp) + f1_class1 * (tp+fn)) / (float)total);
}

void bounds_print_scaling_analysis() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║              SCALING ANALYSIS                            ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    // Training throughput
    uint64_t train_duration_us = g_bounds_monitor.training_end_us - g_bounds_monitor.training_start_us;
    float train_duration_sec = train_duration_us / 1000000.0f;
    float train_throughput = g_bounds_monitor.training_samples / train_duration_sec;
    float train_time_per_sample = train_duration_us / (float)g_bounds_monitor.training_samples;
    
    // Testing throughput
    uint64_t test_duration_us = g_bounds_monitor.testing_end_us - g_bounds_monitor.testing_start_us;
    float test_duration_sec = test_duration_us / 1000000.0f;
    float test_throughput = g_bounds_monitor.testing_samples / test_duration_sec;
    float test_time_per_sample = test_duration_us / (float)g_bounds_monitor.testing_samples;
    
    LOG_SERIAL.println("\n⚡ TRAINING PERFORMANCE:");
    LOG_SERIAL.printf("  Total Samples:      %u\n", g_bounds_monitor.training_samples);
    LOG_SERIAL.printf("  Total Time:         %.2f seconds (%.2f minutes)\n", 
                  train_duration_sec, train_duration_sec / 60.0f);
    LOG_SERIAL.printf("  Throughput:         %.2f samples/second\n", train_throughput);
    LOG_SERIAL.printf("  Time per Sample:    %.2f μs (%.2f ms)\n", 
                  train_time_per_sample, train_time_per_sample / 1000.0f);
    
    LOG_SERIAL.println("\n🧪 TESTING PERFORMANCE:");
    LOG_SERIAL.printf("  Total Samples:      %u\n", g_bounds_monitor.testing_samples);
    LOG_SERIAL.printf("  Total Time:         %.2f seconds (%.2f minutes)\n",
                  test_duration_sec, test_duration_sec / 60.0f);
    LOG_SERIAL.printf("  Throughput:         %.2f samples/second\n", test_throughput);
    LOG_SERIAL.printf("  Time per Sample:    %.2f μs (%.2f ms)\n",
                  test_time_per_sample, test_time_per_sample / 1000.0f);
    
    LOG_SERIAL.println("\n📊 THROUGHPUT COMPARISON:");
    float speedup = test_throughput / train_throughput;
    LOG_SERIAL.printf("  Testing vs Training:  %.2fx faster\n", speedup);
    LOG_SERIAL.printf("  Training overhead:    %.2f%%\n", 
                  (train_time_per_sample - test_time_per_sample) * 100.0f / train_time_per_sample);
}

void bounds_print_performance_bounds() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║         PERFORMANCE BOUNDS (Min/Max Analysis)            ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    // Find min/max throughput from samples
    uint32_t min_throughput = 0xFFFFFFFF;
    uint32_t max_throughput = 0;
    uint32_t valid_samples = (g_bounds_monitor.throughput_index < 25) ? 
                             g_bounds_monitor.throughput_index : 25;
    
    for (uint32_t i = 0; i < valid_samples; i++) {
        uint32_t val = g_bounds_monitor.throughput_samples[i];
        if (val > 0) {
            if (val < min_throughput) min_throughput = val;
            if (val > max_throughput) max_throughput = val;
        }
    }
    
    LOG_SERIAL.println("\n📊 THROUGHPUT BOUNDS:");
    LOG_SERIAL.printf("  Minimum (Lower Bound):  %u samples/sec\n", min_throughput);
    LOG_SERIAL.printf("  Maximum (Upper Bound):  %u samples/sec\n", max_throughput);
    LOG_SERIAL.printf("  Range (Variance):       %u samples/sec\n", max_throughput - min_throughput);
    LOG_SERIAL.printf("  Stability:              %.2f%% (lower is better)\n",
                  (max_throughput - min_throughput) * 100.0f / max_throughput);
    
    LOG_SERIAL.println("\n💾 MEMORY BOUNDS:");
    LOG_SERIAL.printf("  Minimum Free Heap:      %u bytes (%.1f KB)\n",
                  g_bounds_monitor.heap_min, g_bounds_monitor.heap_min / 1024.0f);
    LOG_SERIAL.printf("  Maximum Free Heap:      %u bytes (%.1f KB)\n",
                  g_bounds_monitor.heap_max, g_bounds_monitor.heap_max / 1024.0f);
    LOG_SERIAL.printf("  Memory Range:           %u bytes (%.1f KB)\n",
                  g_bounds_monitor.heap_max - g_bounds_monitor.heap_min,
                  (g_bounds_monitor.heap_max - g_bounds_monitor.heap_min) / 1024.0f);
    LOG_SERIAL.printf("  Peak Usage:             %.2f%% of 320KB\n",
                  (1 - g_bounds_monitor.heap_min / 327680.0f) * 100);
    
    LOG_SERIAL.println("\n📚 STACK BOUNDS:");
    LOG_SERIAL.printf("  Minimum Free Stack:     %u bytes\n", 
                  g_bounds_monitor.stack_min_free * 4);
    LOG_SERIAL.printf("  Maximum Used Stack:     %u bytes\n",
                  g_bounds_monitor.stack_max_used);
    LOG_SERIAL.printf("  Peak Usage:             %.2f%% of 8KB\n",
                  g_bounds_monitor.stack_max_used * 100.0f / 8192);
    LOG_SERIAL.printf("  Safety Margin:          %.2f%%\n",
                  100 - (g_bounds_monitor.stack_max_used * 100.0f / 8192));
    
    LOG_SERIAL.println("\n🎯 ESTIMATED PERFORMANCE BOUNDS:");
    LOG_SERIAL.printf("  Lower Bound (Worst):    %u samples/sec @ peak memory\n", min_throughput);
    LOG_SERIAL.printf("  Upper Bound (Best):     %u samples/sec @ optimal memory\n", max_throughput);
    LOG_SERIAL.printf("  Expected (Average):     %.0f samples/sec\n",
                  (min_throughput + max_throughput) / 2.0f);
}

#else

BoundsMonitor g_bounds_monitor;

void bounds_monitor_init() { memset(&g_bounds_monitor, 0, sizeof(BoundsMonitor)); }
void bounds_monitor_reset() { bounds_monitor_init(); }
void bounds_record_heap_sample() {}
void bounds_record_stack_sample() {}
void bounds_record_array_access(bool) {}

void bounds_record_frame_validation(bool valid, uint8_t error_type) {
    g_bounds_monitor.frames_received++;
    if (valid) {
        g_bounds_monitor.frames_valid++;
    } else {
        switch (error_type) {
            case 1: g_bounds_monitor.frames_rejected_size++; break;
            case 2: g_bounds_monitor.frames_rejected_checksum++; break;
            case 3: g_bounds_monitor.frames_rejected_format++; break;
        }
    }
}

void bounds_record_malloc(size_t, bool) {}
void bounds_record_free() {}

void bounds_record_memory_breakdown(uint32_t automata, uint32_t clause, uint32_t feedback) {
    g_bounds_monitor.tm_automata_size = automata;
    g_bounds_monitor.tm_clause_size = clause;
    g_bounds_monitor.tm_feedback_size = feedback;
    g_bounds_monitor.total_allocated_by_tm = automata + clause + feedback;
}

void bounds_record_model_size(uint32_t size, uint8_t) {
    g_bounds_monitor.total_allocated_by_tm = size;
}

void bounds_record_training_start() {
    g_bounds_monitor.training_start_us = (uint64_t)esp_timer_get_time();
}
void bounds_record_training_end(uint32_t) {
    g_bounds_monitor.training_end_us = (uint64_t)esp_timer_get_time();
}
void bounds_record_testing_start() {
    g_bounds_monitor.testing_start_us = (uint64_t)esp_timer_get_time();
}
void bounds_record_testing_end(uint32_t) {
    g_bounds_monitor.testing_end_us = (uint64_t)esp_timer_get_time();
}

void bounds_record_classification(uint8_t predicted, uint8_t actual) {
    if (predicted == 1 && actual == 1)      g_bounds_monitor.true_positives++;
    else if (predicted == 0 && actual == 0) g_bounds_monitor.true_negatives++;
    else if (predicted == 1 && actual == 0) g_bounds_monitor.false_positives++;
    else if (predicted == 0 && actual == 1) g_bounds_monitor.false_negatives++;
}

void bounds_record_throughput_sample(uint32_t) {}
float bounds_get_heap_avg() { return 0.0f; }
float bounds_get_stack_usage_percent() { return 0.0f; }
float bounds_get_frame_rejection_rate() {
    if (g_bounds_monitor.frames_received == 0) return 0.0f;
    uint32_t rejected = g_bounds_monitor.frames_rejected_size +
                        g_bounds_monitor.frames_rejected_checksum +
                        g_bounds_monitor.frames_rejected_format;
    return (rejected / (float)g_bounds_monitor.frames_received) * 100.0f;
}
float bounds_get_bounds_check_overhead_us() { return 0.0f; }
void bounds_export_csv() {}
void bounds_export_json() {}
void bounds_print_summary() {}
void bounds_print_memory_breakdown() {}
void bounds_print_classification_report() {}
void bounds_print_scaling_analysis() {}
void bounds_print_performance_bounds() {}

#endif

