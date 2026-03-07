#pragma once

#include "core/protocol.h"
#include "core/transport.h"
#include "core/utils.h"
#include "debug/bounds_monitor.h"
#include "Arduino.h"
#include "config/bnn_config.h"

// Forward declarations
extern Transport TR;

// Build and send memory efficiency analysis as structured data
static inline void send_memory_efficiency_analysis(int model_feat) {
    MemEfficiencyPayload payload = {};
    
    // Use bounds monitor data which tracks BNN model size
    uint32_t bnn_model_size = g_bounds_monitor.total_allocated_by_tm; // Reuses TM field for BNN
    
    payload.tm_memory_le = bnn_model_size;
    payload.memory_per_clause = 0.0f; // Not applicable for BNN
    payload.memory_per_feature = model_feat > 0 ? (float)bnn_model_size / model_feat : 0.0f;
    payload.memory_per_cf = 0.0f;
    
    payload.free_heap_le = ESP.getFreeHeap();
    payload.max_alloc_le = ESP.getMaxAllocHeap();
    payload.fragmentation = payload.free_heap_le > 0
        ? (float)(payload.free_heap_le - payload.max_alloc_le) / payload.free_heap_le * 100.0f
        : 0.0f;
    
    payload.current_allocated_le = bnn_model_size;
    payload.peak_allocated_le = bnn_model_size;
    payload.memory_efficiency = 100.0f; // BNN uses static allocation
    
    payload.alloc_count_le = 0; // BNN uses std::unique_ptr, not tracked
    payload.free_count_le = 0;
    payload.total_allocated_le = bnn_model_size;
    payload.avg_alloc_size = (float)bnn_model_size;
    payload.alloc_frequency = 0.0f;
    
    TR.sendMemEfficiency(payload);
}

// Build and send performance benchmarks as structured data
static inline void send_performance_benchmarks(uint32_t trained, uint32_t tested, uint32_t correct, int model_feat) {
    PerfBenchPayload payload = {};
    
    uint32_t bnn_model_size = g_bounds_monitor.total_allocated_by_tm; // BNN model size
    
    if (trained > 0 && tested > 0) {
        uint64_t train_duration = 0;
        if (g_bounds_monitor.training_end_us >= g_bounds_monitor.training_start_us) {
            train_duration = g_bounds_monitor.training_end_us - g_bounds_monitor.training_start_us;
        }
        uint64_t test_duration = 0;
        if (g_bounds_monitor.testing_end_us >= g_bounds_monitor.testing_start_us) {
            test_duration = g_bounds_monitor.testing_end_us - g_bounds_monitor.testing_start_us;
        }
        
        if (train_duration > 0) {
            payload.train_throughput = (float)trained / (train_duration / 1000000.0f);
        }
        if (test_duration > 0) {
            payload.test_throughput = (float)tested / (test_duration / 1000000.0f);
        }
        
        payload.mem_per_train_sample = trained > 0 ? (float)bnn_model_size / trained : 0.0f;
        payload.mem_per_test_sample = tested > 0 ? (float)bnn_model_size / tested : 0.0f;
        payload.throughput_mem_ratio = (bnn_model_size > 0) ? payload.test_throughput / (bnn_model_size / 1024.0f) : 0.0f;
    }
    
    uint32_t chip_total_heap = ESP.getHeapSize();
    uint32_t chip_free_heap  = ESP.getFreeHeap();
    payload.chip_used_heap_le = chip_total_heap - chip_free_heap;
    payload.tm_core_memory_le = bnn_model_size;
    payload.sys_overhead_le = (payload.chip_used_heap_le > payload.tm_core_memory_le)
        ? (payload.chip_used_heap_le - payload.tm_core_memory_le) : 0;
    payload.tm_mem_percent = payload.chip_used_heap_le > 0
        ? (float)payload.tm_core_memory_le / payload.chip_used_heap_le * 100.0f : 0.0f;
    payload.overhead_percent = payload.chip_used_heap_le > 0
        ? (float)payload.sys_overhead_le / payload.chip_used_heap_le * 100.0f : 0.0f;
    payload.ta_percentage = 0.0f;
    
    payload.avg_alloc_size = (float)bnn_model_size;
    payload.alloc_overhead_pct = 0.0f;
    payload.total_overhead_le = 0;
    
    if (trained > 0) {
        payload.memory_utilization = ESP.getFreeHeap() > 0
            ? (float)bnn_model_size / ESP.getFreeHeap() * 100.0f : 0.0f;
        payload.performance_score = tested > 0 ? (float)correct / tested * 100.0f : 0.0f;
        payload.efficiency_ratio = payload.memory_utilization > 0
            ? payload.performance_score / payload.memory_utilization : 0.0f;
    }
    
    // Calculate precision, recall, and F1 score from confusion matrix
    uint32_t tp = g_bounds_monitor.true_positives;
    uint32_t fp = g_bounds_monitor.false_positives;
    uint32_t fn = g_bounds_monitor.false_negatives;
    
    payload.precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
    payload.recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
    payload.f1_score = (payload.precision + payload.recall > 0) ? 
        2.0f * payload.precision * payload.recall / (payload.precision + payload.recall) : 0.0f;
    
    TR.sendPerfBench(payload);
}

// Build and send protocol statistics as structured data
static inline void send_protocol_statistics() {
    ProtocolStatsPayload payload = {};
    
    payload.frames_received_le = g_bounds_monitor.frames_received;
    payload.frames_valid_le = g_bounds_monitor.frames_valid;
    payload.rejected_size_le = g_bounds_monitor.frames_rejected_size;
    payload.rejected_checksum_le = g_bounds_monitor.frames_rejected_checksum;
    payload.rejected_format_le = g_bounds_monitor.frames_rejected_format;
    
    if (g_bounds_monitor.frames_received > 0) {
        payload.rejection_rate = 100.0f * (payload.rejected_size_le + 
            payload.rejected_checksum_le + payload.rejected_format_le) / 
            g_bounds_monitor.frames_received;
    }
    
    TR.sendProtocolStats(payload);
}

