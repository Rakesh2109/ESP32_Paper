/*
 * Comprehensive Benchmarking System Implementation
 */

#include "debug/benchmark.h"
#include <stdio.h>
#include <string.h>

#ifdef ESP32
#include <Arduino.h>
#define GET_TIME_US() micros()
#define GET_TIME_MS() millis()
#define GET_FREE_HEAP() ESP.getFreeHeap()
#else
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
static uint32_t GET_TIME_US() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}
static uint32_t GET_TIME_MS() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}
static uint32_t GET_FREE_HEAP() { 
    // Rough estimate for non-ESP32 platforms
    return 1000000; // 1MB placeholder
}
#endif

// Initialize benchmark system
void benchmark_init() {
    // Platform-specific initialization
#ifdef ESP32
    // Set CPU frequency to maximum for consistent benchmarks
    // setCpuFrequencyMhz(240);
#endif
    printf("Benchmark system initialized\n");
}

// Start benchmark measurement
void benchmark_start(BenchmarkContext* ctx) {
    ctx->start_time_us = GET_TIME_US();
    ctx->start_memory = GET_FREE_HEAP();
    ctx->peak_memory = 0;
    
#ifdef ESP32_POWER_MONITOR
    ctx->start_voltage_mv = benchmark_get_voltage_mv();
    ctx->start_current_ma = benchmark_get_current_ma();
#else
    ctx->start_voltage_mv = 3300.0f; // Default 3.3V
    ctx->start_current_ma = 80.0f;   // Typical ESP32 current
#endif
}

// End benchmark measurement and return results
BenchmarkResult benchmark_end(BenchmarkContext* ctx) {
    BenchmarkResult result;
    
    uint32_t end_time_us = GET_TIME_US();
    result.time_us = end_time_us - ctx->start_time_us;
    result.time_ms = result.time_us / 1000;
    
    result.memory_after = GET_FREE_HEAP();
    result.memory_before = ctx->start_memory;
    
    if (result.memory_before > result.memory_after) {
        result.memory_used = result.memory_before - result.memory_after;
    } else {
        result.memory_used = 0;
    }
    
    result.peak_memory_used = ctx->peak_memory;
    
#ifdef ESP32_POWER_MONITOR
    float end_voltage_mv = benchmark_get_voltage_mv();
    float end_current_ma = benchmark_get_current_ma();
    
    float avg_voltage_mv = (ctx->start_voltage_mv + end_voltage_mv) / 2.0f;
    float avg_current_ma = (ctx->start_current_ma + end_current_ma) / 2.0f;
    
    result.power_mw = (avg_voltage_mv * avg_current_ma) / 1000.0f;
    result.energy_mj = result.power_mw * (result.time_ms / 1000.0f);
#else
    // Estimate power consumption (ESP32 typical values)
    // Active mode: ~160mA @ 240MHz, ~80mA @ 160MHz
    // Light sleep: ~0.8mA, Deep sleep: ~0.15mA
    float estimated_current_ma = 80.0f; // Typical active current
    float voltage_mv = 3300.0f;         // 3.3V supply
    
    result.power_mw = (voltage_mv * estimated_current_ma) / 1000.0f;
    result.energy_mj = result.power_mw * (result.time_ms / 1000.0f);
#endif
    
    return result;
}

// Print benchmark results (compact)
void benchmark_print(const char* name, BenchmarkResult* result) {
    printf("%s: %u ms, %.2f KB mem, %.2f mW\n", 
           name, result->time_ms, result->memory_used / 1024.0f, result->power_mw);
}

// Run detailed benchmark over multiple iterations
DetailedBenchmark benchmark_detailed(void (*func)(void*), void* data, int iterations) {
    DetailedBenchmark detailed;
    detailed.iterations = iterations;
    
    // Initialize min/max
    memset(&detailed.min, 0xFF, sizeof(BenchmarkResult)); // Set to max values
    memset(&detailed.max, 0, sizeof(BenchmarkResult));
    memset(&detailed.avg, 0, sizeof(BenchmarkResult));
    
    uint64_t total_time_us = 0;
    uint64_t total_memory_used = 0;
    uint64_t total_power_mw = 0;
    
    for (int i = 0; i < iterations; i++) {
        BenchmarkContext ctx;
        benchmark_start(&ctx);
        
        func(data);
        
        BenchmarkResult result = benchmark_end(&ctx);
        
        // Update min
        if (result.time_us < detailed.min.time_us) detailed.min.time_us = result.time_us;
        if (result.memory_used < detailed.min.memory_used) detailed.min.memory_used = result.memory_used;
        
        // Update max
        if (result.time_us > detailed.max.time_us) detailed.max.time_us = result.time_us;
        if (result.memory_used > detailed.max.memory_used) detailed.max.memory_used = result.memory_used;
        
        // Accumulate for average
        total_time_us += result.time_us;
        total_memory_used += result.memory_used;
        total_power_mw += (uint64_t)(result.power_mw * 100); // Scale to avoid float precision issues
    }
    
    // Calculate averages
    detailed.avg.time_us = (uint32_t)(total_time_us / iterations);
    detailed.avg.time_ms = detailed.avg.time_us / 1000;
    detailed.avg.memory_used = (uint32_t)(total_memory_used / iterations);
    detailed.avg.power_mw = (float)(total_power_mw / iterations) / 100.0f;
    
    detailed.min.time_ms = detailed.min.time_us / 1000;
    detailed.max.time_ms = detailed.max.time_us / 1000;
    
    return detailed;
}

// Print detailed benchmark results
void benchmark_print_detailed(const char* name, DetailedBenchmark* result) {
    printf("\n========== Detailed Benchmark: %s ==========\n", name);
    printf("Iterations: %d\n\n", result->iterations);
    
    printf("Time (us):\n");
    printf("  Min: %u\n", result->min.time_us);
    printf("  Max: %u\n", result->max.time_us);
    printf("  Avg: %u\n", result->avg.time_us);
    
    printf("\nMemory (bytes):\n");
    printf("  Min: %u (%.2f KB)\n", result->min.memory_used, result->min.memory_used / 1024.0f);
    printf("  Max: %u (%.2f KB)\n", result->max.memory_used, result->max.memory_used / 1024.0f);
    printf("  Avg: %u (%.2f KB)\n", result->avg.memory_used, result->avg.memory_used / 1024.0f);
    
    printf("\nPower (mW):\n");
    printf("  Avg: %.2f\n", result->avg.power_mw);
    printf("=============================================\n\n");
}

// Memory profiler
void memory_profile_init(MemoryProfile* profile) {
    memset(profile, 0, sizeof(MemoryProfile));
}

void memory_profile_alloc(MemoryProfile* profile, uint32_t size) {
    profile->total_allocated += size;
    profile->current_usage += size;
    profile->allocation_count++;
    
    if (profile->current_usage > profile->peak_usage) {
        profile->peak_usage = profile->current_usage;
    }
}

void memory_profile_free(MemoryProfile* profile, uint32_t size) {
    profile->total_freed += size;
    if (profile->current_usage >= size) {
        profile->current_usage -= size;
    }
    profile->free_count++;
}

void memory_profile_print(MemoryProfile* profile) {
    printf("\n========== Memory Profile ==========\n");
    printf("Total Allocated:   %u bytes (%.2f KB)\n", 
           profile->total_allocated, profile->total_allocated / 1024.0f);
    printf("Total Freed:       %u bytes (%.2f KB)\n",
           profile->total_freed, profile->total_freed / 1024.0f);
    printf("Current Usage:     %u bytes (%.2f KB)\n",
           profile->current_usage, profile->current_usage / 1024.0f);
    printf("Peak Usage:        %u bytes (%.2f KB)\n",
           profile->peak_usage, profile->peak_usage / 1024.0f);
    printf("Allocations:       %u\n", profile->allocation_count);
    printf("Frees:             %u\n", profile->free_count);
    printf("Memory Leaks:      %u bytes\n", 
           profile->total_allocated - profile->total_freed);
    printf("=====================================\n\n");
}

// Get current free heap
uint32_t benchmark_get_free_heap() {
    return GET_FREE_HEAP();
}

// Get current time in microseconds
uint32_t benchmark_get_time_us() {
    return GET_TIME_US();
}

// Get current time in milliseconds
uint32_t benchmark_get_time_ms() {
    return GET_TIME_MS();
}

// Power measurement (ESP32-specific with INA219)
#ifdef ESP32_POWER_MONITOR
// These would require INA219 or similar power monitor
float benchmark_get_voltage_mv() {
    // TODO: Implement actual voltage reading from INA219
    return 3300.0f; // Default 3.3V
}

float benchmark_get_current_ma() {
    // TODO: Implement actual current reading from INA219
    return 80.0f; // Typical ESP32 current
}

float benchmark_get_power_mw() {
    return (benchmark_get_voltage_mv() * benchmark_get_current_ma()) / 1000.0f;
}
#endif

