/*
 * Comprehensive Benchmarking System for ESP32
 * Measures time, memory, and power consumption
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdint.h>

// Benchmark result structure
typedef struct {
    uint32_t time_us;           // Execution time in microseconds
    uint32_t time_ms;           // Execution time in milliseconds
    uint32_t memory_before;     // Free heap before operation
    uint32_t memory_after;      // Free heap after operation
    uint32_t memory_used;       // Memory consumed
    uint32_t peak_memory_used;  // Peak memory usage
    float power_mw;             // Power consumption in milliwatts (if available)
    float energy_mj;            // Energy consumed in millijoules
} BenchmarkResult;

// Benchmark context for tracking
typedef struct {
    uint32_t start_time_us;
    uint32_t start_memory;
    uint32_t peak_memory;
    float start_voltage_mv;
    float start_current_ma;
} BenchmarkContext;

// Initialize benchmark system
void benchmark_init();

// Start benchmark measurement
void benchmark_start(BenchmarkContext* ctx);

// End benchmark measurement and return results
BenchmarkResult benchmark_end(BenchmarkContext* ctx);

// Print benchmark results
void benchmark_print(const char* name, BenchmarkResult* result);

// Detailed benchmark with multiple iterations
typedef struct {
    BenchmarkResult min;
    BenchmarkResult max;
    BenchmarkResult avg;
    int iterations;
} DetailedBenchmark;

// Run detailed benchmark over multiple iterations
DetailedBenchmark benchmark_detailed(void (*func)(void*), void* data, int iterations);

// Print detailed benchmark results
void benchmark_print_detailed(const char* name, DetailedBenchmark* result);

// Memory profiler
typedef struct {
    uint32_t total_allocated;
    uint32_t total_freed;
    uint32_t current_usage;
    uint32_t peak_usage;
    uint32_t allocation_count;
    uint32_t free_count;
} MemoryProfile;

// Initialize memory profiler
void memory_profile_init(MemoryProfile* profile);

// Track allocation
void memory_profile_alloc(MemoryProfile* profile, uint32_t size);

// Track free
void memory_profile_free(MemoryProfile* profile, uint32_t size);

// Print memory profile
void memory_profile_print(MemoryProfile* profile);

// Get current free heap
uint32_t benchmark_get_free_heap();

// Get current time in microseconds
uint32_t benchmark_get_time_us();

// Get current time in milliseconds
uint32_t benchmark_get_time_ms();

// Power measurement (ESP32-specific, requires INA219 or similar)
#ifdef ESP32_POWER_MONITOR
float benchmark_get_voltage_mv();
float benchmark_get_current_ma();
float benchmark_get_power_mw();
#endif

#endif // BENCHMARK_H

