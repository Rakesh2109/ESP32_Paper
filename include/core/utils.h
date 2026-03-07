/*
 * Lightweight profiling and memory tracking utilities for ESP32 (Arduino/ESP-IDF)
 * Usage in a function:
 *   PROFILER_START();
 *   ... body ...
 *   PROFILER_END(); // no-op; RAII will record duration at scope exit
 * 
 * Memory tracking:
 *   void* ptr = utils_malloc(size);
 *   utils_free(ptr);
 *   utils_print_memory_stats();
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <esp_timer.h>
#include <stdlib.h>

#ifndef PROFILER_MAX_ENTRIES
#define PROFILER_MAX_ENTRIES 64
#endif

#ifndef MEMORY_TRACKER_MAX_ENTRIES
#define MEMORY_TRACKER_MAX_ENTRIES 128
#endif

typedef struct {
    const char* name;
    uint64_t total_us;
    uint32_t count;
    uint64_t last_us;
    uint64_t max_us;
} ProfilerEntry;

// Global profiler registry (single instance defined in utils.cpp)
extern ProfilerEntry __profiler_entries__[PROFILER_MAX_ENTRIES];
extern size_t __profiler_entry_count__;

static inline uint64_t profiler_now_us() {
    return (uint64_t)esp_timer_get_time();
}

static inline void profiler_record(const char* name, uint64_t dur_us) {
    size_t idx = 0;
    for (; idx < __profiler_entry_count__; ++idx) {
        // Pointer equality fast path, fallback to strcmp if different storage
        if (__profiler_entries__[idx].name == name ||
            (__profiler_entries__[idx].name && name && strcmp(__profiler_entries__[idx].name, name) == 0)) {
            break;
        }
    }
    if (idx == __profiler_entry_count__) {
        if (__profiler_entry_count__ >= PROFILER_MAX_ENTRIES) return; // drop if full
        __profiler_entries__[idx].name = name;
        __profiler_entries__[idx].total_us = 0;
        __profiler_entries__[idx].count = 0;
        __profiler_entries__[idx].last_us = 0;
        __profiler_entries__[idx].max_us = 0;
        __profiler_entry_count__++;
    }
    __profiler_entries__[idx].total_us += dur_us;
    __profiler_entries__[idx].last_us = dur_us;
    __profiler_entries__[idx].count += 1;
    if (dur_us > __profiler_entries__[idx].max_us) __profiler_entries__[idx].max_us = dur_us;
}

class ProfilerScope {
public:
    explicit ProfilerScope(const char* func_name)
        : name(func_name), start_us(profiler_now_us()) {}
    ~ProfilerScope() {
        profiler_record(name, profiler_now_us() - start_us);
    }
private:
    const char* name;
    uint64_t start_us;
};

#define PROFILER_START() ProfilerScope __profiler_scope__(__FUNCTION__)
#define PROFILER_END() do { } while (0)

static inline void profiler_reset() {
    __profiler_entry_count__ = 0;
}

static inline const ProfilerEntry* profiler_get_entries(size_t* out_count) {
    if (out_count) *out_count = __profiler_entry_count__;
    return __profiler_entries__;
}

// Profiler control functions
void profiler_print_stats();
void profiler_print_entries();
void profiler_print_summary();
void utils_print_performance_analysis();
uint32_t profiler_get_entry_count();
uint64_t profiler_get_total_time();
float profiler_get_average_time(const char* name);
uint64_t profiler_get_max_time(const char* name);
uint32_t profiler_get_call_count(const char* name);

// ========== Memory Tracking ==========

typedef struct {
    void* ptr;
    size_t size;
    const char* file;
    int line;
    uint64_t timestamp;
} MemoryEntry;

typedef struct {
    uint64_t total_allocated;
    uint64_t total_freed;
    uint64_t current_allocated;
    uint64_t peak_allocated;
    uint32_t allocation_count;
    uint32_t free_count;
    uint32_t active_allocations;
    MemoryEntry entries[MEMORY_TRACKER_MAX_ENTRIES];
    uint32_t entry_count;
} MemoryTracker;

// Global memory tracker instance
extern MemoryTracker __memory_tracker__;

// Memory tracking functions
void utils_memory_init();
void utils_print_memory_stats();
void utils_print_memory_entries();
void utils_print_memory_breakdown();
void utils_memory_reset();
uint64_t utils_get_total_allocated();
uint64_t utils_get_total_freed();
uint64_t utils_get_current_allocated();
uint64_t utils_get_peak_allocated();
uint32_t utils_get_allocation_count();
uint32_t utils_get_free_count();
uint32_t utils_get_active_allocations();

// Convenience macros for automatic file/line tracking
#define utils_malloc(size) utils_malloc_internal(size, __FILE__, __LINE__)
#define utils_realloc(ptr, size) utils_realloc_internal(ptr, size, __FILE__, __LINE__)
#define utils_free(ptr) utils_free_internal(ptr, __FILE__, __LINE__)

// Internal function declarations (not macros)
void* utils_malloc_internal(size_t size, const char* file, int line);
void* utils_realloc_internal(void* ptr, size_t size, const char* file, int line);
void utils_free_internal(void* ptr, const char* file, int line);

// Binary protocol-safe logging (defined in main.cpp, callable from anywhere)
#ifdef __cplusplus
extern "C" {
#endif
void utils_log_binary(const char* msg, uint16_t len);
void utils_log_printf(const char* fmt, ...);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <Arduino.h>

// Binary-protocol-safe logging wrapper class
// Accumulates output into lines and sends each line as a binary LOG frame
class BinaryLogSerial : public Print {
  char line_buf[256];
  size_t line_pos;
public:
  BinaryLogSerial();
  size_t write(uint8_t c) override;
  size_t write(const uint8_t *buffer, size_t size) override;
  void flush_line();
  void flush() { flush_line(); }  // Standard flush method
};

// Global instance accessible from all translation units
extern BinaryLogSerial LOG_SERIAL;
#endif

#endif // UTILS_H


