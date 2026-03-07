/*
 * Utils Implementation - Profiling and Memory Tracking
 */

#include "core/utils.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <Arduino.h>

// ========== Binary Logging Implementation ==========

// BinaryLogSerial class implementation
BinaryLogSerial::BinaryLogSerial() : line_pos(0) {}

size_t BinaryLogSerial::write(uint8_t c) {
  if (c == '\n' || c == '\r') {
    if (line_pos > 0) {
      utils_log_binary(line_buf, line_pos);
      line_pos = 0;
    }
  } else if (line_pos < sizeof(line_buf) - 1) {
    line_buf[line_pos++] = c;
  } else {
    // Buffer full, send it
    utils_log_binary(line_buf, line_pos);
    line_pos = 0;
    line_buf[line_pos++] = c;
  }
  return 1;
}

size_t BinaryLogSerial::write(const uint8_t *buffer, size_t size) {
  for (size_t i = 0; i < size; i++) {
    write(buffer[i]);
  }
  return size;
}

void BinaryLogSerial::flush_line() {
  if (line_pos > 0) {
    utils_log_binary(line_buf, line_pos);
    line_pos = 0;
  }
}

// Global LOG_SERIAL instance - replaces Serial for binary-safe logging
BinaryLogSerial LOG_SERIAL;

// Printf-style binary logging function
extern "C" void utils_log_printf(const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  
  if (n > 0 && n < (int)sizeof(buf)) {
    utils_log_binary(buf, (uint16_t)n);
  }
}

// Global profiler registry (shared across all translation units)
ProfilerEntry __profiler_entries__[PROFILER_MAX_ENTRIES];
size_t __profiler_entry_count__ = 0;

// Global memory tracker instance
MemoryTracker __memory_tracker__;

// Initialize memory tracking
void utils_memory_init() {
    memset(&__memory_tracker__, 0, sizeof(MemoryTracker));
}

// Memory allocation with tracking
void* utils_malloc_internal(size_t size, const char* file, int line) {
    // Allocate memory using standard malloc
    void* ptr = malloc(size);
    
    if (ptr == NULL) {
        return NULL;
    }
    
    // Update tracker statistics
    __memory_tracker__.total_allocated += size;
    __memory_tracker__.current_allocated += size;
    __memory_tracker__.allocation_count++;
    __memory_tracker__.active_allocations++;
    
    // Update peak allocation
    if (__memory_tracker__.current_allocated > __memory_tracker__.peak_allocated) {
        __memory_tracker__.peak_allocated = __memory_tracker__.current_allocated;
    }
    
    // Add entry to tracking list if space available
    if (__memory_tracker__.entry_count < MEMORY_TRACKER_MAX_ENTRIES) {
        MemoryEntry* entry = &__memory_tracker__.entries[__memory_tracker__.entry_count];
        entry->ptr = ptr;
        entry->size = size;
        entry->file = file;
        entry->line = line;
        entry->timestamp = profiler_now_us();
        __memory_tracker__.entry_count++;
    }
    
    return ptr;
}

// Memory deallocation with tracking
void utils_free_internal(void* ptr, const char* file, int line) {
    if (ptr == NULL) {
        return;
    }
    
    // Find the entry for this pointer
    size_t freed_size = 0;
    bool found = false;
    
    for (uint32_t i = 0; i < __memory_tracker__.entry_count; i++) {
        if (__memory_tracker__.entries[i].ptr == ptr) {
            freed_size = __memory_tracker__.entries[i].size;
            found = true;
            
            // Remove entry by shifting remaining entries
            for (uint32_t j = i; j < __memory_tracker__.entry_count - 1; j++) {
                __memory_tracker__.entries[j] = __memory_tracker__.entries[j + 1];
            }
            __memory_tracker__.entry_count--;
            break;
        }
    }
    
    // Update tracker statistics
    if (found) {
        __memory_tracker__.total_freed += freed_size;
        __memory_tracker__.current_allocated -= freed_size;
        __memory_tracker__.free_count++;
        __memory_tracker__.active_allocations--;
    }
    
    // Free the memory
    free(ptr);
}

void* utils_realloc_internal(void* ptr, size_t size, const char* file, int line) {
    // If ptr is NULL, this is equivalent to malloc
    if (ptr == NULL) {
        return utils_malloc_internal(size, file, line);
    }
    
    // If size is 0, this is equivalent to free
    if (size == 0) {
        utils_free_internal(ptr, file, line);
        return NULL;
    }
    
    // Find the old size
    size_t old_size = 0;
    for (uint32_t i = 0; i < __memory_tracker__.entry_count; i++) {
        if (__memory_tracker__.entries[i].ptr == ptr) {
            old_size = __memory_tracker__.entries[i].size;
            break;
        }
    }
    
    // Reallocate memory
    void* new_ptr = realloc(ptr, size);
    
    if (new_ptr == NULL) {
        return NULL;
    }
    
    // Update tracker: adjust sizes
    for (uint32_t i = 0; i < __memory_tracker__.entry_count; i++) {
        if (__memory_tracker__.entries[i].ptr == ptr) {
            // Update the entry with new pointer and size
            __memory_tracker__.entries[i].ptr = new_ptr;
            __memory_tracker__.entries[i].size = size;
            __memory_tracker__.entries[i].file = file;
            __memory_tracker__.entries[i].line = line;
            
            // Update statistics
            if (size > old_size) {
                __memory_tracker__.total_allocated += (size - old_size);
                __memory_tracker__.current_allocated += (size - old_size);
            } else {
                __memory_tracker__.total_freed += (old_size - size);
                __memory_tracker__.current_allocated -= (old_size - size);
            }
            
            // Update peak allocation
            if (__memory_tracker__.current_allocated > __memory_tracker__.peak_allocated) {
                __memory_tracker__.peak_allocated = __memory_tracker__.current_allocated;
            }
            
            return new_ptr;
        }
    }
    
    // If we reach here, the pointer wasn't tracked (shouldn't happen)
    // Just return the new pointer
    return new_ptr;
}

// Print memory statistics with rich formatting
void utils_print_memory_stats() {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              UTILS MEMORY STATISTICS SUMMARY            ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    printf("\n💾 MEMORY ALLOCATION TRACKING:\n");
    printf("  Total Allocated:    %llu bytes (%.2f KB)\n", 
           (unsigned long long)__memory_tracker__.total_allocated, 
           __memory_tracker__.total_allocated / 1024.0f);
    printf("  Total Freed:        %llu bytes (%.2f KB)\n",
           (unsigned long long)__memory_tracker__.total_freed,
           __memory_tracker__.total_freed / 1024.0f);
    printf("  Current Allocated:  %llu bytes (%.2f KB)\n",
           (unsigned long long)__memory_tracker__.current_allocated,
           __memory_tracker__.current_allocated / 1024.0f);
    printf("  Peak Allocated:     %llu bytes (%.2f KB)\n",
           (unsigned long long)__memory_tracker__.peak_allocated,
           __memory_tracker__.peak_allocated / 1024.0f);
    
    printf("\n📊 ALLOCATION STATISTICS:\n");
    printf("  Total Allocations:  %u\n", __memory_tracker__.allocation_count);
    printf("  Total Frees:        %u\n", __memory_tracker__.free_count);
    printf("  Active Allocations: %u\n", __memory_tracker__.active_allocations);
    
    // Calculate memory efficiency
    uint64_t memory_leaks = __memory_tracker__.total_allocated - __memory_tracker__.total_freed;
    float efficiency = (__memory_tracker__.total_freed / (float)__memory_tracker__.total_allocated) * 100.0f;
    
    printf("\n🔍 MEMORY HEALTH ANALYSIS:\n");
    // printf("  Memory Leaks:       %llu bytes (%.2f KB)\n", 
    //        (unsigned long long)memory_leaks, memory_leaks / 1024.0f);
    // printf("  Free Efficiency:    %.2f%%\n", efficiency);
    
    // if (memory_leaks == 0) {
    //     printf("  Status:             ✅ No memory leaks detected\n");
    // } else {
    //     printf("  Status:             ⚠️  Memory leaks detected\n");
    // }
    
    printf("\n════════════════════════════════════════════════════════════\n\n");
}

// Print active memory entries with rich formatting
void utils_print_memory_entries() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║              ACTIVE MEMORY ALLOCATIONS                   ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    if (__memory_tracker__.entry_count == 0) {
        LOG_SERIAL.println("\n✅ No active memory allocations.");
    } else {
        LOG_SERIAL.printf("\n📋 Active Allocations: %u\n", __memory_tracker__.entry_count);
        LOG_SERIAL.println("Address     | Size      | File:Line");
        LOG_SERIAL.println("------------|-----------|----------");
        for (uint32_t i = 0; i < __memory_tracker__.entry_count; i++) {
            MemoryEntry* entry = &__memory_tracker__.entries[i];
            LOG_SERIAL.printf("%p | %8zu | %s:%d\n", 
                   entry->ptr, entry->size, entry->file, entry->line);
        }
    }
    LOG_SERIAL.println("\n════════════════════════════════════════════════════════════\n");
}

// Print detailed memory breakdown analysis
void utils_print_memory_breakdown() {
    LOG_SERIAL.println("\n╔══════════════════════════════════════════════════════════╗");
    LOG_SERIAL.println("║              UTILS MEMORY BREAKDOWN ANALYSIS             ║");
    LOG_SERIAL.println("╚══════════════════════════════════════════════════════════╝");
    
    LOG_SERIAL.println("\n💾 MEMORY ALLOCATION BREAKDOWN:");
    LOG_SERIAL.printf("  Total Allocated:    %llu bytes (%.2f KB) [100.0%%]\n", 
           (unsigned long long)__memory_tracker__.total_allocated, 
           __memory_tracker__.total_allocated / 1024.0f);
    LOG_SERIAL.printf("  Total Freed:        %llu bytes (%.2f KB) [%.1f%%]\n",
           (unsigned long long)__memory_tracker__.total_freed,
           __memory_tracker__.total_freed / 1024.0f,
           __memory_tracker__.total_allocated > 0 ? 
           (__memory_tracker__.total_freed * 100.0f / __memory_tracker__.total_allocated) : 0.0f);
    LOG_SERIAL.printf("  Currently Active:   %llu bytes (%.2f KB) [%.1f%%]\n",
           (unsigned long long)__memory_tracker__.current_allocated,
           __memory_tracker__.current_allocated / 1024.0f,
           __memory_tracker__.total_allocated > 0 ? 
           (__memory_tracker__.current_allocated * 100.0f / __memory_tracker__.total_allocated) : 0.0f);
    LOG_SERIAL.printf("  Peak Usage:         %llu bytes (%.2f KB) [%.1f%%]\n",
           (unsigned long long)__memory_tracker__.peak_allocated,
           __memory_tracker__.peak_allocated / 1024.0f,
           __memory_tracker__.total_allocated > 0 ? 
           (__memory_tracker__.peak_allocated * 100.0f / __memory_tracker__.total_allocated) : 0.0f);
    
    LOG_SERIAL.println("\n📊 ALLOCATION PATTERNS:");
    LOG_SERIAL.printf("  Allocation Count:   %u\n", __memory_tracker__.allocation_count);
    LOG_SERIAL.printf("  Free Count:         %u\n", __memory_tracker__.free_count);
    LOG_SERIAL.printf("  Active Allocations: %u\n", __memory_tracker__.active_allocations);
    
    // Calculate patterns
    float avg_allocation_size = __memory_tracker__.allocation_count > 0 ? 
        (float)__memory_tracker__.total_allocated / __memory_tracker__.allocation_count : 0.0f;
    float allocation_frequency = __memory_tracker__.allocation_count > 0 ? 
        (float)__memory_tracker__.allocation_count / (__memory_tracker__.total_allocated / 1024.0f) : 0.0f;
    
    LOG_SERIAL.printf("  Average Size:       %.2f bytes per allocation\n", avg_allocation_size);
    LOG_SERIAL.printf("  Allocation Rate:    %.2f allocations per KB\n", allocation_frequency);
    
    LOG_SERIAL.println("\n🔍 MEMORY HEALTH ANALYSIS:");
    uint64_t memory_leaks = __memory_tracker__.total_allocated - __memory_tracker__.total_freed;
    float leak_percentage = __memory_tracker__.total_allocated > 0 ? 
        (memory_leaks * 100.0f / __memory_tracker__.total_allocated) : 0.0f;
    
    // LOG_SERIAL.printf("  Memory Leaks:       %llu bytes (%.2f KB) [%.2f%%]\n", 
    //        (unsigned long long)memory_leaks, memory_leaks / 1024.0f, leak_percentage);
    
    // if (memory_leaks == 0) {
    //     LOG_SERIAL.println("  Status:             ✅ Perfect memory management");
    // } else if (leak_percentage < 1.0f) {
    //     LOG_SERIAL.println("  Status:             ⚠️  Minor memory leaks detected");
    // } else if (leak_percentage < 5.0f) {
    //     LOG_SERIAL.println("  Status:             ⚠️  Moderate memory leaks detected");
    // } else {
    //     LOG_SERIAL.println("  Status:             🚨 Significant memory leaks detected");
    // }
    
    LOG_SERIAL.println("\n🎯 MEMORY EFFICIENCY METRICS:");
    float efficiency = __memory_tracker__.total_allocated > 0 ? 
        (__memory_tracker__.total_freed * 100.0f / __memory_tracker__.total_allocated) : 100.0f;
    float peak_efficiency = __memory_tracker__.peak_allocated > 0 ? 
        (__memory_tracker__.current_allocated * 100.0f / __memory_tracker__.peak_allocated) : 100.0f;
    
    LOG_SERIAL.printf("  Free Efficiency:    %.2f%%\n", efficiency);
    LOG_SERIAL.printf("  Peak Utilization:   %.2f%%\n", peak_efficiency);
    LOG_SERIAL.printf("  Memory Reuse Rate:  %.2f%%\n", 
           __memory_tracker__.allocation_count > 0 ? 
           (__memory_tracker__.free_count * 100.0f / __memory_tracker__.allocation_count) : 0.0f);
    
    LOG_SERIAL.println("\n════════════════════════════════════════════════════════════\n");
}

// Reset memory tracking
void utils_memory_reset() {
    memset(&__memory_tracker__, 0, sizeof(MemoryTracker));
}

// Getter functions
uint64_t utils_get_total_allocated() {
    return __memory_tracker__.total_allocated;
}

uint64_t utils_get_total_freed() {
    return __memory_tracker__.total_freed;
}

uint64_t utils_get_current_allocated() {
    return __memory_tracker__.current_allocated;
}

uint64_t utils_get_peak_allocated() {
    return __memory_tracker__.peak_allocated;
}

uint32_t utils_get_allocation_count() {
    return __memory_tracker__.allocation_count;
}

uint32_t utils_get_free_count() {
    return __memory_tracker__.free_count;
}

uint32_t utils_get_active_allocations() {
    return __memory_tracker__.active_allocations;
}

// ========== Profiler Control Functions ==========

// Print detailed profiler statistics with rich formatting
void profiler_print_stats() {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              PROFILER PERFORMANCE STATISTICS             ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    if (__profiler_entry_count__ == 0) {
        printf("\n📊 No profiler data available.\n");
    } else {
        printf("\n📈 Total Functions Profiled: %zu\n", __profiler_entry_count__);
        printf("Function Name           | Total (us) | Count | Avg (us) | Max (us) | Last (us)\n");
        printf("------------------------|-------------|-------|----------|----------|----------\n");
        
        for (size_t i = 0; i < __profiler_entry_count__; i++) {
            const ProfilerEntry* entry = &__profiler_entries__[i];
            float avg_us = (entry->count > 0) ? (float)entry->total_us / entry->count : 0.0f;
            
            printf("%-23s | %10llu | %5u | %8.2f | %8llu | %8llu\n",
                   entry->name ? entry->name : "NULL",
                   (unsigned long long)entry->total_us,
                   entry->count,
                   avg_us,
                   (unsigned long long)entry->max_us,
                   (unsigned long long)entry->last_us);
        }
    }
    printf("\n════════════════════════════════════════════════════════════\n\n");
}

// Print profiler entries in a compact format
void profiler_print_entries() {
    printf("\n========== Profiler Entries ==========\n");
    if (__profiler_entry_count__ == 0) {
        printf("No profiler data available.\n");
    } else {
        for (size_t i = 0; i < __profiler_entry_count__; i++) {
            const ProfilerEntry* entry = &__profiler_entries__[i];
            printf("%s: %llu us (count: %u, avg: %.2f us, max: %llu us)\n",
                   entry->name ? entry->name : "NULL",
                   (unsigned long long)entry->total_us,
                   entry->count,
                   (entry->count > 0) ? (float)entry->total_us / entry->count : 0.0f,
                   (unsigned long long)entry->max_us);
        }
    }
    printf("=====================================\n\n");
}

// Print profiler summary with rich formatting
void profiler_print_summary() {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║              PROFILER PERFORMANCE SUMMARY                ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    if (__profiler_entry_count__ == 0) {
        printf("\n📊 No profiler data available.\n");
    } else {
        uint64_t total_time = 0;
        uint32_t total_calls = 0;
        uint64_t max_time = 0;
        const char* slowest_function = NULL;
        uint64_t min_time = UINT64_MAX;
        const char* fastest_function = NULL;
        
        for (size_t i = 0; i < __profiler_entry_count__; i++) {
            const ProfilerEntry* entry = &__profiler_entries__[i];
            total_time += entry->total_us;
            total_calls += entry->count;
            
            if (entry->total_us > max_time) {
                max_time = entry->total_us;
                slowest_function = entry->name;
            }
            
            float avg_us = (entry->count > 0) ? (float)entry->total_us / entry->count : 0.0f;
            if (avg_us < min_time && entry->count > 0) {
                min_time = (uint64_t)avg_us;
                fastest_function = entry->name;
            }
        }
        
        printf("\n📈 OVERALL PERFORMANCE METRICS:\n");
        printf("  Total Functions:     %zu\n", __profiler_entry_count__);
        printf("  Total Time:          %llu us (%.2f ms)\n", 
               (unsigned long long)total_time, total_time / 1000.0f);
        printf("  Total Calls:         %u\n", total_calls);
        printf("  Average Call Time:   %.2f us\n", 
               total_calls > 0 ? (float)total_time / total_calls : 0.0f);
        
        printf("\n⚡ PERFORMANCE HIGHLIGHTS:\n");
        printf("  Slowest Function:    %s (%llu us total)\n", 
               slowest_function ? slowest_function : "None",
               (unsigned long long)max_time);
        printf("  Fastest Function:    %s (%.2f us avg)\n", 
               fastest_function ? fastest_function : "None",
               min_time == UINT64_MAX ? 0.0f : (float)min_time);
        
        // Calculate performance distribution
        uint32_t fast_functions = 0;    // < 100us avg
        uint32_t medium_functions = 0;  // 100us - 1ms avg
        uint32_t slow_functions = 0;    // > 1ms avg
        
        for (size_t i = 0; i < __profiler_entry_count__; i++) {
            const ProfilerEntry* entry = &__profiler_entries__[i];
            if (entry->count > 0) {
                float avg_us = (float)entry->total_us / entry->count;
                if (avg_us < 100.0f) {
                    fast_functions++;
                } else if (avg_us < 1000.0f) {
                    medium_functions++;
                } else {
                    slow_functions++;
                }
            }
        }
        
        printf("\n📊 PERFORMANCE DISTRIBUTION:\n");
        printf("  Fast Functions:      %u (%.1f%%) - < 100μs avg\n", 
               fast_functions, (float)fast_functions * 100.0f / __profiler_entry_count__);
        printf("  Medium Functions:    %u (%.1f%%) - 100μs-1ms avg\n", 
               medium_functions, (float)medium_functions * 100.0f / __profiler_entry_count__);
        printf("  Slow Functions:      %u (%.1f%%) - > 1ms avg\n", 
               slow_functions, (float)slow_functions * 100.0f / __profiler_entry_count__);
        
        // Performance recommendations
        printf("\n🎯 PERFORMANCE RECOMMENDATIONS:\n");
        if (slow_functions > 0) {
            printf("  ⚠️  %u functions are slow (>1ms) - consider optimization\n", slow_functions);
        }
        if (fast_functions == __profiler_entry_count__) {
            printf("  ✅ All functions are performing well (<100μs)\n");
        }
        if (total_calls > 1000) {
            printf("  📈 High call frequency detected - consider caching\n");
        }
    }
    printf("\n════════════════════════════════════════════════════════════\n\n");
}

// Get profiler entry count
uint32_t profiler_get_entry_count() {
    return (uint32_t)__profiler_entry_count__;
}

// Get total profiled time
uint64_t profiler_get_total_time() {
    uint64_t total = 0;
    for (size_t i = 0; i < __profiler_entry_count__; i++) {
        total += __profiler_entries__[i].total_us;
    }
    return total;
}

// Get average time for a specific function
float profiler_get_average_time(const char* name) {
    for (size_t i = 0; i < __profiler_entry_count__; i++) {
        const ProfilerEntry* entry = &__profiler_entries__[i];
        if (entry->name && name && strcmp(entry->name, name) == 0) {
            return (entry->count > 0) ? (float)entry->total_us / entry->count : 0.0f;
        }
    }
    return 0.0f;
}

// Get maximum time for a specific function
uint64_t profiler_get_max_time(const char* name) {
    for (size_t i = 0; i < __profiler_entry_count__; i++) {
        const ProfilerEntry* entry = &__profiler_entries__[i];
        if (entry->name && name && strcmp(entry->name, name) == 0) {
            return entry->max_us;
        }
    }
    return 0;
}

// Get call count for a specific function
uint32_t profiler_get_call_count(const char* name) {
    for (size_t i = 0; i < __profiler_entry_count__; i++) {
        const ProfilerEntry* entry = &__profiler_entries__[i];
        if (entry->name && name && strcmp(entry->name, name) == 0) {
            return entry->count;
        }
    }
    return 0;
}

// Print comprehensive performance analysis
void utils_print_performance_analysis() {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║           COMPREHENSIVE PERFORMANCE ANALYSIS             ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    
    // Memory Analysis
    printf("\n💾 MEMORY PERFORMANCE ANALYSIS:\n");
    uint64_t memory_leaks = __memory_tracker__.total_allocated - __memory_tracker__.total_freed;
    float memory_efficiency = __memory_tracker__.total_allocated > 0 ? 
        (__memory_tracker__.total_freed * 100.0f / __memory_tracker__.total_allocated) : 100.0f;
    
    printf("  Memory Efficiency:   %.2f%%\n", memory_efficiency);
    printf("  Peak Usage:          %llu bytes (%.2f KB)\n", 
           (unsigned long long)__memory_tracker__.peak_allocated,
           __memory_tracker__.peak_allocated / 1024.0f);
    printf("  Current Usage:       %llu bytes (%.2f KB)\n",
           (unsigned long long)__memory_tracker__.current_allocated,
           __memory_tracker__.current_allocated / 1024.0f);
    printf("  Memory Leaks:        %llu bytes (%.2f KB)\n",
           (unsigned long long)memory_leaks, memory_leaks / 1024.0f);
    
    // Profiler Analysis
    if (__profiler_entry_count__ > 0) {
        printf("\n⚡ FUNCTION PERFORMANCE ANALYSIS:\n");
        
        uint64_t total_time = 0;
        uint32_t total_calls = 0;
        uint64_t max_time = 0;
        const char* slowest_function = NULL;
        uint32_t slow_functions = 0;
        
        for (size_t i = 0; i < __profiler_entry_count__; i++) {
            const ProfilerEntry* entry = &__profiler_entries__[i];
            total_time += entry->total_us;
            total_calls += entry->count;
            
            if (entry->total_us > max_time) {
                max_time = entry->total_us;
                slowest_function = entry->name;
            }
            
            if (entry->count > 0) {
                float avg_us = (float)entry->total_us / entry->count;
                if (avg_us > 1000.0f) {  // > 1ms
                    slow_functions++;
                }
            }
        }
        
        printf("  Total Functions:     %zu\n", __profiler_entry_count__);
        printf("  Total Execution:     %llu us (%.2f ms)\n", 
               (unsigned long long)total_time, total_time / 1000.0f);
        printf("  Total Calls:         %u\n", total_calls);
        printf("  Average Call Time:   %.2f us\n", 
               total_calls > 0 ? (float)total_time / total_calls : 0.0f);
        printf("  Slow Functions:      %u (%.1f%%)\n", 
               slow_functions, (float)slow_functions * 100.0f / __profiler_entry_count__);
        printf("  Slowest Function:    %s\n", slowest_function ? slowest_function : "None");
    } else {
        printf("\n⚡ FUNCTION PERFORMANCE ANALYSIS:\n");
        printf("  No profiler data available\n");
    }
    
    // Memory allocation frequency analysis
    printf("\n📊 MEMORY ALLOCATION FREQUENCY:\n");
    printf("  Total Allocations:  %u\n", __memory_tracker__.allocation_count);
    printf("  Total Frees:        %u\n", __memory_tracker__.free_count);
    printf("  Active Allocations: %u\n", __memory_tracker__.active_allocations);
    printf("  Allocation Rate:    %.2f allocations per second\n", 
           __memory_tracker__.allocation_count > 0 ? 
           (float)__memory_tracker__.allocation_count / (__memory_tracker__.total_allocated / 1024.0f) : 0.0f);
    
    printf("\n════════════════════════════════════════════════════════════\n\n");
}
