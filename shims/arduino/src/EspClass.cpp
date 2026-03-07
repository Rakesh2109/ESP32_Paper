// EspClass implementation with REAL heap tracking for native builds
#ifdef NATIVE_BUILD

#include "Arduino.h"
#include <cstdlib>
#include <algorithm>

// Platform-specific includes for memory stats
#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
    #include <sys/sysinfo.h>
    #include <unistd.h>
    #include <fstream>
    #include <sstream>
#elif defined(__APPLE__)
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/task.h>
#endif

// External memory tracker from utils.cpp (C++ linkage)
// These functions are defined in src/utils.cpp
extern uint64_t utils_get_current_allocated();
extern uint64_t utils_get_peak_allocated();

// ============================================================================
// Platform-Specific Memory Queries
// ============================================================================

struct SystemMemoryInfo {
    size_t total_physical;      // Total system RAM
    size_t available_physical;  // Available system RAM
    size_t process_used;        // Memory used by this process
    size_t process_peak;        // Peak memory used by this process
};

static SystemMemoryInfo get_system_memory() {
    SystemMemoryInfo info = {0};
    
#if defined(_WIN32) || defined(_WIN64)
    // Windows: Use GlobalMemoryStatusEx for system, GetProcessMemoryInfo for process
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    if (GlobalMemoryStatusEx(&mem_status)) {
        info.total_physical = mem_status.ullTotalPhys;
        info.available_physical = mem_status.ullAvailPhys;
    }
    
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        info.process_used = pmc.WorkingSetSize;
        info.process_peak = pmc.PeakWorkingSetSize;
    }
    
#elif defined(__linux__)
    // Linux: Parse /proc/meminfo and /proc/self/status
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        std::istringstream iss(line);
        std::string key;
        size_t value;
        std::string unit;
        
        if (iss >> key >> value >> unit) {
            if (key == "MemTotal:") {
                info.total_physical = value * 1024; // Convert KB to bytes
            } else if (key == "MemAvailable:") {
                info.available_physical = value * 1024;
            }
        }
    }
    
    // Parse /proc/self/status for process memory
    std::ifstream status("/proc/self/status");
    while (std::getline(status, line)) {
        std::istringstream iss(line);
        std::string key;
        size_t value;
        std::string unit;
        
        if (iss >> key >> value >> unit) {
            if (key == "VmRSS:") {
                info.process_used = value * 1024; // Convert KB to bytes
            } else if (key == "VmHWM:") {
                info.process_peak = value * 1024;
            }
        }
    }
    
#elif defined(__APPLE__)
    // macOS: Use sysctl for system memory, task_info for process
    int mib[2];
    size_t length;
    
    // Total physical memory
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(info.total_physical);
    sysctl(mib, 2, &info.total_physical, &length, NULL, 0);
    
    // Process memory
    struct mach_task_basic_info task_info_data;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, 
                  (task_info_t)&task_info_data, &count) == KERN_SUCCESS) {
        info.process_used = task_info_data.resident_size;
        info.process_peak = task_info_data.resident_size_max;
    }
    
    // Available memory (approximate - macOS doesn't provide this easily)
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t vm_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, 
                          (host_info64_t)&vm_stats, &vm_count) == KERN_SUCCESS) {
        info.available_physical = (vm_stats.free_count + vm_stats.inactive_count) * vm_page_size;
    }
#endif
    
    return info;
}

// ============================================================================
// EspClass Implementation
// ============================================================================

// Static variables to track min heap (mimics ESP32 behavior)
static uint32_t s_min_free_heap = UINT32_MAX;
static uint32_t s_initial_heap = 0;

uint32_t EspClass::getFreeHeap() {
    SystemMemoryInfo sys_mem = get_system_memory();
    
    // Simulate ESP32 behavior: "free heap" = available memory for this process
    // We approximate this as: available system memory - what we've allocated
    uint64_t our_allocated = utils_get_current_allocated();
    
    // Cap at 4GB (uint32_t max) to match ESP32 API
    uint64_t free_heap = sys_mem.available_physical;
    if (free_heap > UINT32_MAX) {
        free_heap = UINT32_MAX;
    }
    
    uint32_t current_free = (uint32_t)free_heap;
    
    // Track minimum free heap (ESP32 does this automatically)
    if (current_free < s_min_free_heap) {
        s_min_free_heap = current_free;
    }
    
    return current_free;
}

uint32_t EspClass::getHeapSize() {
    SystemMemoryInfo sys_mem = get_system_memory();
    
    // Return total system RAM (capped at 4GB for API compatibility)
    if (sys_mem.total_physical > UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)sys_mem.total_physical;
}

uint32_t EspClass::getMinFreeHeap() {
    // Return the minimum free heap observed (like ESP32)
    if (s_min_free_heap == UINT32_MAX) {
        return getFreeHeap(); // First call
    }
    return s_min_free_heap;
}

uint32_t EspClass::getMaxAllocHeap() {
    SystemMemoryInfo sys_mem = get_system_memory();
    
    // Return largest contiguous block available
    // On desktop, this is typically close to available memory
    // since we have virtual memory and memory fragmentation is handled by OS
    uint64_t max_alloc = sys_mem.available_physical;
    
    if (max_alloc > UINT32_MAX) {
        return UINT32_MAX;
    }
    return (uint32_t)max_alloc;
}

uint32_t EspClass::getPsramSize() {
    // Desktop has no dedicated PSRAM; map to heap size for compatibility.
    return getHeapSize();
}

uint32_t EspClass::getFreePsram() {
    // Desktop has no dedicated PSRAM; map to free heap for compatibility.
    return getFreeHeap();
}

uint8_t EspClass::getChipRevision() {
    // Return a fake chip revision (not applicable on desktop)
    return 0;
}

const char* EspClass::getSdkVersion() {
    // Return platform identifier
#if defined(_WIN32) || defined(_WIN64)
    return "native-windows";
#elif defined(__linux__)
    return "native-linux";
#elif defined(__APPLE__)
    return "native-macos";
#else
    return "native-unknown";
#endif
}

uint32_t EspClass::getCpuFreqMHz() {
    // Return a reasonable CPU frequency (not critical for desktop)
    // Could query actual CPU freq, but 240 matches ESP32-S3
    return 240;
}

uint32_t EspClass::getCycleCount() {
    // Not meaningful on desktop (different CPU architecture)
    return 0;
}

void EspClass::restart() {
    // Exit the program (equivalent to restart)
    exit(0);
}

// Global ESP instance
EspClass ESP;

#endif // NATIVE_BUILD

