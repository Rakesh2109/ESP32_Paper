#pragma once
// macOS-specific platform utilities

#ifdef __APPLE__

// macOS-specific includes
#include <unistd.h>
#include <sys/time.h>
#include <mach/mach_time.h>

// Platform detection
#define HAL_PLATFORM_MACOS 1
#define HAL_PLATFORM_NAME "macOS"

// Detect Apple Silicon vs Intel
#if defined(__arm64__) || defined(__aarch64__)
    #define HAL_PLATFORM_APPLE_SILICON 1
#else
    #define HAL_PLATFORM_INTEL 1
#endif

// macOS-specific utilities can be added here as needed
// Examples:
// - Mach absolute time for high-precision timing
// - Memory profiling with Instruments
// - ANSI color output
// - File I/O optimizations

#endif // __APPLE__

