#pragma once
// Linux-specific platform utilities

#ifdef __linux__

// Linux-specific includes
#include <unistd.h>
#include <sys/time.h>

// Platform detection
#define HAL_PLATFORM_LINUX 1
#define HAL_PLATFORM_NAME "Linux"

// Linux-specific utilities can be added here as needed
// Examples:
// - High-precision timers (clock_gettime)
// - Memory profiling hooks
// - ANSI color output
// - File I/O optimizations

#endif // __linux__

