#pragma once
// Windows-specific platform utilities

#if defined(_WIN32) || defined(_WIN64)

// Windows-specific includes
#include <windows.h>

// Platform detection
#define HAL_PLATFORM_WINDOWS 1
#define HAL_PLATFORM_NAME "Windows"

// Windows-specific utilities can be added here as needed
// Examples:
// - High-precision timers
// - Memory allocation tracking
// - Console color output
// - File I/O optimizations

#endif // _WIN32 || _WIN64

