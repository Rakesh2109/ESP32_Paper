#pragma once
// esp_timer.h shim for native compilation

#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get time in microseconds since boot (stub implementation)
static inline int64_t esp_timer_get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
}

#ifdef __cplusplus
}
#endif

