#pragma once
// esp_task_wdt.h shim for native compilation

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Task watchdog timer stubs
typedef void* esp_task_wdt_user_handle_t;

static inline int esp_task_wdt_init(uint32_t timeout, bool panic_on_timeout) {
    (void)timeout; (void)panic_on_timeout;
    return 0;
}

static inline int esp_task_wdt_deinit(void) {
    return 0;
}

static inline int esp_task_wdt_add(esp_task_wdt_user_handle_t handle) {
    (void)handle;
    return 0;
}

static inline int esp_task_wdt_delete(esp_task_wdt_user_handle_t handle) {
    (void)handle;
    return 0;
}

static inline int esp_task_wdt_reset(void) {
    return 0;
}

#ifdef __cplusplus
}
#endif

