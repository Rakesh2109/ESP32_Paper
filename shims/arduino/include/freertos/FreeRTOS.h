#pragma once
// FreeRTOS.h shim for native compilation

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal FreeRTOS types and functions for native compilation
typedef void* TaskHandle_t;
typedef void* QueueHandle_t;
typedef uint32_t TickType_t;
typedef uint32_t UBaseType_t;
typedef int32_t BaseType_t;

#define portMAX_DELAY 0xFFFFFFFFUL
#define portTICK_PERIOD_MS 1
#define pdMS_TO_TICKS(ms) (ms)
#define pdTRUE  1
#define pdFALSE 0
#define pdPASS  1
#define pdFAIL  0

// Real implementations (defined in freertos_impl.cpp)
TickType_t xTaskGetTickCount(void);
void vTaskDelay(TickType_t xTicksToDelay);

// Yield to other threads (hint to OS scheduler)
static inline void taskYIELD(void) { 
    vTaskDelay(0);  // Sleep for 0ms = yield to other threads
}

#ifdef __cplusplus
}
#endif

