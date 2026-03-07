#pragma once
// freertos/task.h shim for native compilation with REAL thread support

#include "FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*TaskFunction_t)(void*);

#define configMINIMAL_STACK_SIZE 128

// Real implementations (defined in freertos_impl.cpp)
BaseType_t xTaskCreate(
    TaskFunction_t pvTaskCode,
    const char* pcName,
    uint32_t usStackDepth,
    void* pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t* pvCreatedTask);

BaseType_t xTaskCreatePinnedToCore(
    TaskFunction_t pvTaskCode,
    const char* pcName,
    uint32_t usStackDepth,
    void* pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t* pvCreatedTask,
    BaseType_t xCoreID);

void vTaskDelete(TaskHandle_t xTask);

#ifdef __cplusplus
}
#endif

