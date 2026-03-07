#pragma once
// freertos/queue.h shim for native compilation with REAL queue support

#include "FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

// Real implementations (defined in freertos_impl.cpp)
QueueHandle_t xQueueCreate(UBaseType_t uxQueueLength, UBaseType_t uxItemSize);
void vQueueDelete(QueueHandle_t xQueue);
BaseType_t xQueueSend(QueueHandle_t xQueue, const void* pvItemToQueue, TickType_t xTicksToWait);
BaseType_t xQueueReceive(QueueHandle_t xQueue, void* pvBuffer, TickType_t xTicksToWait);
UBaseType_t uxQueueMessagesWaiting(QueueHandle_t xQueue);
UBaseType_t uxQueueSpacesAvailable(QueueHandle_t xQueue);

#ifdef __cplusplus
}
#endif

