// FreeRTOS implementation for native builds using C++ threads
#ifdef NATIVE_BUILD

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>
#include <chrono>
#include <iostream>

// ============================================================================
// Task Management
// ============================================================================

struct TaskData {
    std::thread thread;
    TaskFunction_t function;
    void* parameter;
    const char* name;
    bool running;
    
    TaskData() : function(nullptr), parameter(nullptr), name(nullptr), running(false) {}
};

static std::map<TaskHandle_t, TaskData*> g_tasks;
static std::mutex g_task_mutex;
static std::atomic<uint64_t> g_task_id_counter{1};

// Task wrapper that runs the user's task function
static void task_wrapper(TaskData* task_data) {
    if (task_data && task_data->function) {
        std::cout << "🧵 Task '" << (task_data->name ? task_data->name : "unnamed") 
                  << "' started" << std::endl;
        task_data->running = true;
        task_data->function(task_data->parameter);
        task_data->running = false;
        std::cout << "🧵 Task '" << (task_data->name ? task_data->name : "unnamed") 
                  << "' finished" << std::endl;
    }
}

extern "C" {

BaseType_t xTaskCreate(
    TaskFunction_t pvTaskCode,
    const char* pcName,
    uint32_t usStackDepth,
    void* pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t* pvCreatedTask)
{
    return xTaskCreatePinnedToCore(pvTaskCode, pcName, usStackDepth, 
                                    pvParameters, uxPriority, pvCreatedTask, 0);
}

BaseType_t xTaskCreatePinnedToCore(
    TaskFunction_t pvTaskCode,
    const char* pcName,
    uint32_t usStackDepth,
    void* pvParameters,
    UBaseType_t uxPriority,
    TaskHandle_t* pvCreatedTask,
    BaseType_t xCoreID)
{
    (void)usStackDepth;  // Stack size not used in native
    (void)uxPriority;    // Priority not used in native
    (void)xCoreID;       // Core affinity not used in native
    
    if (!pvTaskCode) return pdFAIL;
    
    std::lock_guard<std::mutex> lock(g_task_mutex);
    
    // Create task data
    TaskData* task_data = new TaskData();
    task_data->function = pvTaskCode;
    task_data->parameter = pvParameters;
    task_data->name = pcName;
    
    // Generate unique task handle
    TaskHandle_t handle = (TaskHandle_t)(g_task_id_counter.fetch_add(1));
    
    // Start thread
    try {
        task_data->thread = std::thread(task_wrapper, task_data);
        g_tasks[handle] = task_data;
        
        if (pvCreatedTask) {
            *pvCreatedTask = handle;
        }
        
        return pdPASS;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to create task '" << pcName << "': " << e.what() << std::endl;
        delete task_data;
        return pdFAIL;
    }
}

void vTaskDelete(TaskHandle_t xTask) {
    std::lock_guard<std::mutex> lock(g_task_mutex);
    
    auto it = g_tasks.find(xTask);
    if (it != g_tasks.end()) {
        TaskData* task_data = it->second;
        if (task_data->thread.joinable()) {
            task_data->thread.detach();  // Let it finish naturally
        }
        delete task_data;
        g_tasks.erase(it);
    }
}

void vTaskDelay(TickType_t xTicksToDelay) {
    std::this_thread::sleep_for(std::chrono::milliseconds(xTicksToDelay));
}

TickType_t xTaskGetTickCount(void) {
    return (TickType_t)(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

// ============================================================================
// Queue Management
// ============================================================================

struct QueueData {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<uint8_t*> items;
    size_t item_size;
    size_t max_items;
    
    QueueData(size_t item_sz, size_t max_cnt) 
        : item_size(item_sz), max_items(max_cnt) {}
    
    ~QueueData() {
        // Clean up any remaining items
        while (!items.empty()) {
            delete[] items.front();
            items.pop();
        }
    }
};

QueueHandle_t xQueueCreate(UBaseType_t uxQueueLength, UBaseType_t uxItemSize) {
    if (uxQueueLength == 0 || uxItemSize == 0) return nullptr;
    
    try {
        QueueData* queue = new QueueData(uxItemSize, uxQueueLength);
        return (QueueHandle_t)queue;
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to create queue: " << e.what() << std::endl;
        return nullptr;
    }
}

void vQueueDelete(QueueHandle_t xQueue) {
    if (xQueue) {
        delete (QueueData*)xQueue;
    }
}

BaseType_t xQueueSend(QueueHandle_t xQueue, const void* pvItemToQueue, TickType_t xTicksToWait) {
    if (!xQueue || !pvItemToQueue) return pdFAIL;
    
    QueueData* queue = (QueueData*)xQueue;
    std::unique_lock<std::mutex> lock(queue->mutex);
    
    // Wait if queue is full
    if (xTicksToWait == portMAX_DELAY) {
        // Wait indefinitely
        queue->cv.wait(lock, [queue] { return queue->items.size() < queue->max_items; });
    } else if (xTicksToWait > 0) {
        // Wait with timeout
        if (!queue->cv.wait_for(lock, std::chrono::milliseconds(xTicksToWait),
                                [queue] { return queue->items.size() < queue->max_items; })) {
            return pdFAIL;  // Timeout
        }
    } else {
        // No wait
        if (queue->items.size() >= queue->max_items) {
            return pdFAIL;
        }
    }
    
    // Copy item data
    uint8_t* item = new uint8_t[queue->item_size];
    memcpy(item, pvItemToQueue, queue->item_size);
    queue->items.push(item);
    
    // Notify waiting receivers
    queue->cv.notify_one();
    
    return pdPASS;
}

BaseType_t xQueueReceive(QueueHandle_t xQueue, void* pvBuffer, TickType_t xTicksToWait) {
    if (!xQueue || !pvBuffer) return pdFAIL;
    
    QueueData* queue = (QueueData*)xQueue;
    std::unique_lock<std::mutex> lock(queue->mutex);
    
    // Wait if queue is empty
    if (xTicksToWait == portMAX_DELAY) {
        // Wait indefinitely
        queue->cv.wait(lock, [queue] { return !queue->items.empty(); });
    } else if (xTicksToWait > 0) {
        // Wait with timeout
        if (!queue->cv.wait_for(lock, std::chrono::milliseconds(xTicksToWait),
                                [queue] { return !queue->items.empty(); })) {
            return pdFAIL;  // Timeout
        }
    } else {
        // No wait
        if (queue->items.empty()) {
            return pdFAIL;
        }
    }
    
    // Copy item data
    uint8_t* item = queue->items.front();
    queue->items.pop();
    memcpy(pvBuffer, item, queue->item_size);
    delete[] item;
    
    // Notify waiting senders
    queue->cv.notify_one();
    
    return pdPASS;
}

UBaseType_t uxQueueMessagesWaiting(QueueHandle_t xQueue) {
    if (!xQueue) return 0;
    
    QueueData* queue = (QueueData*)xQueue;
    std::lock_guard<std::mutex> lock(queue->mutex);
    return (UBaseType_t)queue->items.size();
}

UBaseType_t uxQueueSpacesAvailable(QueueHandle_t xQueue) {
    if (!xQueue) return 0;
    
    QueueData* queue = (QueueData*)xQueue;
    std::lock_guard<std::mutex> lock(queue->mutex);
    return (UBaseType_t)(queue->max_items - queue->items.size());
}

} // extern "C"

#endif // NATIVE_BUILD

