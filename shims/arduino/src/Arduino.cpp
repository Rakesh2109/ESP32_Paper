// Arduino.cpp - Implementation of Arduino shim for native compilation
#include "Arduino.h"
#include <stdarg.h>
#include <stdint.h>
#include <chrono>
#include <thread>

// Startup time reference for millis() and micros()
static auto g_start_time = std::chrono::steady_clock::now();

// Timing functions implementation
unsigned long millis(void) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_start_time);
    return static_cast<unsigned long>(elapsed.count());
}

unsigned long micros(void) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - g_start_time);
    return static_cast<unsigned long>(elapsed.count());
}

void delay(unsigned long ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void delayMicroseconds(unsigned int us) {
    std::this_thread::sleep_for(std::chrono::microseconds(us));
}

// Global Serial object (ESP is in EspClass.cpp for native builds)
SerialStub Serial;

