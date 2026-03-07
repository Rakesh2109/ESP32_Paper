// Desktop HAL - Main entry point for native builds
#ifdef NATIVE_BUILD

#include <iostream>
#include <signal.h>
#include "Arduino.h"

// Forward declarations from main.cpp
extern void setup();
extern void loop();

// HAL transport initialization
extern bool hal_init_transport();

// Global flag for graceful shutdown
static volatile bool g_keep_running = true;

// Signal handler for Ctrl+C
void signal_handler(int signum) {
    (void)signum;
    std::cout << "\n🛑 Interrupt received, shutting down gracefully..." << std::endl;
    g_keep_running = false;
}

// Arduino-style main function for native builds
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    // Setup signal handler for Ctrl+C
    signal(SIGINT, signal_handler);
    
    std::cout << "🚀 Starting ESP32_TM in native mode..." << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    
    // Initialize HAL transport layer
    if (!hal_init_transport()) {
        std::cout << "❌ Failed to initialize transport" << std::endl;
        return 1;
    }
    
    // Call Arduino setup() once
    std::cout << "⚙️  Calling setup()..." << std::endl;
    setup();
    
    std::cout << "✅ Setup complete, entering loop..." << std::endl;
    std::cout << "   Press Ctrl+C to stop" << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" << std::endl;
    
    // Call Arduino loop() repeatedly
    while (g_keep_running) {
        loop();
        
        // Small delay to prevent CPU spinning (mimics vTaskDelay)
        delay(10);
    }
    
    std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
    std::cout << "👋 ESP32_TM terminated cleanly" << std::endl;
    
    return 0;
}

#endif // NATIVE_BUILD

