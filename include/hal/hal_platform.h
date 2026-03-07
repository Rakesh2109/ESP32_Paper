#pragma once
// Hardware Abstraction Layer - Platform Selector
//
// Automatically selects the correct implementation based on build target:
// - ESP32 builds: Use real hardware implementations
// - Desktop builds (Windows/Linux/macOS): Use stubs for testing
//
// Usage: #include "hal/hal_platform.h"

// ============================================================================
// Platform Detection and Selection
// ============================================================================

#ifdef NATIVE_BUILD
    // ========================================================================
    // DESKTOP PLATFORMS (Windows, Linux, macOS)
    // ========================================================================
    
    // Desktop builds use the SAME Transport class as ESP32
    // But with different Stream backends:
    // - HAL_SOCKET_TRANSPORT: SocketStream (TCP socket, for test_serial.py)
    // - Default: StubStream (no communication, for unit tests)
    
    #ifdef HAL_SOCKET_TRANSPORT
        #include "hal/desktop/socket_stream.h"
    // #else
        // StubStream not needed - transport_desktop.h provides stub Transport
    #endif
    
    // Include real Transport class (works with any Stream backend)
    #include "core/transport.h"
    
    // Include platform-specific headers for OS-specific utilities
    #if defined(_WIN32) || defined(_WIN64)
        #include "hal/desktop/platform_windows.h"
    #elif defined(__linux__)
        #include "hal/desktop/platform_linux.h"
    #elif defined(__APPLE__)
        #include "hal/desktop/platform_macos.h"
    #endif
    
#else
    // ========================================================================
    // ESP32 PLATFORM (all variants)
    // ========================================================================
    
    // Use real Transport implementation from src/
    #include "core/transport.h"
    
#endif // NATIVE_BUILD

// ============================================================================
// Platform-Agnostic Utilities
// ============================================================================

// Protocol utils works with both real and stub Transport
#include "core/protocol_utils.h"

