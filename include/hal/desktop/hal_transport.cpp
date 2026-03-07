// Desktop HAL - Transport instance and initialization
#ifdef NATIVE_BUILD

#include "core/transport.h"

#ifdef HAL_SOCKET_TRANSPORT
    // Socket transport mode - use SocketStream backend
    #include "hal/desktop/socket_stream.h"
    #include <cstdlib>
    #include <cstdio>
    static SocketStream g_socket_stream;
    Transport TR(g_socket_stream);
    
    // Initialize socket transport
    bool hal_init_transport() {
        printf("[HAL] Initializing socket transport...\n");
        uint16_t base_port = 5555;
        if (const char* env = std::getenv("HAL_SOCKET_PORT")) {
            int parsed = std::atoi(env);
            if (parsed > 0 && parsed < 65535) {
                base_port = static_cast<uint16_t>(parsed);
            }
        }

        for (uint16_t offset = 0; offset < 5; ++offset) {
            uint16_t port = base_port + offset;
            if (g_socket_stream.listen(port)) {
                printf("[HAL] Socket transport ready on port %u\n", port);
                printf("      Run: python test_serial.py --port socket://localhost:%u\n\n", port);
                return true;
            }
            printf("[HAL] Failed to listen on port %u, trying next...\n", port);
        }
        printf("[HAL] Unable to open any socket port\n");
        return false;
    }
    
#else
    // Stub transport mode - use ACTIVE_SERIAL (stub)
    Transport TR(Serial);
    
    bool hal_init_transport() {
        printf("[HAL] Stub transport mode (no communication)\n\n");
        return true;
    }
    
#endif

#endif // NATIVE_BUILD

