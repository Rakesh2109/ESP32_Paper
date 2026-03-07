#pragma once
// Socket Stream - Arduino Stream interface over TCP sockets

#ifdef NATIVE_BUILD

#include "Arduino.h"  // Get Stream base class

/**
 * SocketStream - Arduino Stream interface backed by TCP socket
 * 
 * Implements the Arduino Stream interface so it can be used with
 * the existing Transport class without any modifications.
 * 
 * Usage:
 *   SocketStream sock;
 *   sock.listen(5555);  // Listen on port 5555
 *   Transport tr(sock); // Use with existing Transport class
 *   
 * Compatible with test_serial.py:
 *   python test_serial.py --port socket://localhost:5555
 */
class SocketStream : public Stream {
public:
    SocketStream();
    ~SocketStream();
    
    // Setup methods
    bool listen(uint16_t port);          // Listen on port (server mode)
    bool connect(const char* host, uint16_t port); // Connect to host (client mode)
    void close();
    bool isConnected() const;
    
    // Arduino Stream interface - read methods
    int available() override;
    int read() override;
    int peek() override;
    
    // Arduino Print interface - write methods  
    size_t write(uint8_t byte) override;
    size_t write(const uint8_t *buffer, size_t size) override;
    void flush();  // Not override - Print doesn't have flush()
    
    // Arduino Stream timing
    void setTimeout(unsigned long timeout);
    
private:
    enum Mode {
        MODE_NONE,
        MODE_SERVER,
        MODE_CLIENT
    };
    
    // Platform-specific socket types
    #if defined(_WIN32) || defined(_WIN64)
        typedef uintptr_t socket_t;  // Windows uses SOCKET (64-bit)
        static constexpr socket_t INVALID_SOCKET_FD = (socket_t)~0;
    #else
        typedef int socket_t;        // Unix uses int file descriptor
        static constexpr socket_t INVALID_SOCKET_FD = -1;
    #endif
    
    Mode mode_;
    socket_t socket_fd_;      // Listening socket (server) or connection socket (client)
    socket_t client_fd_;      // Client connection (server mode only)
    uint16_t port_;
    unsigned long timeout_ms_;
    
    bool acceptClient();  // Accept client connection (server mode)
    bool waitForData(unsigned long timeout_ms);
};

#endif // NATIVE_BUILD

