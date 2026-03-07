// Socket Stream Implementation
#ifdef NATIVE_BUILD

#include "socket_stream.h"
#include <stdio.h>
#include <string.h>

// Platform-specific socket headers
#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
    #define SOCKET_ERROR_CHECK(x) ((x) == SOCKET_ERROR)
    #define CLOSE_SOCKET(fd) closesocket(fd)
    #define MSG_NOSIGNAL 0
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    #include <netdb.h>
    #define SOCKET_ERROR_CHECK(x) ((x) < 0)
    #define CLOSE_SOCKET(fd) ::close(fd)
    #define INVALID_SOCKET_FD -1
    #define SOCKET_ERROR -1
#endif

#include <chrono>
#include <thread>

SocketStream::SocketStream() 
    : mode_(MODE_NONE)
    , socket_fd_(INVALID_SOCKET_FD)
    , client_fd_(INVALID_SOCKET_FD)
    , port_(0)
    , timeout_ms_(1000)  // Default 1 second timeout
{
#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

SocketStream::~SocketStream() {
    close();
    
#ifdef _WIN32
    WSACleanup();
#endif
}

bool SocketStream::listen(uint16_t port) {
    close(); // Close any existing connection
    
    mode_ = MODE_SERVER;
    port_ = port;
    
    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ == INVALID_SOCKET_FD) {
        printf("[SocketStream] Failed to create socket\n");
        return false;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));
    
    // Set non-blocking mode
    #ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(socket_fd_, FIONBIO, &mode);
    #else
        int flags = fcntl(socket_fd_, F_GETFL, 0);
        fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK);
    #endif
    
    // Bind to port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    
    if (SOCKET_ERROR_CHECK(bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)))) {
        printf("[SocketStream] Failed to bind to port %d\n", port);
        CLOSE_SOCKET(socket_fd_);
        socket_fd_ = INVALID_SOCKET_FD;
        return false;
    }
    
    // Listen
    if (SOCKET_ERROR_CHECK(::listen(socket_fd_, 1))) {
        printf("[SocketStream] Failed to listen on port %d\n", port);
        CLOSE_SOCKET(socket_fd_);
        socket_fd_ = INVALID_SOCKET_FD;
        return false;
    }
    
    printf("[SocketStream] Listening on port %d\n", port);
    printf("               Connect with: python test_serial.py --port socket://localhost:%d\n", port);
    
    return true;
}

bool SocketStream::connect(const char* host, uint16_t port) {
    close(); // Close any existing connection
    
    mode_ = MODE_CLIENT;
    port_ = port;
    
    // Create socket
    socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd_ == INVALID_SOCKET_FD) {
        printf("[SocketStream] Failed to create socket\n");
        return false;
    }
    
    // Resolve hostname
    struct hostent* server = gethostbyname(host);
    if (server == NULL) {
        printf("[SocketStream] Failed to resolve host: %s\n", host);
        CLOSE_SOCKET(socket_fd_);
        socket_fd_ = INVALID_SOCKET_FD;
        return false;
    }
    
    // Connect
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    memcpy(&addr.sin_addr.s_addr, server->h_addr, server->h_length);
    addr.sin_port = htons(port);
    
    if (SOCKET_ERROR_CHECK(::connect(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)))) {
        printf("[SocketStream] Failed to connect to %s:%d\n", host, port);
        CLOSE_SOCKET(socket_fd_);
        socket_fd_ = INVALID_SOCKET_FD;
        return false;
    }
    
    // Set TCP_NODELAY for low latency
    int flag = 1;
    setsockopt(socket_fd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
    
    printf("[SocketStream] Connected to %s:%d\n", host, port);
    
    return true;
}

void SocketStream::close() {
    if (client_fd_ != INVALID_SOCKET_FD) {
        CLOSE_SOCKET(client_fd_);
        client_fd_ = INVALID_SOCKET_FD;
    }
    if (socket_fd_ != INVALID_SOCKET_FD) {
        CLOSE_SOCKET(socket_fd_);
        socket_fd_ = INVALID_SOCKET_FD;
    }
    mode_ = MODE_NONE;
}

bool SocketStream::isConnected() const {
    if (mode_ == MODE_SERVER) {
        return client_fd_ != INVALID_SOCKET_FD;
    } else if (mode_ == MODE_CLIENT) {
        return socket_fd_ != INVALID_SOCKET_FD;
    }
    return false;
}

bool SocketStream::acceptClient() {
    if (mode_ != MODE_SERVER || socket_fd_ == INVALID_SOCKET_FD) {
        return false;
    }
    
    if (client_fd_ != INVALID_SOCKET_FD) {
        return true; // Already connected
    }
    
    // Try to accept (non-blocking)
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    client_fd_ = accept(socket_fd_, (struct sockaddr*)&client_addr, &client_len);
    
    if (client_fd_ == INVALID_SOCKET_FD) {
        // No client available (expected with non-blocking socket)
        return false;
    }
    
    // Set TCP_NODELAY for low latency
    int flag = 1;
    setsockopt(client_fd_, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
    
    printf("[SocketStream] Client connected from %s:%d\n", 
           inet_ntoa(client_addr.sin_addr), 
           ntohs(client_addr.sin_port));
    
    return true;
}

bool SocketStream::waitForData(unsigned long timeout_ms) {
    int fd = (mode_ == MODE_SERVER) ? client_fd_ : socket_fd_;
    if (fd == INVALID_SOCKET_FD) {
        return false;
    }
    
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);
    
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    
    int result = select(fd + 1, &readfds, NULL, NULL, &tv);
    return result > 0;
}

// Arduino Stream interface implementation
int SocketStream::available() {
    // Ensure we're connected
    if (mode_ == MODE_SERVER && client_fd_ == INVALID_SOCKET_FD) {
        acceptClient();
    }
    
    if (!isConnected()) {
        return 0;
    }
    
    // Use select with zero timeout to check if data is available
    if (waitForData(0)) {
        // Data is available - use recv with MSG_PEEK to check how much
        int fd = (mode_ == MODE_SERVER) ? client_fd_ : socket_fd_;
        char buf[1];
        int result = recv(fd, buf, 1, MSG_PEEK);
        return (result > 0) ? 1 : 0;  // Return 1 if data available, 0 otherwise
    }
    
    return 0;
}

int SocketStream::read() {
    // Ensure we're connected
    if (mode_ == MODE_SERVER && client_fd_ == INVALID_SOCKET_FD) {
        if (!acceptClient()) {
            return -1;
        }
    }
    
    if (!isConnected()) {
        return -1;
    }
    
    int fd = (mode_ == MODE_SERVER) ? client_fd_ : socket_fd_;
    uint8_t byte;
    int result = recv(fd, (char*)&byte, 1, 0);
    
    if (result == 1) {
        return byte;
    } else if (result == 0) {
        // Connection closed
        printf("[SocketStream] Connection closed\n");
        if (mode_ == MODE_SERVER) {
            CLOSE_SOCKET(client_fd_);
            client_fd_ = INVALID_SOCKET_FD;
        } else {
            CLOSE_SOCKET(socket_fd_);
            socket_fd_ = INVALID_SOCKET_FD;
        }
    }
    
    return -1;
}

int SocketStream::peek() {
    if (!isConnected()) {
        return -1;
    }
    
    int fd = (mode_ == MODE_SERVER) ? client_fd_ : socket_fd_;
    uint8_t byte;
    int result = recv(fd, (char*)&byte, 1, MSG_PEEK);
    
    return (result == 1) ? byte : -1;
}

size_t SocketStream::write(uint8_t byte) {
    return write(&byte, 1);
}

size_t SocketStream::write(const uint8_t *buffer, size_t size) {
    // Ensure we're connected
    if (mode_ == MODE_SERVER && client_fd_ == INVALID_SOCKET_FD) {
        if (!acceptClient()) {
            return 0;
        }
    }
    
    if (!isConnected() || !buffer || size == 0) {
        return 0;
    }
    
    int fd = (mode_ == MODE_SERVER) ? client_fd_ : socket_fd_;
    int result = send(fd, (const char*)buffer, size, MSG_NOSIGNAL);
    
    if (result < 0) {
        // Send failed - connection may be closed
        if (mode_ == MODE_SERVER) {
            CLOSE_SOCKET(client_fd_);
            client_fd_ = INVALID_SOCKET_FD;
        } else {
            CLOSE_SOCKET(socket_fd_);
            socket_fd_ = INVALID_SOCKET_FD;
        }
        return 0;
    }
    
    return result;
}

void SocketStream::flush() {
    // TCP sockets don't have a flush operation like serial
    // Data is sent immediately
}

void SocketStream::setTimeout(unsigned long timeout) {
    timeout_ms_ = timeout;
}

#endif // NATIVE_BUILD

