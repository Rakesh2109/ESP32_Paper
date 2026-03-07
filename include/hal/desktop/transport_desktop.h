#pragma once
// Desktop Transport HAL - stub implementation for Windows, Linux, and macOS

#ifdef NATIVE_BUILD

#include <stdint.h>
#include <cstddef>
#include <stdio.h>
#include "core/protocol.h"  // Get FrameHeader and MetricPayload definitions

// Forward declaration
class Stream;

/**
 * Transport stub for desktop platforms (Windows, Linux, macOS)
 * Provides same API as ESP32 Transport but with no-op implementations
 * Allows compilation and basic testing without hardware
 */
class Transport {
public:
    Transport(Stream& ser) { (void)ser; }
    
    // Stub implementations
    // sendLog prints to stdout for native debugging
    void sendLog(const char* msg, uint16_t len) {
        if (msg && len > 0) {
            fwrite(msg, 1, len, stdout);
            fflush(stdout);
        }
    }
    
    void sendReady(uint8_t major, uint8_t minor, uint32_t trained, uint32_t tested) { 
        (void)major; (void)minor; (void)trained; (void)tested; 
    }
    
    void sendStat(uint32_t seq, uint32_t trained, uint32_t tested, float acc) { 
        (void)seq; (void)trained; (void)tested; (void)acc; 
    }
    
    void sendMetric(uint32_t seq, const MetricPayload& mp) { 
        (void)seq; (void)mp; 
    }
    
    void sendAck(uint32_t count, uint32_t seq) { 
        (void)count; (void)seq; 
    }
    
    void sendDone(uint32_t count, uint32_t seq) { 
        (void)count; (void)seq; 
    }
    
    void sendError(uint8_t code, uint32_t seq) { 
        (void)code; (void)seq; 
    }
    
    void sendRawFrame(uint8_t type, uint32_t seq, const uint8_t* payload, uint16_t len) {
        (void)type; (void)seq; (void)payload; (void)len;
    }
    
    bool readHeader(FrameHeader& h, bool& with_chk, uint16_t& chk, unsigned long timeout_ms) {
        (void)h; (void)with_chk; (void)chk; (void)timeout_ms;
        return false; // No data available in desktop stub
    }
    
    bool readExact(uint8_t* buf, size_t n, unsigned long timeout_ms) {
        (void)buf; (void)n; (void)timeout_ms;
        return false; // No data available in desktop stub
    }
    
private:
    // No serial member needed for stubs
};

#endif // NATIVE_BUILD

