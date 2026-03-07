#pragma once
#ifdef NATIVE_BUILD

#include "Arduino.h"
#include "core/transport.h"
#include <string>
#include <cstdarg>

// Forward declaration
extern Transport TR;

/**
 * LogSerial - Wrapper around Serial that sends output as LOG frames
 * This ensures all LOG_SERIAL output is sent to the Python client via binary protocol
 */
class LogSerial : public Print {
private:
    static constexpr size_t BUFFER_SIZE = 512;
    char line_buffer[BUFFER_SIZE];
    size_t buffer_pos = 0;
    
    void flush_line() {
        if (buffer_pos > 0) {
            // Send as LOG frame
            TR.sendLog(line_buffer, buffer_pos);
            // Also print to console for debugging
            Serial.write((const uint8_t*)line_buffer, buffer_pos);
            buffer_pos = 0;
        }
    }
    
public:
    // Override write() - called by all print functions
    size_t write(uint8_t c) override {
        if (c == '\n') {
            line_buffer[buffer_pos++] = c;
            flush_line();
        } else if (buffer_pos < BUFFER_SIZE - 1) {
            line_buffer[buffer_pos++] = c;
        } else {
            // Buffer full, flush it
            flush_line();
            line_buffer[buffer_pos++] = c;
        }
        return 1;
    }
    
    size_t write(const uint8_t *buffer, size_t size) override {
        for (size_t i = 0; i < size; i++) {
            write(buffer[i]);
        }
        return size;
    }
    
    // printf support
    size_t printf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        
        char temp_buffer[512];
        int len = vsnprintf(temp_buffer, sizeof(temp_buffer), format, args);
        va_end(args);
        
        if (len > 0) {
            return write((const uint8_t*)temp_buffer, len);
        }
        return 0;
    }
    
    // Explicit flush
    void flush() {
        flush_line();
        Serial.flush();
    }
    
    // Print/println overrides
    size_t println(const char* s) {
        size_t n = print(s);
        n += println();
        return n;
    }
    
    size_t println() {
        return write('\n');
    }
};

#endif // NATIVE_BUILD

