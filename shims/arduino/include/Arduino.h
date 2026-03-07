#pragma once
// Arduino.h shim for native compilation

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>
#include <freertos/FreeRTOS.h>  // For taskYIELD()
#ifdef __cplusplus
extern "C" {
#endif

// Basic Arduino types
typedef uint8_t byte;

// Don't define boolean on Windows - conflicts with Windows SDK
#if !defined(_WIN32) && !defined(_WIN64)
typedef bool boolean;
#endif

// Pin modes (unused in native, but needed for compilation)
// Don't define INPUT/OUTPUT on Windows - conflicts with Windows SDK
#if !defined(_WIN32) && !defined(_WIN64)
#define INPUT           0x00
#define OUTPUT          0x01
#define INPUT_PULLUP    0x02
#endif

// Pin states
#define LOW             0x00
#define HIGH            0x01

// Math constants
#define PI              3.1415926535897932384626433832795
#define HALF_PI         1.5707963267948966192313216916398
#define TWO_PI          6.283185307179586476925286766559
#define DEG_TO_RAD      0.017453292519943295769236907684886
#define RAD_TO_DEG      57.295779513082320876798154814105

// Serial print formats
#define DEC             10
#define HEX             16
#define OCT             8
#define BIN             2

// Timing functions (implemented in Arduino.cpp for native)
unsigned long millis(void);
unsigned long micros(void);
void delay(unsigned long ms);
void delayMicroseconds(unsigned int us);

// GPIO functions (stubs)
static inline void pinMode(uint8_t pin, uint8_t mode) { (void)pin; (void)mode; }
static inline void digitalWrite(uint8_t pin, uint8_t val) { (void)pin; (void)val; }
static inline int digitalRead(uint8_t pin) { (void)pin; return 0; }

// Math functions
// Note: min/max/abs/round macros removed to avoid conflicts with C++ std library
// Use std::min/std::max/std::abs/std::round in C++ code instead
#define constrain(amt,low,high) ((amt)<(low)?(low):((amt)>(high)?(high):(amt)))
#define radians(deg) ((deg)*DEG_TO_RAD)
#define degrees(rad) ((rad)*RAD_TO_DEG)
#define sq(x) ((x)*(x))

// Bit manipulation
#define lowByte(w) ((uint8_t) ((w) & 0xff))
#define highByte(w) ((uint8_t) ((w) >> 8))
#define bitRead(value, bit) (((value) >> (bit)) & 0x01)
#define bitSet(value, bit) ((value) |= (1UL << (bit)))
#define bitClear(value, bit) ((value) &= ~(1UL << (bit)))
#define bitWrite(value, bit, bitvalue) (bitvalue ? bitSet(value, bit) : bitClear(value, bit))
#define bit(b) (1UL << (b))

// Simple String class for Arduino compatibility (must be before Print)
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
class String {
private:
    char* buffer;
    size_t len;
    
    void init(const char* str) {
        if (str) {
            len = strlen(str);
            buffer = (char*)malloc(len + 1);
            if (buffer) strcpy(buffer, str);
        } else {
            buffer = NULL;
            len = 0;
        }
    }
    
public:
    String() : buffer(NULL), len(0) {}
    
    String(const char* str) { init(str); }
    
    String(int n, int base = DEC) {
        char tmp[32];
        (void)base;
        snprintf(tmp, sizeof(tmp), "%d", n);
        init(tmp);
    }
    
    String(unsigned int n, int base = DEC) {
        char tmp[32];
        (void)base;
        snprintf(tmp, sizeof(tmp), "%u", n);
        init(tmp);
    }
    
    String(long n, int base = DEC) {
        char tmp[32];
        (void)base;
        snprintf(tmp, sizeof(tmp), "%ld", n);
        init(tmp);
    }
    
    String(unsigned long n, int base = DEC) {
        char tmp[32];
        (void)base;
        snprintf(tmp, sizeof(tmp), "%lu", n);
        init(tmp);
    }
    
    String(double n, int digits = 2) {
        char tmp[64];
        snprintf(tmp, sizeof(tmp), "%.*f", digits, n);
        init(tmp);
    }
    
    String(const String& other) {
        init(other.buffer);
    }
    
    ~String() {
        if (buffer) free(buffer);
    }
    
    String& operator=(const String& other) {
        if (this != &other) {
            if (buffer) free(buffer);
            init(other.buffer);
        }
        return *this;
    }
    
    String operator+(const String& other) const {
        String result;
        size_t new_len = len + other.len;
        result.buffer = (char*)malloc(new_len + 1);
        result.len = new_len;
        if (result.buffer) {
            if (buffer) strcpy(result.buffer, buffer);
            else result.buffer[0] = '\0';
            if (other.buffer) strcat(result.buffer, other.buffer);
        }
        return result;
    }
    
    const char* c_str() const { return buffer ? buffer : ""; }
    size_t length() const { return len; }
};

// Allow const char* + String concatenation
inline String operator+(const char* lhs, const String& rhs) {
    String result(lhs);
    return result + rhs;
}

// Print class stub for native
class Print {
public:
    virtual size_t write(uint8_t c) { return fputc(c, stdout) == EOF ? 0 : 1; }
    virtual size_t write(const uint8_t *buffer, size_t size) {
        return fwrite(buffer, 1, size, stdout);
    }
    
    // printf-style formatted printing
    size_t printf(const char* format, ...) {
        va_list args;
        va_start(args, format);
        int result = vprintf(format, args);
        va_end(args);
        return result < 0 ? 0 : (size_t)result;
    }
    
    size_t print(const char* s) { return ::printf("%s", s); }
    size_t println(const char* s) { return ::printf("%s\n", s); }
    size_t print(const String& s) { return print(s.c_str()); }
    size_t println(const String& s) { return println(s.c_str()); }
    size_t print(int n, int base = DEC) { (void)base; return ::printf("%d", n); }
    size_t println(int n, int base = DEC) { (void)base; return ::printf("%d\n", n); }
    size_t print(unsigned int n, int base = DEC) { (void)base; return ::printf("%u", n); }
    size_t println(unsigned int n, int base = DEC) { (void)base; return ::printf("%u\n", n); }
    size_t print(long n, int base = DEC) { (void)base; return ::printf("%ld", n); }
    size_t println(long n, int base = DEC) { (void)base; return ::printf("%ld\n", n); }
    size_t print(unsigned long n, int base = DEC) { (void)base; return ::printf("%lu", n); }
    size_t println(unsigned long n, int base = DEC) { (void)base; return ::printf("%lu\n", n); }
    size_t print(double n, int digits = 2) { (void)digits; return ::printf("%f", n); }
    size_t println(double n, int digits = 2) { (void)digits; return ::printf("%f\n", n); }
    size_t println() { return ::printf("\n"); }
};

// Stream base class (for Serial compatibility)
class Stream : public Print {
public:
    virtual int available() = 0;
    virtual int read() = 0;
    virtual int peek() { return -1; }  // Default implementation
    void setTimeout(unsigned long timeout) { (void)timeout; }
    
    // readBytes - read multiple bytes with timeout
    size_t readBytes(uint8_t* buffer, size_t length) {
        size_t count = 0;
        while (count < length) {
            int c = read();
            if (c < 0) break;  // No more data or timeout
            buffer[count++] = (uint8_t)c;
        }
        return count;
    }
};

// Serial stub
class SerialStub : public Stream {
public:
    void begin(unsigned long baud) { (void)baud; }
    void end() {}
    int available() override { return 0; }
    int read() override { return -1; }
    int peek() override { return -1; }
    size_t write(uint8_t c) override { return fputc(c, stdout) == EOF ? 0 : 1; }
    size_t write(const uint8_t *buffer, size_t size) override {
        return fwrite(buffer, 1, size, stdout);
    }
    void flush() { fflush(stdout); }
    void setRxBufferSize(size_t size) { (void)size; }
    void setTimeout(unsigned long timeout) { (void)timeout; }
};

// ESP class for ESP32 compatibility (real implementation in EspClass.cpp)
class EspClass {
public:
    uint32_t getFreeHeap();
    uint32_t getHeapSize();
    uint32_t getMinFreeHeap();
    uint32_t getMaxAllocHeap();
    uint32_t getPsramSize();
    uint32_t getFreePsram();
    uint8_t getChipRevision();
    const char* getSdkVersion();
    uint32_t getCpuFreqMHz();
    uint32_t getCycleCount();
    void restart();
};

extern SerialStub Serial;
extern EspClass ESP;
#endif // __cplusplus

