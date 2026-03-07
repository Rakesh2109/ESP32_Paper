# ESP32 HAL

This directory documents the ESP32 Hardware Abstraction Layer.

## Overview

The ESP32 HAL uses the **real** implementations from `src/`:
- `src/transport.h` / `src/transport.cpp` - Real serial communication
- `src/protocol_utils.h` / `src/protocol_utils.cpp` - Protocol helpers

## Platform Support

- **ESP32** (original)
- **ESP32-S2** (single-core, USB native)
- **ESP32-S3** (dual-core, USB native)
- **ESP32-C3** (RISC-V, single-core)
- **ESP32-P4** (high-performance variant)

## Transport Implementation

The real `Transport` class provides:
- **Serial communication** over UART/USB CDC
- **Frame-based protocol** with checksums
- **Binary logging** via LOG frames
- **Windowed ACK** mechanism for reliability

## How It Works

When `NATIVE_BUILD` is **NOT** defined (default for ESP32 builds):

```cpp
// hal_platform.h includes:
#include "transport.h"        // Real Transport class
#include "protocol_utils.h"   // Real protocol functions
```

The build system compiles:
- `src/transport.cpp` - Full serial I/O implementation
- `src/protocol_utils.cpp` - Protocol frame builders
- `src/main.cpp` - Application code (no changes needed!)

## Configuration

In `platformio.ini`:

```ini
[env:esp32]
platform = espressif32
framework = arduino
# No NATIVE_BUILD flag - uses real ESP32 implementations
```

## Key Features

### Transport Class
- `readExact()` - Blocking read with timeout
- `readHeader()` - Parse frame header with checksum
- `sendReady()` - Device ready beacon
- `sendAck()` - Acknowledge received frames
- `sendStat()` - Training/testing statistics
- `sendMetric()` - Detailed metrics
- `sendLog()` - Binary log messages
- `sendRawFrame()` - Generic frame sender

### Protocol Utils
- `proto_send_memprof()` - Memory profiling data
- Packs complex data structures into binary frames
- Handles truncation for large payloads

## Arduino Framework

ESP32 builds use the Arduino framework which provides:
- `Serial` - USB CDC or UART
- `Stream` - Base class for I/O
- `millis()`, `delay()` - Timing functions
- FreeRTOS - Multi-tasking

## Adding New ESP32 Variants

1. Add board configuration in `platformio.ini`
2. No HAL changes needed - same code works across all ESP32 variants
3. Adjust pins/peripherals as needed in application code

## See Also

- `src/transport.h` - Transport API reference
- `src/protocol.h` - Frame format definitions
- `../desktop/` - Desktop stubs for comparison

