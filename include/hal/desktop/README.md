# Desktop HAL

This directory contains Hardware Abstraction Layer implementations for desktop platforms (Windows, Linux, macOS).

## Overview

The desktop HAL **reuses the same `Transport` class as ESP32** but provides different Stream backends:

1. **Stub Stream** (default) - For unit testing without communication
2. **Socket Stream** (`socket_stream.h`) - TCP socket backend for integration testing with `test_serial.py`

This design means:
- ✅ **Zero code duplication** - Transport protocol logic is shared
- ✅ **Same behavior** - Desktop and ESP32 use identical Transport implementation  
- ✅ **Just swap backends** - Change the Stream, not the Transport

## Platform Support

- **Windows** (MSVC, MinGW)
- **Linux** (GCC, Clang)
- **macOS** (Clang, Apple Silicon & Intel)

## Files

- `transport_desktop.h` - Transport stub (for unit tests without communication)
- `socket_stream.h` - Socket Stream header (Arduino Stream over TCP)
- `socket_stream.cpp` - Socket Stream implementation
- `platform_windows.h` - Windows-specific utilities
- `platform_linux.h` - Linux-specific utilities
- `platform_macos.h` - macOS-specific utilities

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           Application Code (main.cpp)                │
│           Uses: Transport class                      │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ Transport tr(stream);
                    ▼
┌─────────────────────────────────────────────────────┐
│         Transport Class (src/transport.cpp)          │
│         - SAME code for ESP32 and Desktop            │
│         - Frames, checksums, protocol logic          │
└───────────────────┬─────────────────────────────────┘
                    │
                    │ Uses: Stream interface
                    ▼
       ┌────────────┴────────────┐
       │                         │
       ▼                         ▼
┌─────────────┐          ┌──────────────┐
│  ESP32      │          │  Desktop     │
│  Serial     │          │  SocketStream│
│  (Hardware) │          │  (TCP)       │
└─────────────┘          └──────────────┘
```

### Stream Backends

#### 1. Stub Stream (Default - Unit Tests)

Used when `HAL_SOCKET_TRANSPORT` is NOT defined:
- **No communication** - All operations are no-ops
- **Fast** - No network overhead
- **Lightweight** - No socket dependencies
- Perfect for unit testing Transport protocol logic

#### 2. Socket Stream (`socket_stream.h`)

Used when `HAL_SOCKET_TRANSPORT` is defined:
- **Implements Arduino `Stream` interface**
- **TCP socket** server or client
- **Compatible with `test_serial.py`** over `socket://`
- **Same Transport code** as ESP32 hardware

## Usage

Desktop HAL is automatically selected when `NATIVE_BUILD` is defined:

```cpp
// In platformio.ini
[env:native]
build_flags = -D NATIVE_BUILD
```

## Adding Platform-Specific Code

If you need platform-specific behavior:

1. Create `platform_<os>.h` in this directory
2. Use compiler detection:
   ```cpp
   #if defined(_WIN32)
       // Windows-specific
   #elif defined(__linux__)
       // Linux-specific
   #elif defined(__APPLE__)
       // macOS-specific
   #endif
   ```
3. Include in `hal_platform.h`

## Testing

The desktop builds allow:
- Unit testing without hardware
- Memory leak detection with Valgrind (Linux/macOS)
- Profiling with native tools
- Continuous integration on GitHub Actions

## Limitations

Desktop stubs do NOT provide:
- Serial communication
- RTOS (FreeRTOS)
- Hardware timers
- GPIO
- ESP-specific peripherals

These are shimmed in `include/arduino_shim/`.

