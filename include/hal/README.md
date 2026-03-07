# Hardware Abstraction Layer (HAL)

This directory contains the Hardware Abstraction Layer that provides platform-specific implementations for ESP32 and desktop builds.

The HAL enables the same application code (`src/main.cpp`) to compile and run on:
- **ESP32 hardware** (ESP32, S2, S3, C3, P4) - Real hardware with serial communication
- **Desktop platforms** (Windows, Linux, macOS) - Stubs for development and testing

## Directory Structure

```
include/hal/
├── README.md                          # This file
├── hal_platform.h                     # Main HAL header (platform selector)
├── desktop/                           # Desktop platform implementations
│   ├── README.md                      # Desktop HAL documentation
│   ├── transport_desktop.h            # Transport stub for Windows/Linux/macOS
│   ├── platform_windows.h             # Windows-specific utilities
│   ├── platform_linux.h               # Linux-specific utilities
│   └── platform_macos.h               # macOS-specific utilities
└── esp32/                             # ESP32 platform documentation
    └── README.md                      # ESP32 HAL documentation
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Application Code                      │
│                   (src/main.cpp)                         │
│  - No #ifdef for platforms                               │
│  - Uses Transport, proto_send_memprof, etc.             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      │ #include "hal/hal_platform.h"
                      ▼
┌─────────────────────────────────────────────────────────┐
│              HAL Platform Selector                       │
│             (hal_platform.h)                             │
│  Detects platform and includes correct implementation    │
└─────┬───────────────────────────────────────────┬───────┘
      │                                           │
      │ NATIVE_BUILD defined?                     │
      │                                           │
  YES │                                           │ NO
      │                                           │
      ▼                                           ▼
┌─────────────────────┐              ┌──────────────────────┐
│  Desktop Platform   │              │   ESP32 Platform     │
│   (Stubs)           │              │   (Real Hardware)    │
├─────────────────────┤              ├──────────────────────┤
│ transport_desktop.h │              │ src/transport.h      │
│   - No-op methods   │              │   - Serial I/O       │
│   - sendLog→stdout  │              │   - Real frames      │
│                     │              │                      │
│ platform_*.h        │              │ Arduino framework    │
│   - Windows         │              │ FreeRTOS             │
│   - Linux           │              │ ESP-IDF              │
│   - macOS           │              │                      │
└─────────────────────┘              └──────────────────────┘
         │                                      │
         │                                      │
         └──────────┬───────────────────────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │   protocol_utils.h     │
         │  (Platform-agnostic)   │
         │  - Works with both     │
         └────────────────────────┘
```

## Platform Selection (`hal_platform.h`)

The main HAL header automatically includes the correct implementation based on the build target:

- **Desktop Build** (`NATIVE_BUILD` defined): Includes stub implementations
  - `desktop/transport_desktop.h` - Transport class stubs (shared)
  - `desktop/platform_windows.h` - Windows-specific utilities
  - `desktop/platform_linux.h` - Linux-specific utilities  
  - `desktop/platform_macos.h` - macOS-specific utilities
  
- **ESP32 Build** (default): Includes real implementations
  - `transport.h` - Real Transport class (from `src/`)
  - `protocol_utils.h` - Protocol helpers (platform-agnostic)

### Desktop Platform Stubs

The desktop platform provides stub implementations that:
- Allow compilation without ESP32-specific code
- Enable testing and development on desktop platforms
- Maintain API compatibility with ESP32 code

**Key Features:**
- `Transport` class with no-op methods (no serial communication)
- `sendLog()` prints to stdout for debugging
- `utils_log_binary()` prints to stdout
- Shared implementation across Windows, Linux, and macOS
- Platform-specific headers available for OS-specific code

## Usage

In `main.cpp`, simply include the HAL header:

```cpp
#include "hal/hal_platform.h"
```

This provides:
- `Transport` class (real or stub depending on platform)
- `proto_send_memprof()` function
- `utils_log_binary()` function

**No code changes needed in `main.cpp`!** The HAL automatically provides the right implementation.

## Key Benefits

✅ **Single codebase** - No `#ifdef` needed in application code (`main.cpp`)  
✅ **API compatibility** - Same Transport interface across all platforms  
✅ **Easy testing** - Build and run on desktop for quick iteration  
✅ **Clean separation** - Platform-specific code isolated in HAL layer  
✅ **Cross-platform** - Automatically detects Windows/Linux/macOS at compile time  
✅ **Maintainable** - Changes to application code work everywhere automatically  

## Design Principles

1. **Zero Application Changes** - `main.cpp` includes only `hal/hal_platform.h`
2. **Compile-Time Selection** - Platform detection via preprocessor macros
3. **Shared Code** - `protocol_utils.h` works with both real and stub Transport
4. **No Runtime Overhead** - All platform selection happens at compile time
5. **Consistent APIs** - Desktop stubs match ESP32 signatures exactly

## Desktop Build Configuration

In `platformio.ini`:

```ini
[env:native]
platform = native
build_flags = 
    -I include/hal           # HAL headers
    -I include/arduino_shim  # Arduino shims
    -D NATIVE_BUILD          # Enable desktop build
    -D HAL_SOCKET_TRANSPORT  # Enable socket communication
build_src_filter = 
    +<*>                                        # Include all sources
    +<../include/arduino_shim/*.cpp>            # Arduino shims
    +<../include/hal/desktop/socket_stream.cpp> # Socket backend
    +<../include/hal/desktop/hal_transport.cpp> # Desktop transport
    +<../include/hal/desktop/hal_main.cpp>      # Native main()
    -<*.bak>                                    # Exclude backups
```

### Platform-Specific Detection

The HAL automatically detects the OS at compile time:
- **Windows**: Uses `platform_windows.h` (detects `_WIN32`)
- **Linux**: Uses `platform_linux.h` (detects `__linux__`)
- **macOS**: Uses `platform_macos.h` (detects `__APPLE__`)

## Function Reference

### Transport Class (Native Stub)

```cpp
Transport(Stream& ser);              // Constructor (no-op)
void sendLog(const char*, uint16_t); // No-op
void sendReady(uint8_t, uint8_t, uint32_t, uint32_t); // No-op
void sendStat(uint32_t, uint32_t, uint32_t, float); // No-op
void sendMetric(uint32_t, const MetricPayload&); // No-op
void sendAck(uint32_t, uint32_t); // No-op
void sendDone(uint32_t, uint32_t); // No-op
void sendError(uint8_t, uint32_t); // No-op
bool readHeader(FrameHeader&, bool&, uint16_t&, unsigned long); // Returns false
bool readExact(uint8_t*, size_t, unsigned long); // Returns false
```

### Protocol Utils (Native Stub)

```cpp
void proto_send_memprof(...);  // No-op stub with full signature
```

### Binary Logging

```cpp
void utils_log_binary(const char* msg, uint16_t len); // Prints to stdout
```

