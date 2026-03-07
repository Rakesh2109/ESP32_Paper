// ESP32 HAL - Transport instance
#ifndef NATIVE_BUILD

#include <Arduino.h>
#include "core/transport.h"

#ifndef ACTIVE_SERIAL
  #define ACTIVE_SERIAL  Serial
#endif
Transport TR(ACTIVE_SERIAL);

#endif // !NATIVE_BUILD

