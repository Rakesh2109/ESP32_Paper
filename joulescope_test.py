#!/usr/bin/env python3
"""
Connect to Joulescope and print live voltage, current, and power readings.
Requires: pip install joulescope (and on macOS: brew install libusb)
"""

import sys
import time

def main():
    print("Joulescope – connect and get readings\n")

    # 1. Scan for devices
    import joulescope
    devices = joulescope.scan()
    if not devices:
        print("No Joulescope device found. Connect a Joulescope and try again.")
        return 1
    print(f"Found {len(devices)} device(s):")
    for d in devices:
        print(f"  - {d}")
    print()

    # 2. Connect and take readings
    device = joulescope.scan_require_one(config='auto')
    print("Connecting...")
    with device:
        print("Connected. Streaming readings (Ctrl+C to stop):\n")
        print(f"{'Time (s)':<12} {'Current (A)':<14} {'Voltage (V)':<14} {'Power (W)':<12}")
        print("-" * 52)
        t0 = time.perf_counter()
        duration_sec = 5.0  # run for 5 seconds; use None for infinite (Ctrl+C to stop)
        try:
            while True:
                data = device.read(contiguous_duration=0.001)
                if data is not None and len(data) > 0:
                    current_a, voltage_v = data[-1, 0], data[-1, 1]
                    power_w = current_a * voltage_v
                    elapsed = time.perf_counter() - t0
                    print(f"{elapsed:<12.3f} {current_a:<14.6f} {voltage_v:<14.6f} {power_w:<12.6f}", flush=True)
                if duration_sec is not None and elapsed >= duration_sec:
                    print("\nDuration reached.")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopped.")
    print("Disconnected.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
