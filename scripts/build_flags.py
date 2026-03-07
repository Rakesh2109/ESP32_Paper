# build_flags.py - Dynamic build flags for ESP32 Tsetlin Machine
Import("env")

# Add memory optimization flags
env.Append(
    LINKFLAGS=[
        "-Wl,--gc-sections"  # Remove unused sections
    ],
    CCFLAGS=[
        "-Os",  # Optimize for size
        "-ffunction-sections",  # Place each function in its own section
        "-fdata-sections"  # Place each data item in its own section
    ]
)

# Add ESP32-specific optimizations
env.Append(
    CCFLAGS=[
        "-DCORE_DEBUG_LEVEL=0"  # Minimal debug output
    ]
)

# Memory optimization for Tsetlin Machine
env.Append(
    CCFLAGS=[
        "-DTM_MEMORY_OPTIMIZED=1",  # Enable memory optimizations
        "-DMAX_FEATURES=128",  # Use header definition
        "-DMAX_CLAUSES=1000"
    ]
)

print("Build flags applied: Memory optimization enabled")


