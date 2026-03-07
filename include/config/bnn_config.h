// BinaryNet configuration constants

#pragma once

#ifndef BNN_MAX_FEATURES
#define BNN_MAX_FEATURES 256
#endif

#ifndef BNN_NUM_CLASSES
#define BNN_NUM_CLASSES 2
#endif

#ifndef BNN_LEARNING_RATE
#define BNN_LEARNING_RATE 0.001f
#endif

#ifndef BNN_WEIGHT_DECAY
#define BNN_WEIGHT_DECAY 1e-4f
#endif

#ifndef BNN_CLIP_VALUE
#define BNN_CLIP_VALUE 1.0f
#endif

#ifndef BNN_HIDDEN_LAYERS
#define BNN_HIDDEN_LAYERS {32, 16}
#endif

#ifndef ENABLE_DISPLAY
#define ENABLE_DISPLAY 0
#endif

#ifndef MAX_PACKED_BYTES
#define MAX_PACKED_BYTES 1024  // Reduced from 4096: for 256 features = 32 bytes packed + float = 1024 bytes
#endif

#ifndef MEMPROF_MAX_PAYLOAD
#define MEMPROF_MAX_PAYLOAD 3900
#endif

#ifndef RECV_Q_CAP
#define RECV_Q_CAP 128  // Balanced queue size: good throughput with lower RAM pressure.
#endif

#ifndef ACTIVE_SERIAL
#define ACTIVE_SERIAL Serial
#endif

#ifndef SERIAL2_BAUD
#define SERIAL2_BAUD 2000000
#endif

#ifndef SERIAL_RX_BUFFER_SIZE
#define SERIAL_RX_BUFFER_SIZE 65536
#endif

#ifndef TM_ENABLE_LOG_FRAMES
#define TM_ENABLE_LOG_FRAMES 0  // Disable LOG frames by default for max throughput.
#endif

#ifndef TM_ENABLE_VERBOSE_FINAL_LOGS
#define TM_ENABLE_VERBOSE_FINAL_LOGS 0  // Keep OPC_SHOW_FINAL structured-only by default.
#endif

// Hoeffding speed profile (applies when MODEL_TYPE=MODEL_HOEFFDING).
// These defaults reduce split churn and keep throughput stable for larger feature sets.
#ifndef HT_NUM_THRESHOLD_BINS
#define HT_NUM_THRESHOLD_BINS 8
#endif

#ifndef HT_DELTA
#define HT_DELTA 1e-5f
#endif

#ifndef HT_TIE_THRESHOLD
#define HT_TIE_THRESHOLD 0.10f
#endif

#ifndef HT_GRACE_PERIOD
#define HT_GRACE_PERIOD 128
#endif

#ifndef HT_MIN_SAMPLES_SPLIT
#define HT_MIN_SAMPLES_SPLIT 256
#endif

#ifndef HT_MAX_DEPTH
#define HT_MAX_DEPTH 8
#endif
