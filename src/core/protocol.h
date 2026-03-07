// Protocol definitions (C-compatible)
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FRAME_MAGIC         0xA55A
#define FLAG_FLETCHER       0x01
// Configuration flag bits for OPC_CONFIG 'flags' field
enum {
    CFG_TWINE        = 1 << 0,
    CFG_INPUT_REAL   = 1 << 1,
    CFG_MULTICLASS   = 1 << 2,
    CFG_MODEL_DENSE  = 1 << 3,
    CFG_TM_RUNTIME   = 1 << 4,
};

// Online preprocessor identifiers carried in ConfigPayload.preproc_id.
// Keep values stable for host/device compatibility.
enum {
    PREPROC_NONE              = 0,
    PREPROC_ONLINE_STANDARDIZE = 1,
    PREPROC_TWINE             = 2,
};

// Protocol version
#define PROTO_VER_MAJOR     2
#define PROTO_VER_MINOR     0

// Egress/Ingress types
#define FRAME_TYPE_RECORD   0x01
#define FRAME_TYPE_RECORD_MC 0x07  // Bit-packed multiclass record
#define FRAME_TYPE_SAMPLE_RAW 0x06  // Raw real-valued sample row
#define FRAME_TYPE_RECORD_BATCH 0x09 // Bit-packed binary batch: BatchHeader + count*(label+packed)
#define FRAME_TYPE_CMD      0x02
#define FRAME_TYPE_ACK      0x81
#define FRAME_TYPE_STAT     0x82
#define FRAME_TYPE_READY    0x83
#define FRAME_TYPE_DONE     0x84
#define FRAME_TYPE_ERROR    0x85
#define FRAME_TYPE_METRIC   0x86
#define FRAME_TYPE_MEMPROF  0x87
#define FRAME_TYPE_LOG      0x88
#define FRAME_TYPE_MEM_EFFICIENCY 0x89
#define FRAME_TYPE_PERF_BENCH     0x8A
#define FRAME_TYPE_PROTOCOL_STATS 0x8B

// Opcodes
#define OPC_STATUS          0x3F  // '?'
#define OPC_SYNC            0x59  // 'Y' - Wait until RX queue drained
#define OPC_RESET           0x52  // 'R' - Full hardware reset
#define OPC_RESET_STATE     0x72  // 'r' - Reset state to READY (soft reset)
#define OPC_START_TRAIN     0x53  // 'S'
#define OPC_START_TEST      0x56  // 'V'
#define OPC_BUFFER          0x42  // 'B' + uint16 window
#define OPC_SNAPSHOT        0x4D  // 'M'
#define OPC_MEMPROF         0x50  // 'P'
#define OPC_SHOW_FINAL      0x46  // 'F'
#define OPC_MODEL_SELECT    0x4C  // 'L' + uint8 model_id
// Configuration (feature flags)
#define OPC_CONFIG          0x43  // 'C'

// Error codes
#define ERR_BAD_N           1
#define ERR_BAD_LEN         2
#define ERR_NO_MEM          3
#define ERR_INCOMPLETE      4
#define ERR_BAD_FRAME       5
#define ERR_BAD_HDR         6
#define ERR_BAD_BODY        7
#define ERR_CHECKSUM        8
#define ERR_OVERFLOW        9

#pragma pack(push,1)
typedef struct {
    uint16_t magic;
    uint8_t  type;
    uint8_t  flags;
    uint16_t len;
    uint32_t seq;
} FrameHeader;

typedef struct {
    uint8_t opcode; /* + optional payload */
} CmdPacket;

typedef struct { uint32_t header_le; } PackedHeader; // bit31: label, bits[30:0]: N
typedef struct {
    uint16_t count_le;   // Number of records in batch
    uint16_t nfeat_le;   // Shared feature count for all records
} BatchHeader;
// Multiclass bit-packed header (little-endian)
typedef struct {
    uint16_t nfeat_le;  // number of features (bit-packed)
    uint8_t  label;     // class label 0..K-1
    uint8_t  _rsv;      // reserved
} PackedHeaderMC;
// Raw sample header: little-endian
typedef struct {
    uint16_t nfeat_le;   // number of real-valued features in row
    uint8_t  label;      // 0/1 label
    uint8_t  dtype;      // 1=float32
} RawSampleHeader;
typedef struct { uint8_t major, minor; uint16_t _pad; uint32_t trained_le, tested_le; } ReadyPayload;
typedef struct { uint32_t count_le, last_seq_le; } AckDonePayload;
typedef struct { uint32_t trained_le, tested_le; float acc_le; } StatPayload;

// Config payload (feature flags)
typedef struct {
    uint8_t flags;          // CFG_* bits (CFG_TWINE kept as legacy TWINE compatibility bit)
    uint8_t preproc_param0; // Preprocessor parameter 0 (TWINE: bits per feature)
    uint16_t preproc_id;    // PREPROC_* identifier
} ConfigPayload;

// Extended config payload with runtime TM hyperparameters.
// Backward compatible parsing:
// - V1 payload: ConfigPayload
// - V2 payload: ConfigPayloadV2 (when CFG_TM_RUNTIME is set)
typedef struct {
    uint8_t flags;          // CFG_* bits
    uint8_t preproc_param0; // Preprocessor parameter 0 (TWINE: bits per feature)
    uint16_t preproc_id;    // PREPROC_* identifier
    uint16_t tm_clauses;    // Runtime TM clause count
    int16_t tm_threshold;   // Runtime TM threshold T
    int16_t tm_specificity; // Runtime TM specificity s
    uint32_t tm_seed;       // Runtime TM seed
    uint8_t tm_init_density_pct; // Runtime sparse/BO init literal density [0..100]
} ConfigPayloadV2;
typedef struct { uint32_t trained_le, train_ok_le; float ema_acc_le, ema_score_le; int32_t last_score_le; uint8_t last_pred; uint8_t _r[3]; } MetricPayload;

typedef struct {
    uint32_t trained_le;
    uint32_t tested_le;
    float    acc_le;
    uint32_t free_heap_le;
    uint32_t min_heap_le;
    uint32_t max_alloc_heap_le;
    uint64_t util_cur_alloc_le;
    uint64_t util_peak_alloc_le;
    uint32_t util_alloc_count_le;
    uint32_t util_free_count_le;
    uint32_t util_active_allocs_le;
    uint32_t total_entries_le;
    uint32_t sent_entries_le;
    uint8_t  truncated;
    uint8_t  _pad[3];
} MemProfHeader;

// Memory Efficiency Analysis Payload
typedef struct {
    uint32_t tm_memory_le;          // TM total memory usage
    float    memory_per_clause;     // Memory per clause
    float    memory_per_feature;    // Memory per feature
    float    memory_per_cf;         // Memory per clause*feature
    uint32_t free_heap_le;          // Free heap
    uint32_t max_alloc_le;          // Largest free block
    float    fragmentation;         // Fragmentation percentage
    uint64_t current_allocated_le;  // Current allocated
    uint64_t peak_allocated_le;     // Peak allocated
    float    memory_efficiency;     // Efficiency percentage
    uint32_t alloc_count_le;        // Allocation count
    uint32_t free_count_le;         // Free count
    uint64_t total_allocated_le;    // Total allocated
    float    avg_alloc_size;        // Average allocation size
    float    alloc_frequency;       // Allocations per KB
} MemEfficiencyPayload;

// Performance Benchmarks Payload
typedef struct {
    float    train_throughput;      // Training samples/sec
    float    test_throughput;       // Testing samples/sec
    float    mem_per_train_sample;  // Memory per training sample
    float    mem_per_test_sample;   // Memory per testing sample
    float    throughput_mem_ratio;  // Throughput/memory ratio
    uint32_t chip_used_heap_le;     // Chip RAM used
    uint32_t tm_core_memory_le;     // TM core memory
    uint32_t sys_overhead_le;       // System overhead
    float    tm_mem_percent;        // TM memory percentage
    float    overhead_percent;      // Overhead percentage
    float    avg_alloc_size;        // Average allocation size
    float    alloc_overhead_pct;    // Allocation overhead %
    uint32_t total_overhead_le;     // Total overhead bytes
    float    ta_percentage;         // TA state contiguity %
    float    memory_utilization;    // Memory utilization %
    float    performance_score;     // Performance score %
    float    efficiency_ratio;      // Efficiency ratio
    float    precision;             // Precision = TP / (TP + FP)
    float    recall;                // Recall = TP / (TP + FN)
    float    f1_score;              // F1 = 2 * (precision * recall) / (precision + recall)
} PerfBenchPayload;

// Protocol Frame Statistics Payload
typedef struct {
    uint32_t frames_received_le;
    uint32_t frames_valid_le;
    uint32_t rejected_size_le;
    uint32_t rejected_checksum_le;
    uint32_t rejected_format_le;
    float    rejection_rate;
} ProtocolStatsPayload;

#pragma pack(pop)

static inline uint16_t fletcher16_update(uint16_t prev, const uint8_t* d, uint32_t len){
    uint16_t s1 = (uint16_t)(prev & 0xFF), s2 = (uint16_t)((prev >> 8) & 0xFF);
    while (len--) { s1 += *d++; if (s1 >= 255) s1 -= 255; s2 += s1; if (s2 >= 255) s2 -= 255; }
    return (uint16_t)((s2 << 8) | s1);
}

#ifdef __cplusplus
}
#endif


