/*
 * ESP32 BinaryNet Streamer (Protocol v2, Fletcher-32) - RELIABLE
 * - Keeps original data structures and frame formats.
 * - Transport task never exits; DONE and READY are periodic beacons.
 * - Windowed ACKs plus periodic ACKs if outstanding.
 * - Safer timeouts and soft resync.
 */

// --- Core selection helpers (works across ESP32, S2, S3, C3, etc.) ---
#if __has_include("soc/soc_caps.h")
  #include "soc/soc_caps.h"
  #define HAS_DUAL_CORE   (SOC_CPU_CORES_NUM > 1)
#else
  // Fallback if soc_caps isn't available
  #if defined(CONFIG_IDF_TARGET_ESP32) || defined(CONFIG_IDF_TARGET_ESP32S3)
    #define HAS_DUAL_CORE 1
  #else
    #define HAS_DUAL_CORE 0
  #endif
#endif

// Allow overriding from build flags, e.g. -DCORE_TRANSPORT=1
#ifndef CORE_TRANSPORT
  #if HAS_DUAL_CORE
    #define CORE_TRANSPORT 1   // put transport on the "other" core
    #define CORE_COMPUTE   0
  #else
    #define CORE_TRANSPORT 0   // single-core chips: both on core 0
    #define CORE_COMPUTE   0
  #endif
#endif




#include <Arduino.h>
#include <stdarg.h>
#include <stdint.h>
#include <cstddef>
#include <cstring>
#include <cmath>  // For std::sqrt in feature normalization
#include <string.h>
#include <algorithm>  // std::min
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_task_wdt.h"
 

 
#include <memory>
#include <vector>
#include <variant>
#if defined(ESP_PLATFORM)
#include "esp_system.h"
#endif

#include "core/utils.h"
#include "debug/benchmark.h"
#include "core/protocol.h"
#include "hal/hal_platform.h"
#include "debug/bounds_monitor.h"
#include "core/analysis_helpers.h"
#include "config/bnn_config.h"
#include "config/tm_config.h"
#include "models/bnn_types.h"
#include "models/bnn_model.h"
#include "models/efdt_model.h"
#include "models/hat_model.h"
#include "models/hoeffding_tree_model.h"
#include "models/sgt_model.h"
#include "models/tm_sparse_model.h"
#include "models/tm_vanilla_model.h"
#include "models/tm_bo_model.h"

#ifndef MAX_FEATURES
#define MAX_FEATURES 256
#endif

#define MODEL_EFDT 1
#define MODEL_HOEFFDING 2
#define MODEL_HAT 3
#define MODEL_SGT 4
#define MODEL_BNN 5
#define MODEL_TM_SPARSE 6
#define MODEL_TM_VANILLA 7
#define MODEL_TM_BO 8

#ifndef MODEL_TYPE
#define MODEL_TYPE MODEL_HOEFFDING
#endif

enum class RuntimeModelType : uint8_t {
  EFDT = MODEL_EFDT,
  HOEFFDING = MODEL_HOEFFDING,
  HAT = MODEL_HAT,
  SGT = MODEL_SGT,
  BNN = MODEL_BNN,
  TM_SPARSE = MODEL_TM_SPARSE,
  TM_VANILLA = MODEL_TM_VANILLA,
  TM_BO = MODEL_TM_BO,
};

static constexpr RuntimeModelType compile_default_model_type() {
#if MODEL_TYPE == MODEL_EFDT
  return RuntimeModelType::EFDT;
#elif MODEL_TYPE == MODEL_HOEFFDING
  return RuntimeModelType::HOEFFDING;
#elif MODEL_TYPE == MODEL_HAT
  return RuntimeModelType::HAT;
#elif MODEL_TYPE == MODEL_SGT
  return RuntimeModelType::SGT;
#elif MODEL_TYPE == MODEL_BNN
  return RuntimeModelType::BNN;
#elif MODEL_TYPE == MODEL_TM_SPARSE
  return RuntimeModelType::TM_SPARSE;
#elif MODEL_TYPE == MODEL_TM_VANILLA
  return RuntimeModelType::TM_VANILLA;
#elif MODEL_TYPE == MODEL_TM_BO
  return RuntimeModelType::TM_BO;
#else
  return RuntimeModelType::HOEFFDING;
#endif
}

struct BnnRuntimeConfig {
  std::size_t num_classes = BNN_NUM_CLASSES;
  std::vector<std::size_t> hidden_layers = std::vector<std::size_t> BNN_HIDDEN_LAYERS;
  float learning_rate = BNN_LEARNING_RATE;
  float weight_decay = BNN_WEIGHT_DECAY;
  float clip_value = BNN_CLIP_VALUE;
  unsigned int seed = 0u;
};

using ModelVariant = std::variant<
    std::monostate,
    tree::EfdtModel,
    tree::HoeffdingTreeModel,
    tree::HatModel,
    tree::SgtModel,
    bnn::BinaryNeuralNetwork,
    tm_model::TmSparseModel,
    tm_model::TmVanillaModel,
    tm_model::TmBoModel>;

namespace {

ModelVariant g_model{};
RuntimeModelType g_selected_model = compile_default_model_type();
tree::EfdtConfig g_cfg_efdt{};
tree::HoeffdingTreeConfig g_cfg_hoeffding{};
tree::HatConfig g_cfg_hat{};
tree::SgtConfig g_cfg_sgt{};
BnnRuntimeConfig g_cfg_bnn{};
tm_model::TmSparseConfig g_cfg_tm_sparse{};
tm_model::TmVanillaConfig g_cfg_tm_vanilla{};
tm_model::TmBoConfig g_cfg_tm_bo{};
bool g_model_init_failed = false;
int g_model_feature_count = 0;
uint32_t g_model_memory_bytes = 0;

// Online preprocessors (extensible): NONE, STANDARDIZE, TWINE.
struct FeatureStats {
  float mean = 0.0f;
  float m2 = 0.0f;  // Variance accumulator (Welford)
  uint32_t count = 0u;
};

struct TwineFeatureStats {
  float min_value = 0.0f;
  float max_value = 0.0f;
  uint32_t count = 0u;
};

enum class OnlinePreproc : uint16_t {
  NONE = PREPROC_NONE,
  STANDARDIZE = PREPROC_ONLINE_STANDARDIZE,
  TWINE = PREPROC_TWINE,
};

struct OnlinePreprocConfig {
  OnlinePreproc mode = OnlinePreproc::NONE;
  uint8_t param0 = 0;  // TWINE: bits per feature
};

std::vector<FeatureStats> g_standardize_stats;
std::vector<TwineFeatureStats> g_twine_stats;
OnlinePreprocConfig g_preproc_cfg{};

}  // namespace

static bool model_ensure_model(int nfeat);
static void model_reset_runtime();
static void model_apply_runtime_defaults();
static const char* model_name(RuntimeModelType type);
static bool model_from_id(uint8_t raw, RuntimeModelType& out_type);
static bool model_uses_tm_packed_path(RuntimeModelType type);
static uint32_t model_selected_num_classes();
static float model_train_sample_dispatch(const bnn::Sample& sample, int* pred_before_out);
static int model_predict_dispatch(std::vector<float>& features);
static float model_train_packed_dispatch(const uint8_t* packed_bits, std::size_t nfeat, int32_t label, int* pred_before_out);
static int model_predict_packed_dispatch(const uint8_t* packed_bits, std::size_t nfeat);
static void model_convert_bits_to_features(const uint8_t* packed_bits, int nfeat, float* out_features);
static void model_load_raw_features(const uint8_t* raw_bytes, int nfeat, float* out_features);
static const char* preproc_name(OnlinePreproc mode);
static bool preproc_from_id(uint16_t raw, OnlinePreproc& out_mode);
static uint8_t preproc_twine_bits();
static void preproc_reset_state(int nfeat);
static void model_preprocess_features(float* features, int nfeat, bool is_training, bool input_real);
static void model_on_train_sample(int pred, int label, float loss);
static void model_on_test_sample(int pred, int label);

#if ENABLE_DISPLAY
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <SPI.h>
#endif
 
 // ---------------- Serial wiring ----------------
// ESP32-S3: Uses USB Serial (CDC) via Serial
// ESP32-C3: Uses hardware UART via Serial
// Configure hardware UART (Serial2) pins for external serial on GPIO37/38
// Use GPIO38 as RXD and GPIO37 as TXD (swap if your wiring differs)
//#define SERIAL2_TX_PIN 37
//#define SERIAL2_RX_PIN 38
//#define SERIAL2_BAUD   921600

// LOG_SERIAL writes to binary log frames
#ifdef NATIVE_BUILD
  #include "hal/desktop/log_serial.h"
  extern LogSerial LOG_SERIAL_Instance;
  #define LOG_SERIAL LOG_SERIAL_Instance
#else
  // On ESP32, use the BinaryLogSerial instance declared in utils.h (LOG_SERIAL)
  // It forwards lines to host as binary LOG frames via utils_log_binary.
  // No macro needed here.
#endif


#if ENABLE_DISPLAY
// TFT Display pins (adjust for your hardware)
//#define TFT_CS        5
//#define TFT_DC        6
//#define TFT_RST       7
//#define TFT_BACKLITE  8
// TFT Display pins - using board defaults
// TFT_CS, TFT_DC, TFT_RST, TFT_BACKLITE are already defined by the board

// Display Colors
#define COLOR_BG       0x0000
#define COLOR_TITLE    0xFFFF
#define COLOR_INFO     0x07FF
#define COLOR_SUCCESS  0x07E0
#define COLOR_ERROR    0xF800
#define COLOR_WARNING  0xFFE0
#endif
 




 // ---------------- Device state ----------------
 struct DeviceState {
   enum Mode : uint8_t { TRAIN=0, TEST=1 } mode = TRAIN;
   uint16_t window = 16;
   uint8_t  proto_major = PROTO_VER_MAJOR, proto_minor = PROTO_VER_MINOR;
   unsigned long last_hb_ms = 0;
 
  bool model_initialized = false;
   int32_t feat = 0;
 
   uint32_t trained = 0, tested = 0, correct = 0;
 
   // Train metrics (snapshot)
   uint32_t train_ok = 0;
   int32_t  last_score = 0;
   uint8_t  last_pred  = 0;
   float    ema_acc=0.f, ema_score=0.f;
   const float ema_alpha = 0.01f;
 
  // micro-timing accumulators (optional)
  uint64_t score_us_accum = 0;
  uint32_t score_calls    = 0;
  uint64_t update_us_accum= 0;
  uint32_t update_calls   = 0;

  // Power monitoring
  BenchmarkContext power_ctx;
  float total_power_mw = 0.0f;
  float avg_power_mw = 0.0f;
  uint32_t power_samples = 0;

 bool pending_memprof_on_test = false;
} G;

#if ENABLE_DISPLAY
// TFT Display
static Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);
#endif
 
 struct WindowState { uint32_t sent, last_seq, last_acked; uint16_t cap, left; };
 

static constexpr std::size_t RECENTRY_DATA_BYTES =
    (MAX_PACKED_BYTES > (MAX_FEATURES * sizeof(float))) ? MAX_PACKED_BYTES
                                                        : (MAX_FEATURES * sizeof(float));

struct RecEntry { uint16_t nfeat; uint8_t label; uint8_t data[RECENTRY_DATA_BYTES]; };
static RecEntry g_rec_pool[RECV_Q_CAP];
static QueueHandle_t g_free_queue = nullptr;
static QueueHandle_t g_ready_queue = nullptr;
static UBaseType_t g_queue_capacity = RECV_Q_CAP;
static volatile bool g_compute_busy = false;

struct ComputeBusyGuard {
  ComputeBusyGuard() { g_compute_busy = true; }
  ~ComputeBusyGuard() { g_compute_busy = false; }
};
 
// ---------------- Feature configuration (runtime) ----------------
struct FeatureConfig {
  bool input_real = false;          // If true, incoming rows are float32 real values
  bool multiclass_enabled = false;  // default binary classification for tree models
  bool dense_model = false;         // default sparse model
} g_feat_cfg;

 
// ---------------- Binary Logging Bridge ----------------
// Forward declaration - implementation after Transport TR is defined
extern "C" void utils_log_binary(const char* msg, uint16_t len);


// NOTE: Research analysis functions moved to analysis_helpers.h
// They now send structured binary data instead of formatted strings
// This saves memory and bandwidth on the ESP32
// fletcher16_update provided by protocol.h; IO via Transport
// HAL provides the Transport instance (TR) - see hal/hal_platform.h
extern Transport TR;
static volatile bool g_any_rx_seen = false;

#ifndef TM_INLINE_COMPUTE_FASTPATH
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#define TM_INLINE_COMPUTE_FASTPATH 1
#else
#define TM_INLINE_COMPUTE_FASTPATH 0
#endif
#endif

static inline bool use_inline_tm_compute_fastpath() {
#if TM_INLINE_COMPUTE_FASTPATH
  return (!g_feat_cfg.input_real) && (g_selected_model == RuntimeModelType::TM_BO);
#else
  return false;
#endif
}

// Binary logging bridge implementation
extern "C" void utils_log_binary(const char* msg, uint16_t len) {
#if TM_ENABLE_LOG_FRAMES
  if (msg && len) {
    TR.sendLog(msg, len);
  }
#else
  (void)msg;
  (void)len;
#endif
}

// Ensure all enqueued records are processed by the compute task
static void model_reset_runtime() {
  g_model = std::monostate{};
  preproc_reset_state(0);
  g_model_feature_count = 0;
  g_model_init_failed = false;
  g_model_memory_bytes = 0;
  G.model_initialized = false;
  G.feat = 0;
}

static const char* model_name(RuntimeModelType type) {
  switch (type) {
    case RuntimeModelType::EFDT: return "EFDT";
    case RuntimeModelType::HOEFFDING: return "HOEFFDING";
    case RuntimeModelType::HAT: return "HAT";
    case RuntimeModelType::SGT: return "SGT";
    case RuntimeModelType::BNN: return "BNN";
    case RuntimeModelType::TM_SPARSE: return "TM_SPARSE";
    case RuntimeModelType::TM_VANILLA: return "TM_VANILLA";
    case RuntimeModelType::TM_BO: return "TM_BO";
    default: return "UNKNOWN";
  }
}

static bool model_from_id(uint8_t raw, RuntimeModelType& out_type) {
  switch (raw) {
    case MODEL_EFDT: out_type = RuntimeModelType::EFDT; return true;
    case MODEL_HOEFFDING: out_type = RuntimeModelType::HOEFFDING; return true;
    case MODEL_HAT: out_type = RuntimeModelType::HAT; return true;
    case MODEL_SGT: out_type = RuntimeModelType::SGT; return true;
    case MODEL_BNN: out_type = RuntimeModelType::BNN; return true;
    case MODEL_TM_SPARSE: out_type = RuntimeModelType::TM_SPARSE; return true;
    case MODEL_TM_VANILLA: out_type = RuntimeModelType::TM_VANILLA; return true;
    case MODEL_TM_BO: out_type = RuntimeModelType::TM_BO; return true;
    default: return false;
  }
}

static const char* preproc_name(OnlinePreproc mode) {
  switch (mode) {
    case OnlinePreproc::NONE: return "none";
    case OnlinePreproc::STANDARDIZE: return "standardize";
    case OnlinePreproc::TWINE: return "twine";
    default: return "unknown";
  }
}

static bool preproc_from_id(uint16_t raw, OnlinePreproc& out_mode) {
  switch (raw) {
    case PREPROC_NONE: out_mode = OnlinePreproc::NONE; return true;
    case PREPROC_ONLINE_STANDARDIZE: out_mode = OnlinePreproc::STANDARDIZE; return true;
    case PREPROC_TWINE: out_mode = OnlinePreproc::TWINE; return true;
    default: return false;
  }
}

static uint8_t preproc_twine_bits() {
  uint8_t bits = g_preproc_cfg.param0;
  if (bits < 1u) bits = 1u;
  if (bits > 8u) bits = 8u;
  return bits;
}

static void preproc_reset_state(int nfeat) {
  g_standardize_stats.clear();
  g_twine_stats.clear();
  if (nfeat <= 0) {
    return;
  }

  switch (g_preproc_cfg.mode) {
    case OnlinePreproc::STANDARDIZE:
      g_standardize_stats.assign(static_cast<std::size_t>(nfeat), FeatureStats{});
      break;
    case OnlinePreproc::TWINE:
      g_twine_stats.assign(static_cast<std::size_t>(nfeat), TwineFeatureStats{});
      break;
    case OnlinePreproc::NONE:
    default:
      break;
  }
}

static bool model_uses_tm_packed_path(RuntimeModelType type) {
  return type == RuntimeModelType::TM_SPARSE ||
         type == RuntimeModelType::TM_VANILLA ||
         type == RuntimeModelType::TM_BO;
}

static uint32_t model_selected_num_classes() {
  switch (g_selected_model) {
    case RuntimeModelType::EFDT: return static_cast<uint32_t>(g_cfg_efdt.num_classes);
    case RuntimeModelType::HOEFFDING: return static_cast<uint32_t>(g_cfg_hoeffding.num_classes);
    case RuntimeModelType::HAT: return static_cast<uint32_t>(g_cfg_hat.num_classes);
    case RuntimeModelType::SGT: return static_cast<uint32_t>(g_cfg_sgt.num_classes);
    case RuntimeModelType::BNN: return static_cast<uint32_t>(g_cfg_bnn.num_classes);
    case RuntimeModelType::TM_SPARSE: return static_cast<uint32_t>(g_cfg_tm_sparse.num_classes);
    case RuntimeModelType::TM_VANILLA: return static_cast<uint32_t>(g_cfg_tm_vanilla.num_classes);
    case RuntimeModelType::TM_BO: return static_cast<uint32_t>(g_cfg_tm_bo.num_classes);
    default: return 2u;
  }
}

static bool model_ensure_model(int nfeat) {
  if (nfeat <= 0 || nfeat > MAX_FEATURES) {
    return false;
  }
  if (g_model_init_failed) {
    return false;
  }

  if (!std::holds_alternative<std::monostate>(g_model)) {
    if (g_model_feature_count == nfeat) {
      return G.model_initialized;
    }
    // Keep feature dimension fixed within a run; mismatches are treated as bad input.
    // Reinitializing here can reset counters mid-phase and corrupt throughput stats.
    static uint32_t s_feat_mismatch_logs = 0;
    if (s_feat_mismatch_logs < 3) {
      LOG_SERIAL.printf("MODEL DIM mismatch: got=%d expected=%d (sample dropped)\n",
                        nfeat, g_model_feature_count);
      s_feat_mismatch_logs++;
    }
    return false;
  }

  try {
    // Capture memory before model creation
    uint32_t psram_before = ESP.getFreePsram();
    uint32_t heap_before = ESP.getFreeHeap();

    uint32_t num_classes_for_log = 2;
    switch (g_selected_model) {
      case RuntimeModelType::EFDT: {
        tree::EfdtConfig cfg = g_cfg_efdt;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tree::EfdtModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::HOEFFDING: {
        tree::HoeffdingTreeConfig cfg = g_cfg_hoeffding;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        if (nfeat >= 192) {
          cfg.num_threshold_bins = std::min<std::size_t>(cfg.num_threshold_bins, 4u);
          cfg.max_depth = std::min<std::size_t>(cfg.max_depth, 6u);
          cfg.grace_period = std::max<std::size_t>(cfg.grace_period, 256u);
          cfg.min_samples_split = std::max<std::size_t>(cfg.min_samples_split, 512u);
        }
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tree::HoeffdingTreeModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::HAT: {
        tree::HatConfig cfg = g_cfg_hat;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tree::HatModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::SGT: {
        tree::SgtConfig cfg = g_cfg_sgt;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tree::SgtModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::BNN: {
        BnnRuntimeConfig cfg = g_cfg_bnn;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        if (cfg.hidden_layers.empty()) {
          cfg.hidden_layers = std::vector<std::size_t> BNN_HIDDEN_LAYERS;
        }
        bnn::TrainingConfig train_cfg{};
        train_cfg.learning_rate = cfg.learning_rate;
        train_cfg.weight_decay = cfg.weight_decay;
        train_cfg.clip_value = cfg.clip_value;
        train_cfg.seed = cfg.seed;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<bnn::BinaryNeuralNetwork>(
            static_cast<std::size_t>(nfeat),
            cfg.hidden_layers,
            cfg.num_classes,
            train_cfg);
        break;
      }
      case RuntimeModelType::TM_SPARSE: {
        tm_model::TmSparseConfig cfg = g_cfg_tm_sparse;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tm_model::TmSparseModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::TM_VANILLA: {
        tm_model::TmVanillaConfig cfg = g_cfg_tm_vanilla;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tm_model::TmVanillaModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      case RuntimeModelType::TM_BO: {
        tm_model::TmBoConfig cfg = g_cfg_tm_bo;
        if (cfg.num_classes < 2) cfg.num_classes = 2;
        num_classes_for_log = static_cast<uint32_t>(cfg.num_classes);
        g_model.emplace<tm_model::TmBoModel>(static_cast<std::size_t>(nfeat), cfg);
        break;
      }
      default:
        return false;
    }
    preproc_reset_state(nfeat);
    g_model_feature_count = nfeat;
    G.model_initialized = true;
    G.feat = nfeat;
    G.trained = G.tested = G.correct = 0;
    G.train_ok = 0;
    G.last_score = 0;
    G.last_pred = 0;
    G.ema_acc = 0.0f;
    G.ema_score = 0.0f;
    G.score_us_accum = 0;
    G.score_calls = 0;
    G.update_us_accum = 0;
    G.update_calls = 0;

    // Calculate actual memory usage
    uint32_t psram_after = ESP.getFreePsram();
    uint32_t heap_after = ESP.getFreeHeap();
    int32_t psram_used = psram_before - psram_after;
    int32_t heap_used = heap_before - heap_after;

    if (psram_used < 0) psram_used = 0;
    if (heap_used < 0) heap_used = 0;

    g_model_memory_bytes = static_cast<uint32_t>(psram_used + heap_used);

    bounds_record_memory_breakdown(g_model_memory_bytes, 0, 0);
    bounds_record_model_size(g_model_memory_bytes, 0);

    LOG_SERIAL.println("=== Model Initialized ===");
    LOG_SERIAL.printf("  Model: %s\n", model_name(g_selected_model));
    LOG_SERIAL.printf("  Input Features: %d\n", nfeat);
    LOG_SERIAL.printf("  Output Classes: %u\n", (unsigned)num_classes_for_log);
    LOG_SERIAL.printf("Memory Footprint:\n");
    LOG_SERIAL.printf("  Actual PSRAM Used: %u bytes (%.2f KB)\n", (unsigned)psram_used, (float)psram_used / 1024.0f);
    LOG_SERIAL.printf("  Actual Heap Used:  %u bytes (%.2f KB)\n", (unsigned)heap_used, (float)heap_used / 1024.0f);
    LOG_SERIAL.printf("  ACTUAL TOTAL:      %u bytes (%.2f KB)\n", (unsigned)g_model_memory_bytes,
                      (float)g_model_memory_bytes / 1024.0f);
    uint32_t psram_total = ESP.getPsramSize();
    uint32_t psram_remaining = ESP.getFreePsram();
    uint32_t psram_total_used = psram_total - psram_remaining;
    if (psram_total > 0) {
      LOG_SERIAL.printf("PSRAM Status:\n");
      LOG_SERIAL.printf("  Total: %u bytes (%.2f MB)\n", psram_total, (float)psram_total / 1048576.0f);
      LOG_SERIAL.printf("  Used:  %u bytes (%.2f KB) [%.1f%%]\n",
                        (unsigned)psram_total_used, (float)psram_total_used / 1024.0f,
                        (float)psram_total_used * 100.0f / (float)psram_total);
      LOG_SERIAL.printf("  Free:  %u bytes (%.2f KB)\n", psram_remaining, (float)psram_remaining / 1024.0f);
    }
    LOG_SERIAL.println("==========================");

    return true;
  } catch (const std::exception& ex) {
    LOG_SERIAL.printf("MODEL INIT FAILED: nfeat=%d error=%s\n", nfeat, ex.what());
    g_model = std::monostate{};
    g_model_init_failed = true;
    G.model_initialized = false;
    return false;
  }
}

static void model_apply_runtime_defaults() {
  g_preproc_cfg.mode = OnlinePreproc::NONE;
  g_preproc_cfg.param0 = 0;

  g_cfg_efdt = tree::EfdtConfig{};
  g_cfg_efdt.num_classes = 2;

  g_cfg_hoeffding = tree::HoeffdingTreeConfig{};
  g_cfg_hoeffding.num_classes = 2;
  g_cfg_hoeffding.num_threshold_bins = HT_NUM_THRESHOLD_BINS;
  g_cfg_hoeffding.delta = HT_DELTA;
  g_cfg_hoeffding.tie_threshold = HT_TIE_THRESHOLD;
  g_cfg_hoeffding.grace_period = HT_GRACE_PERIOD;
  g_cfg_hoeffding.min_samples_split = HT_MIN_SAMPLES_SPLIT;
  g_cfg_hoeffding.max_depth = HT_MAX_DEPTH;

  g_cfg_hat = tree::HatConfig{};
  g_cfg_hat.num_classes = 2;

  g_cfg_sgt = tree::SgtConfig{};
  g_cfg_sgt.num_classes = 2;

  g_cfg_bnn = BnnRuntimeConfig{};
  g_cfg_bnn.num_classes = BNN_NUM_CLASSES;
  g_cfg_bnn.hidden_layers = std::vector<std::size_t> BNN_HIDDEN_LAYERS;
  g_cfg_bnn.learning_rate = BNN_LEARNING_RATE;
  g_cfg_bnn.weight_decay = BNN_WEIGHT_DECAY;
  g_cfg_bnn.clip_value = BNN_CLIP_VALUE;
  g_cfg_bnn.seed = 0u;

  g_cfg_tm_sparse = tm_model::TmSparseConfig{};
  g_cfg_tm_sparse.num_classes = TM_NUM_CLASSES;
  g_cfg_tm_sparse.clauses = TM_C;
  g_cfg_tm_sparse.threshold = TM_T;
  g_cfg_tm_sparse.specificity = TM_S;
  g_cfg_tm_sparse.seed = TM_SEED;
  g_cfg_tm_sparse.init_literal_density_pct = TM_INIT_LITERAL_DENSITY_PCT;

  g_cfg_tm_vanilla = tm_model::TmVanillaConfig{};
  g_cfg_tm_vanilla.num_classes = TM_NUM_CLASSES;
  g_cfg_tm_vanilla.clauses = TM_C;
  g_cfg_tm_vanilla.threshold = TM_T;
  g_cfg_tm_vanilla.specificity = TM_S;
  g_cfg_tm_vanilla.seed = TM_SEED;
  g_cfg_tm_vanilla.init_literal_density_pct = TM_INIT_LITERAL_DENSITY_PCT;

  g_cfg_tm_bo = tm_model::TmBoConfig{};
  g_cfg_tm_bo.num_classes = TM_NUM_CLASSES;
  g_cfg_tm_bo.clauses = TM_C;
  g_cfg_tm_bo.threshold = TM_T;
  g_cfg_tm_bo.specificity = TM_S;
  g_cfg_tm_bo.seed = TM_SEED;
  g_cfg_tm_bo.init_literal_density_pct = TM_INIT_LITERAL_DENSITY_PCT;
}

static float model_train_sample_dispatch(const bnn::Sample& sample, int* pred_before_out) {
  switch (g_selected_model) {
    case RuntimeModelType::EFDT: {
      auto* m = std::get_if<tree::EfdtModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::HOEFFDING: {
      auto* m = std::get_if<tree::HoeffdingTreeModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::HAT: {
      auto* m = std::get_if<tree::HatModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::SGT: {
      auto* m = std::get_if<tree::SgtModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::BNN: {
      auto* m = std::get_if<bnn::BinaryNeuralNetwork>(&g_model);
      if (!m) return 0.0f;
      if (pred_before_out) {
        *pred_before_out = m->predict(sample.features);
      }
      return m->train_sample(sample);
    }
    case RuntimeModelType::TM_SPARSE: {
      auto* m = std::get_if<tm_model::TmSparseModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::TM_VANILLA: {
      auto* m = std::get_if<tm_model::TmVanillaModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::TM_BO: {
      auto* m = std::get_if<tm_model::TmBoModel>(&g_model);
      return m ? m->train_sample(sample, pred_before_out) : 0.0f;
    }
    default:
      return 0.0f;
  }
}

static int model_predict_dispatch(std::vector<float>& features) {
  switch (g_selected_model) {
    case RuntimeModelType::EFDT: {
      auto* m = std::get_if<tree::EfdtModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::HOEFFDING: {
      auto* m = std::get_if<tree::HoeffdingTreeModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::HAT: {
      auto* m = std::get_if<tree::HatModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::SGT: {
      auto* m = std::get_if<tree::SgtModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::BNN: {
      auto* m = std::get_if<bnn::BinaryNeuralNetwork>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::TM_SPARSE: {
      auto* m = std::get_if<tm_model::TmSparseModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::TM_VANILLA: {
      auto* m = std::get_if<tm_model::TmVanillaModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    case RuntimeModelType::TM_BO: {
      auto* m = std::get_if<tm_model::TmBoModel>(&g_model);
      return m ? m->predict(features) : 0;
    }
    default:
      return 0;
  }
}

static float model_train_packed_dispatch(const uint8_t* packed_bits, std::size_t nfeat, int32_t label, int* pred_before_out) {
  switch (g_selected_model) {
    case RuntimeModelType::TM_SPARSE: {
      auto* m = std::get_if<tm_model::TmSparseModel>(&g_model);
      return m ? m->train_packed_bits(packed_bits, nfeat, label, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::TM_VANILLA: {
      auto* m = std::get_if<tm_model::TmVanillaModel>(&g_model);
      return m ? m->train_packed_bits(packed_bits, nfeat, label, pred_before_out) : 0.0f;
    }
    case RuntimeModelType::TM_BO: {
      auto* m = std::get_if<tm_model::TmBoModel>(&g_model);
      return m ? m->train_packed_bits(packed_bits, nfeat, label, pred_before_out) : 0.0f;
    }
    default:
      return 0.0f;
  }
}

static int model_predict_packed_dispatch(const uint8_t* packed_bits, std::size_t nfeat) {
  switch (g_selected_model) {
    case RuntimeModelType::TM_SPARSE: {
      auto* m = std::get_if<tm_model::TmSparseModel>(&g_model);
      return m ? m->predict_packed_bits(packed_bits, nfeat) : 0;
    }
    case RuntimeModelType::TM_VANILLA: {
      auto* m = std::get_if<tm_model::TmVanillaModel>(&g_model);
      return m ? m->predict_packed_bits(packed_bits, nfeat) : 0;
    }
    case RuntimeModelType::TM_BO: {
      auto* m = std::get_if<tm_model::TmBoModel>(&g_model);
      return m ? m->predict_packed_bits(packed_bits, nfeat) : 0;
    }
    default:
      return 0;
  }
}

static void model_convert_bits_to_features(const uint8_t* packed_bits, int nfeat, float* out_features) {
  int out_idx = 0;
  const int full_bytes = nfeat >> 3;
  for (int i = 0; i < full_bytes; ++i) {
    const uint8_t b = packed_bits[i];
    out_features[out_idx++] = (b & 0x80) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x40) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x20) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x10) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x08) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x04) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x02) ? 1.0f : -1.0f;
    out_features[out_idx++] = (b & 0x01) ? 1.0f : -1.0f;
  }
  const int rem = nfeat & 7;
  if (rem) {
    const uint8_t b = packed_bits[full_bytes];
    if (rem > 0) out_features[out_idx++] = (b & 0x80) ? 1.0f : -1.0f;
    if (rem > 1) out_features[out_idx++] = (b & 0x40) ? 1.0f : -1.0f;
    if (rem > 2) out_features[out_idx++] = (b & 0x20) ? 1.0f : -1.0f;
    if (rem > 3) out_features[out_idx++] = (b & 0x10) ? 1.0f : -1.0f;
    if (rem > 4) out_features[out_idx++] = (b & 0x08) ? 1.0f : -1.0f;
    if (rem > 5) out_features[out_idx++] = (b & 0x04) ? 1.0f : -1.0f;
    if (rem > 6) out_features[out_idx++] = (b & 0x02) ? 1.0f : -1.0f;
  }
}

static void model_load_raw_features(const uint8_t* raw_bytes, int nfeat, float* out_features) {
  std::memcpy(out_features, raw_bytes, static_cast<std::size_t>(nfeat) * sizeof(float));
}

static void model_preprocess_features(float* features, int nfeat, bool is_training, bool input_real) {
  if (!input_real || nfeat <= 0) {
    return;
  }

  switch (g_preproc_cfg.mode) {
    case OnlinePreproc::STANDARDIZE: {
      if ((int)g_standardize_stats.size() < nfeat) {
        preproc_reset_state(nfeat);
      }
      if ((int)g_standardize_stats.size() < nfeat) {
        return;
      }
      for (int i = 0; i < nfeat; ++i) {
        const float value = features[i];
        FeatureStats& stats = g_standardize_stats[static_cast<std::size_t>(i)];

        // Update running mean/variance during training only.
        if (is_training) {
          stats.count++;
          const float delta = value - stats.mean;
          stats.mean += delta / static_cast<float>(stats.count);
          const float delta2 = value - stats.mean;
          stats.m2 += delta * delta2;
        }

        // Apply z-score transform when variance is known.
        if (stats.count > 1u) {
          const float variance = stats.m2 / static_cast<float>(stats.count - 1u);
          const float std_dev = std::sqrt(std::max(0.0f, variance));
          features[i] = (std_dev > 1e-8f) ? ((value - stats.mean) / std_dev) : (value - stats.mean);
        }
      }
      break;
    }
    case OnlinePreproc::TWINE: {
      if ((int)g_twine_stats.size() < nfeat) {
        preproc_reset_state(nfeat);
      }
      if ((int)g_twine_stats.size() < nfeat) {
        return;
      }
      const uint8_t bits = preproc_twine_bits();
      const float levels_f = static_cast<float>((1u << bits) - 1u);
      if (levels_f <= 0.0f) {
        return;
      }

      for (int i = 0; i < nfeat; ++i) {
        const float value = features[i];
        TwineFeatureStats& stats = g_twine_stats[static_cast<std::size_t>(i)];

        // Track feature range online during training.
        if (is_training) {
          if (stats.count == 0u) {
            stats.min_value = value;
            stats.max_value = value;
            stats.count = 1u;
          } else {
            if (value < stats.min_value) stats.min_value = value;
            if (value > stats.max_value) stats.max_value = value;
            stats.count++;
          }
        }

        if (stats.count == 0u) {
          continue;
        }
        const float range = stats.max_value - stats.min_value;
        if (range <= 1e-8f) {
          features[i] = 0.0f;
          continue;
        }

        float norm = (value - stats.min_value) / range;
        if (norm < 0.0f) norm = 0.0f;
        if (norm > 1.0f) norm = 1.0f;
        const uint32_t q = static_cast<uint32_t>(norm * levels_f + 0.5f);
        const float q01 = static_cast<float>(q) / levels_f;
        // Keep output centered for downstream models.
        features[i] = q01 * 2.0f - 1.0f;
      }
      break;
    }
    case OnlinePreproc::NONE:
    default:
      break;
  }
}

static void model_on_train_sample(int pred, int label, float loss) {
  if (G.trained == 0) {
    bounds_record_training_start();
  }
  G.last_pred = static_cast<uint8_t>(pred);
  G.last_score = static_cast<int32_t>(loss * 1000.0f);
  if (pred == label) {
    G.train_ok++;
  }
  G.trained++;
  const float a = G.ema_alpha;
  G.ema_acc = (1.0f - a) * G.ema_acc + a * (pred == label ? 1.0f : 0.0f);
  G.ema_score = (1.0f - a) * G.ema_score + a * loss;
  bounds_record_classification(pred, label);
}

static void model_on_test_sample(int pred, int label) {
  if (G.tested == 0) {
    bounds_record_testing_start();
  }
  G.tested++;
  if (pred == label) {
    G.correct++;
  }
  bounds_record_classification(pred, label);
}

// Ensure all enqueued records are processed by the compute task
static inline void wait_for_queue_empty(){
  // Poll until RX queue is empty and compute task finishes current sample
  // Increase timeout so larger clause counts have time to complete
  const unsigned long timeout_ms = 10000;
  unsigned long start = millis();
  while (g_ready_queue && (uxQueueMessagesWaiting(g_ready_queue) > 0 || g_compute_busy)){
    if (millis() - start > timeout_ms) {
      break;  // Timeout - queue will drain during next phase
    }
    vTaskDelay(1);  // Small delay to allow other tasks to run
  }
}


#if ENABLE_DISPLAY
// ---------------- Display Functions ----------------
static void display_init() {
  // Initialize backlight first
  pinMode(TFT_BACKLITE, OUTPUT);
  digitalWrite(TFT_BACKLITE, HIGH);
  delay(200);
  
  // Initialize TFT display
  tft.init(135, 240);
  delay(100);
  
  tft.setRotation(3);
  delay(50);
  
  tft.fillScreen(COLOR_BG);
  delay(100);
  
  // Draw title
  tft.setTextSize(2);
  tft.setTextColor(COLOR_TITLE);
  tft.setCursor(10, 10);
  tft.println("TM Binary");
  
  // Show initial RAM
  tft.setTextSize(1);
  tft.setTextColor(COLOR_INFO);
  tft.setCursor(10, 30);
  tft.printf("RAM: %d bytes", ESP.getFreeHeap());
  
  tft.drawFastHLine(0, 50, 240, COLOR_INFO);
  delay(100);
}

static void display_status(const char* msg, uint16_t color = COLOR_INFO) {
  tft.fillRect(0, 60, 240, 20, COLOR_BG);
  tft.setTextSize(1);
  tft.setTextColor(color);
  tft.setCursor(10, 60);
  tft.println(msg);
}

static void display_progress(const char* mode, uint32_t count, float accuracy = -1.0f) {
  tft.fillRect(0, 80, 240, 40, COLOR_BG);
  tft.setTextSize(1);
  tft.setTextColor(COLOR_INFO);
  tft.setCursor(10, 80);
  tft.printf("%s: %d samples", mode, count);
  
  if (accuracy >= 0.0f) {
    tft.setTextColor(COLOR_SUCCESS);
    tft.setCursor(10, 95);
    tft.printf("Accuracy: %.2f%%", accuracy * 100);
  }
  
  // NOTE: Power display disabled (was showing fake estimated values)
  // tft.setTextColor(COLOR_WARNING);
  // tft.setCursor(10, 110);
  // tft.printf("Power: %.1f mW", G.avg_power_mw);
}

static void display_final_results(uint32_t trained, uint32_t tested, uint32_t correct) {
  float accuracy = (tested > 0) ? (float)correct / tested : 0.0f;
  
  tft.fillRect(0, 60, 240, 90, COLOR_BG);
  tft.setTextSize(2);
  tft.setTextColor(COLOR_SUCCESS);
  tft.setCursor(10, 70);
  tft.printf("%.2f%%", accuracy * 100);
  
  tft.setTextSize(1);
  tft.setTextColor(COLOR_TITLE);
  tft.setCursor(10, 95);
  tft.printf("Tested: %d", tested);
  
  tft.setCursor(10, 110);
  tft.setTextColor(COLOR_INFO);
  tft.printf("Correct: %d/%d", correct, tested);
  
  tft.setCursor(10, 125);
  tft.setTextColor(COLOR_WARNING);
  tft.printf("Trained: %d", trained);
  
  // NOTE: Power display disabled (was showing fake estimated values)
  // tft.setCursor(10, 140);
  // tft.setTextColor(COLOR_WARNING);
  // tft.printf("Avg Power: %.1f mW", G.avg_power_mw);
}
#endif
 
 // Read payload body and optionally verify Fletcher-16 checksum trailer
 static bool read_payload_buf(uint8_t* dst, uint16_t n, bool with_chk, uint16_t& chk, uint32_t seq){
   if (n && !TR.readExact(dst, n, 1000)) return false;
   if (with_chk) chk = fletcher16_update((uint16_t)chk, dst, n);
   if (with_chk){
     uint16_t rx=0; if (!TR.readExact((uint8_t*)&rx, 2, 300)) return false; // FIX: more relaxed timeout
     if ((uint16_t)rx!=(uint16_t)chk){ TR.sendError(ERR_CHECKSUM, seq); return false; }
   }
   return true;
 }
 
 
static void handle_entry(const RecEntry& e){
  ComputeBusyGuard busy_guard;

  if (!model_ensure_model(static_cast<int>(e.nfeat))) {
    if (G.mode == DeviceState::TRAIN) {
      G.trained++;
    } else {
      G.tested++;
    }
    return;
  }

  if (!g_feat_cfg.input_real && model_uses_tm_packed_path(g_selected_model)) {
    const int label = static_cast<int>(e.label);
    if (G.mode == DeviceState::TRAIN) {
      int pred = 0;
      const float loss = model_train_packed_dispatch(
          e.data, static_cast<std::size_t>(e.nfeat), static_cast<int32_t>(label), &pred);
      model_on_train_sample(pred, label, loss);
    } else {
      const int pred = model_predict_packed_dispatch(e.data, static_cast<std::size_t>(e.nfeat));
      model_on_test_sample(pred, label);
    }

    static uint32_t entry_count = 0;
    if ((++entry_count % 5000) == 0) {
      BOUNDS_RECORD_HEAP();
      BOUNDS_RECORD_STACK();
    }
    return;
  }

  static bnn::Sample sample;
  if (sample.features.size() != e.nfeat) {
    sample.features.resize(e.nfeat);
  }
  float* features = sample.features.data();

  if (g_feat_cfg.input_real) {
    model_load_raw_features(e.data, static_cast<int>(e.nfeat), features);
    model_preprocess_features(features, static_cast<int>(e.nfeat), G.mode == DeviceState::TRAIN, true);
  } else {
    model_convert_bits_to_features(e.data, static_cast<int>(e.nfeat), features);
  }
  sample.label = static_cast<int32_t>(e.label);

  if (G.mode == DeviceState::TRAIN) {
    int pred = 0;
    const float loss = model_train_sample_dispatch(sample, &pred);
    model_on_train_sample(pred, static_cast<int>(sample.label), loss);
  } else {
    const int pred = model_predict_dispatch(sample.features);
    model_on_test_sample(pred, static_cast<int>(sample.label));
  }

  // Record heap/stack samples periodically for parity with TM path
  static uint32_t entry_count = 0;
  if ((++entry_count % 5000) == 0) {
    BOUNDS_RECORD_HEAP();
    BOUNDS_RECORD_STACK();
  }
}
 
 // ---------------- Transport task ----------------
static void transport_task(void* pv){
  LOG_SERIAL.println("transport_task: start");
  TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
  LOG_SERIAL.println("transport_task: READY sent");
 
   WindowState W; W.sent=0; W.last_seq=0; W.last_acked=0; W.cap=G.window?G.window:1; W.left=W.cap;
   static uint8_t packed[MAX_PACKED_BYTES];
 
  unsigned long last_rx_ms = millis();
  const unsigned long READ_TRY_MS = 20;  // Slightly higher to improve robustness at larger windows
  const unsigned long IDLE_DONE_MS = 50;   // Faster DONE beacons when idle
  const unsigned long DONE_BEACON_MS = 1000; // FIX: beacon DONE when idle, do not exit
  const unsigned long READY_BEACON_MS = 2000; // Send READY after being idle
  unsigned long last_beacon_ms = millis();
  unsigned long last_done_beacon_ms = 0;
  unsigned long last_ready_beacon_ms = 0;
 
   for(;;){
     int ingested = 0;
    
    for(;;){
 
 
       FrameHeader h; bool with_chk=false; uint16_t chk=0;
       bool got = TR.readHeader(h, with_chk, chk, READ_TRY_MS);
       if (!got) break;
       g_any_rx_seen = true;
       last_rx_ms = millis();
       W.last_seq = h.seq;
       const uint16_t plen = h.len;
 
       if (h.type == FRAME_TYPE_CMD){
         // Reduced logging for better throughput
         // logf("transport_task: CMD len=%u\n", (unsigned)plen);
        if (plen==0 || plen>64){
          uint8_t junk[64]; uint16_t left=plen;
          while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; }
          if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150); }
          bounds_record_frame_validation(false, 1); // size error
          continue;
        }
        uint8_t cmd[64]; 
        if (!read_payload_buf(cmd, plen, with_chk, chk, h.seq)) {
          bounds_record_frame_validation(false, 2); // checksum error
          continue;
        }
        bounds_record_frame_validation(true, 0); // valid CMD frame
         uint8_t op = cmd[0];
         // Reduced logging for better throughput
         // logf("transport_task: OPC=%c (%u)\n", (int)op, (unsigned)op);
 
        if      (op==OPC_STATUS)   { float acc = (G.tested>0) ? (float)G.correct / (float)G.tested : 0.f; TR.sendStat(h.seq, G.trained, G.tested, acc); }
        else if (op==OPC_CONFIG)   {
          // cmd[0] is the opcode; payload starts at cmd+1
          if (plen >= 1 + (int)sizeof(ConfigPayload)){
            ConfigPayload cp{}; memcpy(&cp, cmd + 1, sizeof(ConfigPayload));
            ConfigPayloadV2 cp_v2{};
            const bool has_cfg_v2 = plen >= 1 + (int)sizeof(ConfigPayloadV2);
            if (has_cfg_v2) {
              memcpy(&cp_v2, cmd + 1, sizeof(ConfigPayloadV2));
            }

            const bool twine_requested = (cp.flags & CFG_TWINE) != 0;
            g_feat_cfg.input_real    = (cp.flags & CFG_INPUT_REAL) != 0;
            g_feat_cfg.multiclass_enabled = false; // BinaryNet currently single-label
            g_feat_cfg.dense_model        = (cp.flags & CFG_MODEL_DENSE) != 0;

            OnlinePreproc requested_preproc = OnlinePreproc::NONE;
            if (!preproc_from_id(cp.preproc_id, requested_preproc)) {
              LOG_SERIAL.printf("CONFIG: unknown preproc id=%u, falling back to none.\n",
                                (unsigned)cp.preproc_id);
              requested_preproc = OnlinePreproc::NONE;
            }
            // Legacy compatibility: old hosts can still request TWINE via CFG_TWINE flag.
            if (requested_preproc == OnlinePreproc::NONE && twine_requested) {
              requested_preproc = OnlinePreproc::TWINE;
            }

            uint8_t requested_param0 = cp.preproc_param0;
            if (requested_preproc == OnlinePreproc::TWINE) {
              if (requested_param0 < 1u) requested_param0 = 1u;
              if (requested_param0 > 8u) requested_param0 = 8u;
            } else {
              requested_param0 = 0u;
            }

            const bool preproc_changed =
                (g_preproc_cfg.mode != requested_preproc) || (g_preproc_cfg.param0 != requested_param0);
            g_preproc_cfg.mode = requested_preproc;
            g_preproc_cfg.param0 = requested_param0;
            if (preproc_changed) {
              preproc_reset_state(g_model_feature_count);
            }

            if (!g_feat_cfg.input_real && g_preproc_cfg.mode != OnlinePreproc::NONE) {
              LOG_SERIAL.println("CONFIG: preprocessor active only for input_real samples; packed path bypasses it.");
            }

            const bool tm_runtime_present =
                has_cfg_v2 && ((cp_v2.flags & CFG_TM_RUNTIME) != 0);
            bool tm_cfg_changed = false;
            if (tm_runtime_present) {
              std::size_t new_clauses = (std::size_t)cp_v2.tm_clauses;
              if (new_clauses < 2u) new_clauses = 2u;
              int new_threshold = (int)cp_v2.tm_threshold;
              if (new_threshold < 1) new_threshold = 1;
              int new_specificity = (int)cp_v2.tm_specificity;
              if (new_specificity < 2) new_specificity = 2;
              int new_init_density = (int)cp_v2.tm_init_density_pct;
              if (new_init_density < 0) new_init_density = 0;
              if (new_init_density > 100) new_init_density = 100;
              const uint32_t new_seed = (uint32_t)cp_v2.tm_seed;

              const bool sparse_changed =
                  (g_cfg_tm_sparse.clauses != new_clauses) ||
                  (g_cfg_tm_sparse.threshold != new_threshold) ||
                  (g_cfg_tm_sparse.specificity != new_specificity) ||
                  (g_cfg_tm_sparse.seed != new_seed) ||
                  (g_cfg_tm_sparse.init_literal_density_pct != new_init_density);
              const bool vanilla_changed =
                  (g_cfg_tm_vanilla.clauses != new_clauses) ||
                  (g_cfg_tm_vanilla.threshold != new_threshold) ||
                  (g_cfg_tm_vanilla.specificity != new_specificity) ||
                  (g_cfg_tm_vanilla.seed != new_seed) ||
                  (g_cfg_tm_vanilla.init_literal_density_pct != new_init_density);
              const bool bo_changed =
                  (g_cfg_tm_bo.clauses != new_clauses) ||
                  (g_cfg_tm_bo.threshold != new_threshold) ||
                  (g_cfg_tm_bo.specificity != new_specificity) ||
                  (g_cfg_tm_bo.seed != new_seed) ||
                  (g_cfg_tm_bo.init_literal_density_pct != new_init_density);

              g_cfg_tm_sparse.clauses = new_clauses;
              g_cfg_tm_sparse.threshold = new_threshold;
              g_cfg_tm_sparse.specificity = new_specificity;
              g_cfg_tm_sparse.seed = new_seed;
              g_cfg_tm_sparse.init_literal_density_pct = new_init_density;

              g_cfg_tm_vanilla.clauses = new_clauses;
              g_cfg_tm_vanilla.threshold = new_threshold;
              g_cfg_tm_vanilla.specificity = new_specificity;
              g_cfg_tm_vanilla.seed = new_seed;
              g_cfg_tm_vanilla.init_literal_density_pct = new_init_density;

              g_cfg_tm_bo.clauses = new_clauses;
              g_cfg_tm_bo.threshold = new_threshold;
              g_cfg_tm_bo.specificity = new_specificity;
              g_cfg_tm_bo.seed = new_seed;
              g_cfg_tm_bo.init_literal_density_pct = new_init_density;

              tm_cfg_changed = sparse_changed || vanilla_changed || bo_changed;
              if (tm_cfg_changed && model_uses_tm_packed_path(g_selected_model)) {
                wait_for_queue_empty();
                model_reset_runtime();
                G.mode = DeviceState::TRAIN;
                G.trained = 0;
                G.tested = 0;
                G.correct = 0;
                G.train_ok = 0;
                G.last_score = 0;
                G.last_pred = 0;
                G.ema_acc = 0.0f;
                G.ema_score = 0.0f;
                G.score_us_accum = 0;
                G.score_calls = 0;
                G.update_us_accum = 0;
                G.update_calls = 0;
              }
            }

            LOG_SERIAL.printf("CONFIG: input_real=%d rep=%s algo=%s preproc=%s param0=%u tm_cfg=%d (MAX_FEATURES=%d)\n",
                              (int)g_feat_cfg.input_real,
                              g_feat_cfg.dense_model ? "dense" : "sparse",
                              model_name(g_selected_model),
                              preproc_name(g_preproc_cfg.mode),
                              (unsigned)g_preproc_cfg.param0,
                              (int)tm_runtime_present,
                              (int)MAX_FEATURES);
            if (tm_runtime_present) {
              LOG_SERIAL.printf("CONFIG_TM: clauses=%u threshold=%d specificity=%d seed=%u init_density=%d changed=%d\n",
                                (unsigned)g_cfg_tm_sparse.clauses,
                                (int)g_cfg_tm_sparse.threshold,
                                (int)g_cfg_tm_sparse.specificity,
                                (unsigned)g_cfg_tm_sparse.seed,
                                (int)g_cfg_tm_sparse.init_literal_density_pct,
                                (int)tm_cfg_changed);
            }
            // Send a READY beacon to acknowledge config applied
            TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
          }
        }
        else if (op==OPC_MODEL_SELECT) {
          if (plen < 2) {
            TR.sendError(ERR_BAD_LEN, h.seq);
            continue;
          }
          RuntimeModelType requested{};
          if (!model_from_id(cmd[1], requested)) {
            TR.sendError(ERR_BAD_BODY, h.seq);
            continue;
          }
          wait_for_queue_empty();
          g_selected_model = requested;
          model_reset_runtime();
          G.mode = DeviceState::TRAIN;
          G.trained = 0;
          G.tested = 0;
          G.correct = 0;
          G.train_ok = 0;
          G.last_score = 0;
          G.last_pred = 0;
          G.ema_acc = 0.0f;
          G.ema_score = 0.0f;
          G.score_us_accum = 0;
          G.score_calls = 0;
          G.update_us_accum = 0;
          G.update_calls = 0;
          bounds_monitor_reset();
          LOG_SERIAL.printf("MODEL_SELECT: %s (%u)\n", model_name(g_selected_model), (unsigned)cmd[1]);
          TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
        }
         else if (op==OPC_SNAPSHOT) { MetricPayload mp{ G.trained, G.train_ok, G.ema_acc, G.ema_score, G.last_score, G.last_pred, {0,0,0} }; TR.sendMetric(h.seq, mp); }
         else if (op==OPC_MEMPROF)  { float acc = (G.tested>0)?(float)G.correct/(float)G.tested:0.f; proto_send_memprof(TR, h.seq, G.trained, G.tested, acc, (uint32_t)ESP.getFreeHeap(), (uint32_t)ESP.getMinFreeHeap(), (uint32_t)ESP.getMaxAllocHeap(), utils_get_current_allocated(), utils_get_peak_allocated(), utils_get_allocation_count(), utils_get_free_count(), utils_get_active_allocations(), __profiler_entries__, __profiler_entry_count__); }
        else if (op==OPC_START_TRAIN){ 
          // Ensure any pending TEST records are processed before switching
          wait_for_queue_empty();
          G.mode=DeviceState::TRAIN; G.score_us_accum=0; G.score_calls=0; G.update_us_accum=0; G.update_calls=0; G.pending_memprof_on_test = true; 
#if ENABLE_DISPLAY
          display_status("Training Mode", COLOR_SUCCESS);
#endif
        }
        else if (op==OPC_START_TEST) { 
          // Before switching to TEST, ensure all queued TRAIN records are processed
          wait_for_queue_empty();
          
          // Record training end and model size after training
          bounds_record_training_end(G.trained);
          if (!std::holds_alternative<std::monostate>(g_model) && g_model_memory_bytes > 0) {
            bounds_record_model_size(static_cast<int>(g_model_memory_bytes), 1);  // After training
            bounds_record_model_size(static_cast<int>(g_model_memory_bytes), 2);  // Before testing
          }
          
          G.mode=DeviceState::TEST; G.tested=0; G.correct=0; 
#if ENABLE_DISPLAY
          display_status("Testing Mode", COLOR_WARNING);
#endif
          if (G.pending_memprof_on_test){ float acc=(G.tested>0)?(float)G.correct/(float)G.tested:0.f; proto_send_memprof(TR, h.seq, G.trained, G.tested, acc, (uint32_t)ESP.getFreeHeap(), (uint32_t)ESP.getMinFreeHeap(), (uint32_t)ESP.getMaxAllocHeap(), utils_get_current_allocated(), utils_get_peak_allocated(), utils_get_allocation_count(), utils_get_free_count(), utils_get_active_allocations(), __profiler_entries__, __profiler_entry_count__); G.pending_memprof_on_test=false; } 
        }
        else if (op==OPC_SYNC) {
          // Explicit drain request from host: reply DONE as soon as queue is empty
          wait_for_queue_empty();
          TR.sendDone(W.sent, W.last_seq);
        }
        else if (op==OPC_BUFFER && plen>=3){ uint16_t w = (uint16_t)(cmd[1] | (cmd[2]<<8)); if(!w) w=1; G.window = w; W.cap=w; W.left=w; }
        else if (op==OPC_SHOW_FINAL){ 
          // Send immediate confirmation
          float acc = (G.tested>0) ? (float)G.correct / (float)G.tested : 0.f; 
          TR.sendStat(h.seq, G.trained, G.tested, acc); 
          
          // Record testing end
          bounds_record_testing_end(G.tested);
          
          // Send comprehensive analysis as structured binary data
          // This saves memory and bandwidth vs formatted strings
          send_memory_efficiency_analysis(G.feat);
          send_performance_benchmarks(G.trained, G.tested, G.correct, G.feat);
          send_protocol_statistics();

#if TM_ENABLE_VERBOSE_FINAL_LOGS
          // Optional verbose diagnostics (disabled by default for max throughput).
          bounds_export_json();
          utils_print_memory_breakdown();
          utils_print_memory_entries();
          bounds_print_memory_breakdown();
#endif
          
#if ENABLE_DISPLAY
          display_final_results(G.trained, G.tested, G.correct);
#endif
          // Explicitly mark report completion.
          TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
        }
        else if (op==OPC_RESET_STATE){
          // Soft reset: return to READY state without hardware reboot
          LOG_SERIAL.println("=== OPC_RESET_STATE: Full reset for new experimental run ===");
          
          // Wait for queue to drain before resetting state
          wait_for_queue_empty();
          
          // Reset all counters and state
          G.trained = 0;
          G.tested = 0;
          G.correct = 0;
          G.train_ok = 0;
          G.last_score = 0;
          G.last_pred = 0;
          G.ema_acc = 0.0f;
          G.ema_score = 0.0f;
          G.score_us_accum = 0;
          G.score_calls = 0;
          G.update_us_accum = 0;
          G.update_calls = 0;
          G.mode = DeviceState::TRAIN;
          
          model_reset_runtime();
          
          // Reset bounds monitoring for new run
          bounds_monitor_reset();
          
          // Send READY frame with zeroed counts
          TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
          
          LOG_SERIAL.println("=== Device ready for next experimental run ===");
          LOG_SERIAL.flush();
        }
        else if (op==OPC_RESET){ 
          // Hardware reset: counters will be reset to 0 by global re-initialization
          LOG_SERIAL.flush(); ACTIVE_SERIAL.flush(); delay(200); ESP.restart(); 
        }
         continue;
       }
 
      if (h.type != FRAME_TYPE_RECORD &&
          h.type != FRAME_TYPE_SAMPLE_RAW &&
          h.type != FRAME_TYPE_RECORD_MC &&
          h.type != FRAME_TYPE_RECORD_BATCH){
         uint8_t junk[64]; uint16_t left=plen;
         while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; }
         if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150); }
         continue;
       }

      // High-throughput binary batch path: one frame can carry multiple packed records.
      // Current implementation is intentionally checksum-free for max speed; host only
      // emits this frame in unsafe mode.
      if (h.type == FRAME_TYPE_RECORD_BATCH){
        if (with_chk){
          uint8_t junk[64];
          uint16_t left = plen;
          while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200)) break; left-=c; }
          uint8_t c2[2]; TR.readExact(c2,2,150);
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_BAD_FRAME, h.seq);
          continue;
        }
        if (plen < sizeof(BatchHeader) || plen > 9000){
          uint8_t junk[64];
          uint16_t left = plen;
          while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200)) break; left-=c; }
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_BAD_FRAME, h.seq);
          continue;
        }

        BatchHeader bh{};
        if (!TR.readExact((uint8_t*)&bh, sizeof(bh), 1000)){
          bounds_record_frame_validation(false, 3);
          if(W.sent>W.last_acked){ TR.sendAck(W.sent,W.last_seq); W.last_acked=W.sent; }
          TR.sendError(ERR_BAD_HDR, h.seq);
          continue;
        }

        const uint16_t batch_count = bh.count_le;
        const int batch_nfeat = static_cast<int>(bh.nfeat_le);
        const int batch_need = (batch_nfeat + 7) / 8;
        const uint32_t body_len = static_cast<uint32_t>(plen - sizeof(BatchHeader));
        const uint32_t expected_len = static_cast<uint32_t>(batch_count) * static_cast<uint32_t>(1 + batch_need);

        auto consume_body = [&](uint32_t bytes_left) {
          uint8_t junk[64];
          while (bytes_left) {
            const uint16_t c = static_cast<uint16_t>(std::min<uint32_t>(bytes_left, 64));
            if (!TR.readExact(junk, c, 200)) break;
            bytes_left -= c;
          }
        };

        if (batch_count == 0 ||
            batch_nfeat <= 0 ||
            batch_nfeat > static_cast<int>(MAX_FEATURES) ||
            batch_need <= 0 ||
            batch_need > static_cast<int>(RECENTRY_DATA_BYTES) ||
            expected_len != body_len){
          consume_body(body_len);
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_BAD_BODY, h.seq);
          continue;
        }

        if (G.model_initialized && G.feat > 0 && batch_nfeat != G.feat) {
          consume_body(body_len);
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_BAD_N, h.seq);
          continue;
        }

        g_feat_cfg.input_real = false;

        const bool inline_tm_fast = use_inline_tm_compute_fastpath();
        uint32_t bytes_left = body_len;
        bool batch_ok = true;
        RecEntry inline_batch_entry{};
        inline_batch_entry.nfeat = static_cast<uint16_t>(batch_nfeat);
        for (uint16_t i = 0; i < batch_count; ++i){
          if (bytes_left < static_cast<uint32_t>(1 + batch_need)) {
            batch_ok = false;
            break;
          }

          if (inline_tm_fast) {
            if (!TR.readExact(reinterpret_cast<uint8_t*>(&inline_batch_entry.label),
                              static_cast<uint16_t>(1 + batch_need), 8000)) {
              batch_ok = false;
              break;
            }
            inline_batch_entry.label &= 1u;
            bytes_left -= static_cast<uint32_t>(1 + batch_need);
            handle_entry(inline_batch_entry);
          } else {
            RecEntry* bslot = nullptr;
            if (xQueueReceive(g_free_queue, &bslot, portMAX_DELAY) != pdTRUE) {
              batch_ok = false;
              break;
            }
            bslot->nfeat = static_cast<uint16_t>(batch_nfeat);
            if (!TR.readExact(reinterpret_cast<uint8_t*>(&bslot->label),
                              static_cast<uint16_t>(1 + batch_need), 8000)) {
              (void)xQueueSend(g_free_queue, &bslot, portMAX_DELAY);
              batch_ok = false;
              break;
            }
            bslot->label &= 1u;
            bytes_left -= static_cast<uint32_t>(1 + batch_need);
            if (xQueueSend(g_ready_queue, &bslot, portMAX_DELAY) != pdTRUE) {
              (void)xQueueSend(g_free_queue, &bslot, portMAX_DELAY);
              batch_ok = false;
              break;
            }
          }

          W.sent++;
          if (W.left > 0) {
            W.left--;
            if (W.left == 0) {
              TR.sendAck(W.sent, W.last_seq);
              W.last_acked = W.sent;
              W.left = W.cap ? W.cap : 1;
            }
          }
          ingested++;
        }

        if (!batch_ok || bytes_left != 0u){
          if (bytes_left > 0u) {
            consume_body(bytes_left);
          }
          if(W.sent>W.last_acked){ TR.sendAck(W.sent,W.last_seq); W.last_acked=W.sent; }
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_INCOMPLETE, h.seq);
          continue;
        }

        bounds_record_frame_validation(true, 0);
        continue;
      }
 
      if ((h.type==FRAME_TYPE_RECORD && (plen < sizeof(PackedHeader))) ||
          (h.type==FRAME_TYPE_RECORD_MC && (plen < sizeof(PackedHeaderMC))) ||
          (plen > 9000)){
        uint8_t junk[64]; uint16_t left=plen;
        while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; }
        if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150); }
        bounds_record_frame_validation(false, 1); // size error
        TR.sendError(ERR_BAD_FRAME, h.seq); continue; // FIX: do not break transport loop
      }
 
      int label = 0;
      int nfeat = 0;
      const uint16_t fbytes_all = (uint16_t)(plen);
      uint16_t fbytes = 0;
      int need = 0;
      if (h.type == FRAME_TYPE_RECORD){
        if (plen < sizeof(PackedHeader)){
          uint8_t junk[64]; uint16_t left=plen;
          while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; }
          if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150); }
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_BAD_FRAME, h.seq); continue;
        }
        PackedHeader ph; if (!TR.readExact((uint8_t*)&ph, sizeof(ph), 1000)){
          if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);} 
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_BAD_HDR, h.seq); continue; 
        }
        if (with_chk) chk = fletcher16_update((uint16_t)chk, (uint8_t*)&ph, sizeof(ph));
        const uint32_t hdr = ph.header_le;
        label = (int)((hdr>>31)&1);
        nfeat = (int)(hdr & 0x7FFFFFFF);
        fbytes = (uint16_t)(plen - sizeof(PackedHeader));
        need = (nfeat + 7)/8; // default bit-packed
      } else if (h.type == FRAME_TYPE_RECORD_MC) {
        PackedHeaderMC ph; if (!TR.readExact((uint8_t*)&ph, sizeof(ph), 1000)){
          if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);} 
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_BAD_HDR, h.seq); continue; 
        }
        if (with_chk) chk = fletcher16_update((uint16_t)chk, (uint8_t*)&ph, sizeof(ph));
        nfeat = (int)ph.nfeat_le; label = (int)ph.label;
        fbytes = (uint16_t)(plen - sizeof(PackedHeaderMC));
        need = (nfeat + 7)/8;
      } else {
        if (plen < sizeof(RawSampleHeader)){
          uint8_t junk[64]; uint16_t left=plen;
          while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; }
          if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150); }
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_BAD_FRAME, h.seq); continue;
        }
        RawSampleHeader rh; if (!TR.readExact((uint8_t*)&rh, sizeof(rh), 1000)){
          if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);} 
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_BAD_HDR, h.seq); continue; 
        }
        if (with_chk) chk = fletcher16_update((uint16_t)chk, (uint8_t*)&rh, sizeof(rh));
        nfeat = (int)rh.nfeat_le;
        label = (int)rh.label;
        const uint8_t dtype = rh.dtype; // 1=float32
        fbytes = (uint16_t)(plen - sizeof(RawSampleHeader));
        if (dtype != 1){ // unsupported dtype
          // consume body
          int left = fbytes; while(left){ int c=std::min(left,64); uint8_t tmp[64]; if(!TR.readExact(tmp,(uint16_t)c,200)) break; if(with_chk) chk=fletcher16_update((uint16_t)chk,tmp,(uint16_t)c); left-=c; }
          if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);} 
          TR.sendError(ERR_BAD_BODY, h.seq); continue;
        }
    // (moved buffer hint after successful body read)
        // mark that this row is real-valued regardless of config
        g_feat_cfg.input_real = true;
      }

      if (nfeat<=0){
        if (fbytes){ uint8_t junk[64]; uint16_t left=fbytes; while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; } }
        if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);}
        bounds_record_frame_validation(false, 3); // format error
        TR.sendError(ERR_BAD_N,h.seq); continue;
      }
      if ((h.type == FRAME_TYPE_RECORD || h.type==FRAME_TYPE_RECORD_MC) && !g_feat_cfg.input_real && fbytes != (uint16_t)need){
        if (fbytes){ uint8_t junk[64]; uint16_t left=fbytes; while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; } }
        if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);}
        bounds_record_frame_validation(false, 1); // size error
        TR.sendError(ERR_BAD_BODY,h.seq); continue;
      }
      if ((h.type == FRAME_TYPE_RECORD || h.type==FRAME_TYPE_RECORD_MC) && (need<=0 || need>(int)MAX_PACKED_BYTES)){
        if (fbytes){ uint8_t junk[64]; uint16_t left=fbytes; while(left){ uint16_t c=std::min<uint16_t>(left,64); if(!TR.readExact(junk,c,200))break; if(with_chk) chk=fletcher16_update((uint16_t)chk,junk,c); left-=c; } }
        if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);}
        bounds_record_frame_validation(false, 1); // size error
        TR.sendError(ERR_OVERFLOW,h.seq); continue;
      }
      RecEntry* slot = nullptr;
      bool body_in_slot = false;

      // Read body
      if (h.type == FRAME_TYPE_RECORD || h.type==FRAME_TYPE_RECORD_MC){
        const bool can_direct_to_slot = (nfeat <= (int)MAX_FEATURES) &&
                                        (need > 0) &&
                                        (need <= (int)RECENTRY_DATA_BYTES) &&
                                        (xQueueReceive(g_free_queue, &slot, 0) == pdTRUE);
        // Standard bit-packed path
        uint8_t* body_dst = can_direct_to_slot ? slot->data : packed;
        if (!TR.readExact(body_dst, need, 8000)){
          if (can_direct_to_slot && slot) {
            (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
            slot = nullptr;
          }
          if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,200);} 
          if(W.sent>W.last_acked){ TR.sendAck(W.sent,W.last_seq); W.last_acked=W.sent; }
          bounds_record_frame_validation(false, 3);
          TR.sendError(ERR_INCOMPLETE,h.seq); continue;
        }
        body_in_slot = can_direct_to_slot;
        if (with_chk) {
          chk = fletcher16_update((uint16_t)chk, body_dst, need);
          uint16_t rx=0; 
          if (!TR.readExact((uint8_t*)&rx,2,2000) || (uint16_t)rx!=(uint16_t)chk){ 
            if (can_direct_to_slot && slot) {
              (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
              slot = nullptr;
            }
            bounds_record_frame_validation(false, 2);
            TR.sendError(ERR_CHECKSUM,h.seq); continue; 
          }
        }
        // One-time RX buffer sizing hint for RECORD frames (after successful body read)
        {
          static bool s_buf_hint_record_logged = false;
          if (!s_buf_hint_record_logged){
            uint32_t frame_bytes = 16u + (uint32_t)need; // 2 magic + 8 hdr + 4 packedHdr + need + 2 chk
            uint32_t recommended = (uint32_t)W.cap * frame_bytes;
            LOG_SERIAL.printf("BUF HINT (RECORD): feats=%d need=%uB frame~%uB window=%u => rx_min~%uB (configured=16384)\n",
                              (int)nfeat, (unsigned)need, (unsigned)frame_bytes, (unsigned)W.cap, (unsigned)recommended);
            s_buf_hint_record_logged = true;
          }
        }
        // If incoming feature count exceeds MAX_FEATURES, crop to fit
        // Note: we've already read the full original 'need' bytes into 'packed'.
        // To avoid overflowing RecEntry::data (sized for MAX_FEATURES), recompute 'need'
        // based on the cropped feature count before copying into the queue entry.
        if (nfeat > (int)MAX_FEATURES){
          if (body_in_slot && slot) {
            // Cropping requires truncation of a larger payload; fall back to packed buffer copy path.
            (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
            slot = nullptr;
            body_in_slot = false;
          }
          static int s_crop_log_count = 0;
          if (s_crop_log_count < 3){
            LOG_SERIAL.printf("RECORD crop: nfeat %d -> %d (using first %d bytes)\n", nfeat, (int)MAX_FEATURES, (int)((MAX_FEATURES + 7) >> 3));
            s_crop_log_count++;
          }
          nfeat = (int)MAX_FEATURES;
          // Recompute 'need' to the bytes required by cropped feature count to prevent overflow
          need  = (int)((MAX_FEATURES + 7) >> 3);
        }
      } else { // FRAME_TYPE_SAMPLE_RAW
        // Real-valued float32 input (one row)
        int need_real = nfeat * (int)sizeof(float);
        const bool can_direct_raw = (nfeat <= (int)MAX_FEATURES) &&
                                    (need_real > 0) &&
                                    (need_real <= (int)RECENTRY_DATA_BYTES) &&
                                    (xQueueReceive(g_free_queue, &slot, 0) == pdTRUE);
        if (fbytes != (uint16_t)need_real){
          if (can_direct_raw && slot) {
            (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
            slot = nullptr;
          }
          // consume and drop frame if size mismatch
          int left = fbytes; while(left){ int c=std::min(left,64); uint8_t tmp[64]; if(!TR.readExact(tmp,(uint16_t)c,200)) break; if(with_chk) chk=fletcher16_update((uint16_t)chk,tmp,(uint16_t)c); left-=c; }
          if (with_chk){ uint8_t c2[2]; TR.readExact(c2,2,150);} 
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_BAD_BODY,h.seq); continue;
        }
        // Read safely even when incoming nfeat exceeds MAX_FEATURES.
        // Keep only the first MAX_PACKED_BYTES bytes and consume the rest into a scratch buffer.
        uint8_t* raw_dst = can_direct_raw ? slot->data : packed;
        const int keep_bytes = can_direct_raw ? need_real : std::min(need_real, (int)MAX_PACKED_BYTES);
        int consumed = 0;
        if (keep_bytes > 0) {
          if (!TR.readExact(raw_dst, (uint16_t)keep_bytes, 8000)) {
            if (can_direct_raw && slot) {
              (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
              slot = nullptr;
            }
            if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,200);} 
            if(W.sent>W.last_acked){ TR.sendAck(W.sent,W.last_seq); W.last_acked=W.sent; }
            bounds_record_frame_validation(false, 3);
            TR.sendError(ERR_INCOMPLETE,h.seq); continue;
          }
          consumed += keep_bytes;
          if (with_chk) {
            chk = fletcher16_update((uint16_t)chk, raw_dst, (uint16_t)keep_bytes);
          }
        }
        bool raw_incomplete = false;
        while (consumed < need_real) {
          uint8_t tmp[64];
          const int chunk = std::min(need_real - consumed, 64);
          if (!TR.readExact(tmp, (uint16_t)chunk, 8000)) {
            if (can_direct_raw && slot) {
              (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
              slot = nullptr;
            }
            if(with_chk){ uint8_t c2[2]; TR.readExact(c2,2,200);} 
            if(W.sent>W.last_acked){ TR.sendAck(W.sent,W.last_seq); W.last_acked=W.sent; }
            bounds_record_frame_validation(false, 3);
            TR.sendError(ERR_INCOMPLETE,h.seq);
            raw_incomplete = true;
            break;
          }
          if (with_chk) {
            chk = fletcher16_update((uint16_t)chk, tmp, (uint16_t)chunk);
          }
          consumed += chunk;
        }
        if (raw_incomplete) {
          continue;
        }
        if (with_chk) {
          uint16_t rx=0; 
          if (!TR.readExact((uint8_t*)&rx,2,2000) || (uint16_t)rx!=(uint16_t)chk){ 
            if (can_direct_raw && slot) {
              (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
              slot = nullptr;
            }
            bounds_record_frame_validation(false, 2);
            TR.sendError(ERR_CHECKSUM,h.seq); continue; 
          }
        }
        if (can_direct_raw) {
          body_in_slot = true;
        }
        // One-time RX buffer sizing hint for RAW frames (after successful body read)
        {
          static bool s_buf_hint_raw_logged = false;
          if (!s_buf_hint_raw_logged){
            uint32_t frame_bytes = 16u + 4u + (uint32_t)need_real; // 2 magic + 8 hdr + 4 rawHdr + need_real + 2 chk
            uint32_t recommended = (uint32_t)W.cap * frame_bytes;
            LOG_SERIAL.printf("BUF HINT (RAW): feats=%d frame~%uB window=%u => rx_min~%uB (configured=16384)\n",
                              (int)nfeat, (unsigned)frame_bytes, (unsigned)W.cap, (unsigned)recommended);
            s_buf_hint_raw_logged = true;
          }
        }

        int original_feats = nfeat;
        if (nfeat > (int)MAX_FEATURES){
          static int s_raw_crop_logs = 0;
          if (s_raw_crop_logs < 3) {
            LOG_SERIAL.printf("RAW crop: nfeat %d -> %d (using first %d floats)\n",
                              nfeat, (int)MAX_FEATURES, (int)MAX_FEATURES);
            s_raw_crop_logs++;
          }
          nfeat = (int)MAX_FEATURES;
        }
        int need_bytes = nfeat * (int)sizeof(float);
        if (need_bytes > (int)MAX_PACKED_BYTES){
          bounds_record_frame_validation(false, 1);
          TR.sendError(ERR_OVERFLOW, h.seq);
          continue;
        }
        if (original_feats != nfeat) {
          // No need to adjust buffer; only first nfeat floats will be copied below
        }
        need = need_bytes;
      }
 
      if (G.model_initialized && G.feat > 0 && nfeat != G.feat) {
        if (slot) {
          (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
          slot = nullptr;
          body_in_slot = false;
        }
        // Keep model dimension fixed for the run; mismatched records are treated as wire corruption.
        bounds_record_frame_validation(false, 3);  // format error
        TR.sendError(ERR_BAD_N, h.seq);
        continue;
      }

      // Acquire a free pool slot, then enqueue pointer to compute queue.
      if (!slot) {
        (void)xQueueReceive(g_free_queue, &slot, portMAX_DELAY);
      }
      slot->nfeat = (uint16_t)nfeat;
      slot->label = (uint8_t)label;
      if (!body_in_slot) {
        memcpy(slot->data, packed, (size_t)need);
      }

      if (use_inline_tm_compute_fastpath()) {
        handle_entry(*slot);
        (void)xQueueSend(g_free_queue, &slot, portMAX_DELAY);
      } else {
        (void)xQueueSend(g_ready_queue, &slot, portMAX_DELAY);
      }

     // Successfully received and queued RECORD frame
     bounds_record_frame_validation(true, 0);
     
     // Optimized ACK strategy - send ACK when window is full
     W.sent++;
      if (W.left > 0) {
        W.left--;
        // ACK when window is exhausted
        if (W.left == 0) {
          TR.sendAck(W.sent, W.last_seq); 
          W.last_acked = W.sent; 
          W.left = W.cap ? W.cap : 1;
        }
      }
      ingested++;
     }
 
   // periodic alive log removed for performance
   // (was logging every 10s with queue stats)
 
    // Periodic READY beacon until any RX to help host sync after reset/open
    if (!g_any_rx_seen && millis() - last_beacon_ms > 1000){
      TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
      last_beacon_ms = millis();
    }
 
    // FIX: Periodic DONE beacon when idle, but do NOT exit the task
    const bool idle_long = (millis() - last_rx_ms) > IDLE_DONE_MS;
    if (idle_long && (millis() - last_done_beacon_ms) > DONE_BEACON_MS){
      if (W.sent>W.last_acked){ TR.sendAck(W.sent, W.last_seq); W.last_acked=W.sent; }
      TR.sendDone(W.sent, W.last_seq);
      last_done_beacon_ms = millis();
    }
    
    // STATE MACHINE: Send READY beacon after idle period to signal ready for next cycle
    if (idle_long && (millis() - last_ready_beacon_ms) > READY_BEACON_MS){
      TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
      last_ready_beacon_ms = millis();
    }

    // FIX: If there are outstanding records but we haven't filled a window for a while, send a keep-alive ACK
     static unsigned long last_keepalive_ack_ms = 0;
     if ((W.sent > W.last_acked) && (millis() - last_keepalive_ack_ms > 1000)){
       TR.sendAck(W.sent, W.last_seq);
       W.last_acked = W.sent;
       last_keepalive_ack_ms = millis();
     }
 
     vTaskDelay(0);  // Changed from 1 to 0 for higher throughput
   }
 }
 
// ---------------- Compute task ----------------
static void compute_task(void* pv){
  LOG_SERIAL.println("compute_task: start");
  RecEntry* e = nullptr;
  for(;;){
    if (xQueueReceive(g_ready_queue, &e, portMAX_DELAY)==pdTRUE){
      if (!e) {
        continue;
      }
      handle_entry(*e);
      (void)xQueueSend(g_free_queue, &e, portMAX_DELAY);
    }
  }
}
 
void setup(){
  // ========== CRITICAL: Initialize hardware serial FIRST ==========
  // setRxBufferSize() MUST be called BEFORE begin() on HardwareSerial,
  // otherwise the UART driver uses the default 256-byte buffer and the
  // call silently fails (returns 0).
  ACTIVE_SERIAL.setRxBufferSize(SERIAL_RX_BUFFER_SIZE);
#if defined(SERIAL2_RX_PIN) && defined(SERIAL2_TX_PIN)
  ACTIVE_SERIAL.begin(SERIAL2_BAUD, SERIAL_8N1, SERIAL2_RX_PIN, SERIAL2_TX_PIN);
#else
  ACTIVE_SERIAL.begin(SERIAL2_BAUD);
#endif
  ACTIVE_SERIAL.setTimeout(1);
  
  // USB CDC enumeration delay
  delay(1000);
  
  // Initialize subsystems
  bounds_monitor_init();
  esp_task_wdt_deinit();
  
#if ENABLE_DISPLAY
  display_init();
#endif
  
  // Initialize memory and benchmark tracking
  utils_memory_init();
  benchmark_init();
  model_apply_runtime_defaults();
  
  // Log system information
  LOG_SERIAL.println("=== ESP32 Model System ===");
  LOG_SERIAL.printf("Protocol: v%u.%u\n", PROTO_VER_MAJOR, PROTO_VER_MINOR);
  
  // Memory status
  uint32_t heap_free = ESP.getFreeHeap();
  uint32_t heap_largest = ESP.getMaxAllocHeap();
  uint32_t heap_min = ESP.getMinFreeHeap();
  uint32_t psram_total = ESP.getPsramSize();
  uint32_t psram_free = ESP.getFreePsram();
  uint32_t psram_used = psram_total - psram_free;
  
  LOG_SERIAL.println("--- Memory Status ---");
  LOG_SERIAL.printf("Internal RAM: free=%u min=%u largest_block=%u\n", heap_free, heap_min, heap_largest);
  if (psram_total > 0) {
    LOG_SERIAL.printf("PSRAM: total=%u used=%u free=%u (%.1f%% used)\n", 
                      psram_total, psram_used, psram_free, 
                      (float)psram_used * 100.0f / (float)psram_total);
  } else {
    LOG_SERIAL.println("PSRAM: not available");
  }
  
  model_reset_runtime();

  // Log model summary
  LOG_SERIAL.println("--- Model Summary ---");
  LOG_SERIAL.printf("Model: runtime-select (default=%s)\n", model_name(g_selected_model));
  LOG_SERIAL.printf("Preprocessor: %s\n", preproc_name(g_preproc_cfg.mode));
  LOG_SERIAL.printf("Max Features: %d\n", (int)MAX_FEATURES);
  LOG_SERIAL.printf("Num Classes: %u\n", (unsigned)model_selected_num_classes());
  
  // Queue configuration
  uint32_t pool_bytes = sizeof(RecEntry) * RECV_Q_CAP;
  uint32_t ptr_queue_bytes = (uint32_t)(sizeof(RecEntry*) * RECV_Q_CAP * 2u);
  LOG_SERIAL.println("--- Queue Configuration ---");
  LOG_SERIAL.printf("RecEntry Size: %u bytes\n", (unsigned)sizeof(RecEntry));
  LOG_SERIAL.printf("Queue Capacity: %u entries\n", (unsigned)RECV_Q_CAP);
  LOG_SERIAL.printf("Pool Memory: %u bytes (%.2f KB)\n", pool_bytes, (float)pool_bytes / 1024.0f);
  LOG_SERIAL.printf("Pointer Queue Memory: %u bytes (%.2f KB)\n", ptr_queue_bytes, (float)ptr_queue_bytes / 1024.0f);
  
  // Drain stale serial data
  while (ACTIVE_SERIAL.available()) { ACTIVE_SERIAL.read(); }
  
  // Send initial READY frame
  TR.sendReady(PROTO_VER_MAJOR, PROTO_VER_MINOR, G.trained, G.tested);
  
  // Create pointer queues (free slots + ready slots)
  g_queue_capacity = RECV_Q_CAP;
  g_free_queue = xQueueCreate(g_queue_capacity, sizeof(RecEntry*));
  g_ready_queue = xQueueCreate(g_queue_capacity, sizeof(RecEntry*));
  if (!g_free_queue || !g_ready_queue) {
    LOG_SERIAL.printf("ERROR: Queue creation failed (free=%u). Trying fallback capacity 16.\n", ESP.getFreeHeap());
    g_queue_capacity = 16;
    if (g_free_queue) { vQueueDelete(g_free_queue); g_free_queue = nullptr; }
    if (g_ready_queue) { vQueueDelete(g_ready_queue); g_ready_queue = nullptr; }
    g_free_queue = xQueueCreate(g_queue_capacity, sizeof(RecEntry*));
    g_ready_queue = xQueueCreate(g_queue_capacity, sizeof(RecEntry*));
    if (!g_free_queue || !g_ready_queue) {
      LOG_SERIAL.println("FATAL: Cannot create pointer queues. System halted.");
      return;
    }
    LOG_SERIAL.println("WARNING: Using reduced queue capacity (16 entries)");
  }

  for (UBaseType_t i = 0; i < g_queue_capacity; ++i) {
    RecEntry* slot = &g_rec_pool[i];
    if (xQueueSend(g_free_queue, &slot, 0) != pdTRUE) {
      LOG_SERIAL.println("FATAL: Failed to populate free queue.");
      return;
    }
  }
  LOG_SERIAL.printf("Active Queue Capacity: %u entries\n", (unsigned)g_queue_capacity);

  // Create FreeRTOS tasks
  xTaskCreatePinnedToCore(transport_task, "transport", 8192, nullptr, 8, nullptr, CORE_TRANSPORT);
  xTaskCreatePinnedToCore(compute_task,   "compute",   8192, nullptr, 2, nullptr, CORE_COMPUTE);
  
  // Final memory snapshot after initialization
  LOG_SERIAL.println("--- Post-Init Memory ---");
  LOG_SERIAL.printf("Internal RAM: free=%u\n", ESP.getFreeHeap());
  if (psram_total > 0) {
    psram_free = ESP.getFreePsram();
    psram_used = psram_total - psram_free;
    LOG_SERIAL.printf("PSRAM: used=%u free=%u (%.1f%% used)\n", 
                      psram_used, psram_free, (float)psram_used * 100.0f / (float)psram_total);
  }
  LOG_SERIAL.println("=== System Ready ===");
}
 
void loop(){
  // Main loop runs idle - all work done in FreeRTOS tasks
  vTaskDelay(pdMS_TO_TICKS(1000));
}
 
