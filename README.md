# ESP32-S3 Streaming ML (Easy Guide)

This project runs **online machine learning on ESP32-S3**.
The board receives samples over serial, learns in streaming mode, and then reports test accuracy and throughput.

---

## What this project does

- Uses ESP32-S3 as an online learning device.
- Streams data from PC to ESP32 in binary frames.
- Trains and tests on-device in one continuous pipeline.
- Supports 4 stream-learning tree models:
  - Hoeffding Tree
  - HAT (Hoeffding Adaptive Tree)
  - EFDT (Extremely Fast Decision Tree)
  - SGT (Streaming Gradient Tree)

---

## Latest results (from `results/` folder)

These are from your latest `window=64`, `throttle=0` experiments on Feb 24, 2026.

| Model | Train Time (s) | Train Throughput (smp/s) | Final Train Acc | Test Time (s) | Final Test Acc | Avg Throughput (smp/s) |
|---|---:|---:|---:|---:|---:|---:|
| EFDT | 40.840 | 2034.0 | 93.74% | 9.719 | 96.85% | 2053.8 |
| HAT | 39.288 | 2114.4 | 91.16% | 9.692 | 97.59% | 2120.0 |
| HOEFFDING | 39.590 | 2098.3 | 91.04% | 9.716 | 97.68% | 2106.0 |

### Quick interpretation

- **Best test accuracy** in this batch: **HOEFFDING (97.68%)**.
- **Fastest overall throughput** in this batch: **HAT (2120 smp/s avg)**.
- EFDT works well, but is slightly slower and lower accuracy in this run set.
- SGT is implemented but not included in these attached `results/` files.

---

## How data is preprocessed

There are two stages:

### 1) Offline dataset preparation (optional)

`code_IoT_clean.py` prepares and compares models in a Python (River/scikit-learn) pipeline.

- Reads `IoT_clean.csv`
- Splits into train/test
- Optionally applies normalization
- Evaluates models offline

### 2) Online preprocessing during device run (main workflow)

Main runtime script is `test_serial.py`.

For CSV mode (your current flow):

1. Load CSV.
2. Pick label column (default: last column).
3. Convert labels to binary `{0,1}` if needed.
4. Split data to train/test (`train_frac`, default 0.8).
5. Convert features to `float32`.
6. Apply `StandardScaler` on host:
   - `fit` on train
   - `transform` on train and test
7. Stream normalized rows as raw `float32` samples to ESP32.

Important:

- If `sklearn` is missing, the script skips scaler and shows a warning.
- Device-side normalization is disabled in firmware; host does normalization.

---

## How data is sent to ESP32

PC (`test_serial.py`) talks to ESP32 (`src/core/main.cpp`) using a custom binary protocol (`src/core/protocol.h`).

### Step-by-step flow

1. Open serial and wait for `READY` frame.
2. Send `OPC_CONFIG` (for example `INPUT_REAL` flag).
3. Send `OPC_BUFFER` with window size (ACK batching).
4. Send `OPC_START_TRAIN`.
5. Stream training samples as frames.
6. Wait for ACK/STAT feedback while streaming.
7. Send `OPC_SYNC` so device queue is fully drained.
8. Send `OPC_START_TEST` and stream test data.
9. Send `OPC_SHOW_FINAL` to receive final stats/analysis.

### Frame format

Protocol uses:

- `FRAME_MAGIC = 0xA55A`
- Header = magic + type + flags + payload length + sequence number
- Optional Fletcher checksum depending on flags

Main frame types:

- `FRAME_TYPE_SAMPLE_RAW` (`0x06`): raw float32 sample row
- `FRAME_TYPE_RECORD` (`0x01`): bit-packed binary features
- `FRAME_TYPE_CMD` (`0x02`): control command
- `FRAME_TYPE_ACK`, `FRAME_TYPE_STAT`, `FRAME_TYPE_DONE`, `FRAME_TYPE_READY`: flow control/status

For your current CSV + `--input-real` pipeline, each sample payload is:

- `nfeat` (`uint16`)
- `label` (`uint8`)
- `dtype` (`uint8`, `1 = float32`)
- `nfeat` float32 values

---

## What each model is doing (simple explanation)

### 1) Hoeffding Tree

- Builds a decision tree incrementally from stream data.
- Uses Hoeffding bound to decide when enough data exists to split a node.
- Good speed/accuracy trade-off for stable streams.

### 2) HAT (Hoeffding Adaptive Tree)

- Starts like Hoeffding Tree.
- Adds drift handling by creating alternate subtrees when errors rise.
- Can switch to alternate branch if it becomes better.
- Good when data distribution changes over time.

### 3) EFDT (Extremely Fast Decision Tree)

- Similar family to Hoeffding Tree.
- Re-evaluates existing splits periodically and can replace weaker splits.
- More adaptive than plain Hoeffding, usually with extra compute cost.

### 4) SGT (Streaming Gradient Tree)

- Gradient-based streaming tree (uses gradient/hessian style updates).
- Optimizes split gain with regularization terms (`lambda`, `gamma`).
- Often very fast; accuracy depends heavily on data and tuning.

---

## How to switch model

Edit `src/core/main.cpp`:

```cpp
#define MODEL_TYPE MODEL_HOEFFDING
```

Options:

```cpp
#define MODEL_TYPE MODEL_EFDT
#define MODEL_TYPE MODEL_HOEFFDING
#define MODEL_TYPE MODEL_HAT
#define MODEL_TYPE MODEL_SGT
#define MODEL_TYPE MODEL_BNN
```

Then rebuild and upload.

---

## Quick run commands

### 1) Build

```bash
platformio run -e esp32_s3_n16r8
```

### 2) Upload

```bash
platformio run -e esp32_s3_n16r8 -t upload
```

### 3) Run all models (ESP32-P4-NANO via built-in USB)

```bash
# P4-NANO with built-in CH343P (Type-C port)
# Use /dev/cu.usbserial-0001 on macOS
python3 run_all_models.py --models ALL --transport bitpack \
  -p /dev/cu.usbserial-0001 -b 921600 \
  --dataset UKMNCT_IIoT_FDIA --runs 1 \
  --window 1024 --batch-records 1024 \
  --unsafe-no-data-checksum --skip-flash \
  --board-profile esp32-p4_nano --js-power --no-hw-reset
```

**Flag explanations:**
| Flag | Meaning |
|------|---------|
| `--models ALL` | Run all models (EFDT, HOEFFDING, HAT, SGT, BNN, TM_SPARSE, TM_VANILLA, TM_BO) |
| `--transport bitpack` | Use bit-packed binary transport |
| `-p /dev/cu.usbserial-0001` | Serial port (built-in USB on P4-NANO) |
| `-b 921600` | Baud rate |
| `--dataset UKMNCT_IIoT_FDIA` | Dataset name (loads from `data/UKMNCT_IIoT_FDIA.csv`) |
| `--runs 1` | Number of experiment runs |
| `--window 1024` | Flow-control window size |
| `--batch-records 1024` | Records per data frame |
| `--unsafe-no-data-checksum` | Skip checksums for throughput |
| `--skip-flash` | Do not flash firmware (device already programmed) |
| `--board-profile esp32-p4_nano` | Board preset (baud, window, pio env) |
| `--js-power` | Joulescope power/energy measurement |
| `--no-hw-reset` | No DTR/RTS reset; use software reset only |

### 4) ESP32-C3-Mini

```bash
python3 run_all_models.py --models ALL --transport bitpack \
  -p /dev/cu.usbmodem1401 -b 921600 \
  --dataset UKMNCT_IIoT_FDIA --runs 1 \
  --window 256 --batch-records 256 \
  --unsafe-no-data-checksum --board-profile esp32_c3_mini --js-power
```

### 5) ESP32-S3-N16R8

```bash
python3 run_all_models.py --models ALL --transport bitpack \
  -p /dev/cu.usbmodem* -b 921600 \
  --dataset UKMNCT_IIoT_FDIA --runs 1 \
  --window 1024 --batch-records 1024 \
  --unsafe-no-data-checksum --board-profile esp32_s3_n16r8 --js-power
```

### 6) Single model (test_serial.py)

```bash
python3 test_serial.py \
  -p /dev/cu.usbmodem1101 \
  --dataset IoT_clean \
  --input-real \
  --window 64
```

---

## Result files you will get

Inside `results/`:

- `experiment_*.json` -> machine-readable metrics
- `experiment_*.txt` -> readable summary
- `experiment_log_*.txt` -> full run logs from host + ESP32

---

## Upload stability note (ESP32-S3)

If upload fails with RAM/stub checksum errors, safer settings are already configured for `esp32_s3_n16r8`:

- lower upload baud
- use `--no-stub`

This is why recent uploads succeeded.

---

## Recommended default today

If you want one practical default right now:

- **Model:** HOEFFDING
- **Input mode:** `--input-real` (raw float32 with host StandardScaler)
- **Window:** `64`
- **Throttle:** `0`

This gives strong accuracy with stable high throughput on your current setup.
