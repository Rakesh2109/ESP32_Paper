# ESP32-S3 Streaming ML (Easy Guide)

This project runs **online machine learning on ESP32-S3**.
The board receives samples over serial, learns in streaming mode, and then reports test accuracy and throughput.

---

## What this project does

- Uses ESP32-S3 as an online learning device.
- Streams data from PC to ESP32 in binary frames.
- Trains and tests on-device in one continuous pipeline.
- Supports 3 stream-learning tree models plus Tsetlin Machine and BNN:
  - Hoeffding Tree
  - HAT (Hoeffding Adaptive Tree)
  - EFDT (Extremely Fast Decision Tree)
  - TM_BO, TM_SPARSE (Tsetlin Machine)
  - TM_VANILLA (Tsetlin Machine)
  - BNN (Binary Neural Network)

---

## Results

Experiment results from device folders `C3_iiot_energy`, `C3_iot_energy`, `S3_IIoT_energy`, `S3_IoT_energy`, `P4_IIoT`, `P4_IoT`, and `p4_BNN`. All runs use bitpack transport, host StandardScaler, and `unsafe_no_data_checksum` for throughput.

### ESP32-C3 | UKMNCT_IIoT_FDIA (C3_iiot_energy)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TM_BO | 3,822 | 9,414 | 3.23 | 0.33 | 25.43 | 6.40 | 0.106 |
| TM_SPARSE | 3,800 | 9,341 | 3.25 | 0.33 | 25.37 | 5.98 | 0.107 |
| TM_VANILLA | 2,067 | 3,805 | 5.97 | 0.81 | 49.05 | 24.85 | 0.263 |
| EFDT | 567 | 13,745 | 21.78 | 0.22 | 169.67 | 5.61 | 0.073 |
| BNN | 68 | 1,049 | 182.56 | 2.94 | 1,439.0 | 93.48 | 0.953 |

### ESP32-C3 | IoT_clean (C3_iot_energy)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TM_BO | 4,059 | 11,416 | 20.47 | 1.82 | 25.28 | 7.99 | 0.088 |
| TM_SPARSE | 4,031 | 11,256 | 20.61 | 1.85 | 25.30 | 8.35 | 0.089 |
| TM_VANILLA | 2,778 | 5,083 | 29.90 | 4.09 | 36.92 | 19.47 | 0.197 |
| BNN | 79 | 1,092 | 1,052.2 | 19.02 | 1,233.7 | 89.76 | 0.916 |

### ESP32-S3 | UKMNCT_IIoT_FDIA (S3_IIoT_energy)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| HOEFFDING | 8,519 | 31,120 | 1.45 | 0.10 | 26.94 | — | 0.032 |
| HAT | 6,225 | 30,549 | 1.98 | 0.10 | 36.75 | — | 0.033 |
| EFDT | 3,969 | 30,695 | 3.11 | 0.10 | 58.14 | — | 0.033 |
| TM_BO | 7,567 | 20,256 | 32.61 | 3.05 | 31.78 | — | 0.049 |
| TM_SPARSE | 7,572 | 20,265 | 32.59 | 3.04 | 31.71 | — | 0.049 |
| TM_VANILLA | 3,325 | 8,458 | 74.23 | 7.30 | 70.07 | 16.49 | 0.118 |
| BNN | 1,030 | 4,494 | 239.71 | 13.73 | 218.15 | 43.36 | 0.222 |

*S3_IIoT_energy Energy Test: `—` for HOEFFDING, HAT, EFDT, TM_BO, TM_SPARSE because the test phase (~0.1 s) was too short for Joulescope to capture; TM_VANILLA and BNN had multi-epoch runs with longer test phases.*

### ESP32-S3 | IoT_clean (S3_IoT_energy)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TM_BO | 7,162 | 21,399 | 231.99 | 19.41 | 34.91 | 12.66 | 0.047 |
| TM_SPARSE | 7,160 | 21,405 | 232.04 | 19.40 | 35.05 | 12.60 | 0.047 |
| TM_VANILLA | 5,251 | 10,697 | 316.42 | 38.83 | 46.29 | 21.91 | 0.093 |
| HOEFFDING | 5,243 | 36,908 | 15.84 | 0.56 | 46.61 | 4.62 | 0.027 |
| HAT | 5,142 | 37,199 | 16.15 | 0.56 | 45.96 | 4.66 | 0.027 |
| EFDT | 2,087 | 37,000 | 39.81 | 0.56 | 110.06 | 4.39 | 0.027 |
| BNN | 1,144 | 4,778 | 1,452.9 | 86.94 | 196.77 | 48.69 | 0.209 |

### ESP32-P4 | UKMNCT_IIoT_FDIA (P4_IIoT)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TM_SPARSE | 15,953 | 27,640 | 15.47 | 2.23 | — | — | 0.036 |
| TM_BO | 12,700 | 32,462 | 19.43 | 1.90 | — | — | 0.031 |
| HAT | 11,667 | 37,166 | 1.06 | 0.08 | — | — | 0.027 |
| HOEFFDING | 11,632 | 36,788 | 1.06 | 0.08 | — | — | 0.027 |
| EFDT | 7,281 | 37,755 | 1.69 | 0.08 | — | — | 0.026 |
| TM_VANILLA | 6,649 | 15,255 | 37.12 | 4.04 | — | — | 0.066 |
| BNN | 1,836 | 9,659 | 134.39 | 6.39 | — | — | 0.104 |

### ESP32-P4 | IoT_clean (P4_IoT)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TM_SPARSE | 15,455 | 39,871 | 107.50 | 10.42 | — | — | 0.025 |
| TM_BO | 12,480 | 34,075 | 133.13 | 12.19 | — | — | 0.029 |
| HOEFFDING | 10,133 | 38,899 | 8.20 | 0.53 | — | — | 0.026 |
| HAT | 7,222 | 39,137 | 11.50 | 0.53 | — | — | 0.026 |
| EFDT | 2,667 | 38,831 | 31.14 | 0.53 | — | — | 0.026 |
| TM_VANILLA | 8,924 | 19,009 | 186.17 | 21.85 | — | — | 0.053 |

### ESP32-P4 | IoT_clean | BNN only (p4_BNN)

| Model | Train Throughput (smp/s) | Test Throughput (smp/s) | Train Time (s) | Test Time (s) | Energy Train (µJ/smp) | Energy Test (µJ/smp) | ms/sample (test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BNN | 2,055 | 10,269 | 404.18 | 20.22 | — | — | 0.097 |

*Note: Energy columns show `—` where Joulescope (`--js-power`) was not used. C3/S3 energy folders used Joulescope; P4 folders did not.*

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

### 4) TM_BO, TM_SPARSE (Tsetlin Machine)

- Tsetlin Machine for streaming classification.
- Boolean inputs, clause-based inference; low energy per sample.

### 5) TM_VANILLA (Tsetlin Machine)

- Standard Tsetlin Machine variant.
- Higher compute per sample than TM_BO/TM_SPARSE.

### 6) BNN (Binary Neural Network)

- Binary neural network for on-device streaming learning.
- Multi-epoch training; higher energy per sample than tree models.

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
#define MODEL_TYPE MODEL_BNN
#define MODEL_TYPE MODEL_TM_SPARSE
#define MODEL_TYPE MODEL_TM_VANILLA
#define MODEL_TYPE MODEL_TM_BO
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
| `--models ALL` | Run all models (EFDT, HOEFFDING, HAT, BNN, TM_SPARSE, TM_VANILLA, TM_BO) |
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
