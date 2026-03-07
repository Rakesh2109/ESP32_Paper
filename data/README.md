# IoT Network Intrusion Detection Dataset

## 📊 Dataset Files

| File | Samples | Size | Purpose |
|------|---------|------|---------|
| **IoT_train.txt** | 159,732 | 25 MB | Training data |
| **IoT_test.txt** | 39,934 | 6.2 MB | Testing data |
| **IoT_data.txt** | 199,666 | 31 MB | Complete dataset |

**Total Samples:** 199,666  
**Total Size:** ~62 MB

---

## 📋 Dataset Description

This dataset contains **IoT network traffic data** for **intrusion detection** and **anomaly detection** tasks.

### Features:
- **22 numerical features** per sample
- Space-separated values
- Network traffic statistics (flow duration, packet counts, byte counts, etc.)
- IP addresses (encoded as numerical values)
- Protocol information
- Timestamp features

### Target Classes:
Binary classification:
- **Class 0:** Normal traffic
- **Class 1:** Attack/Malicious traffic

---

## 🎯 Usage in ESP32 Tsetlin Machine

This dataset is used in the ESP32-optimized Tsetlin Machine implementation for:

1. **Binary Classification:** Normal vs Attack detection
2. **Binarization Testing:** StandardBinarizer with 4 bits
3. **SHA Optimizer Validation:** Hyperparameter optimization
4. **Performance Benchmarking:** Real-world IoT scenario

### Dataset Split:
- **Training:** 159,732 samples (80%)
- **Testing:** 39,934 samples (20%)

---

## 📈 Performance Results

Using ESP32 TM with Xorshift128+ RNG:

| Metric | Value |
|--------|-------|
| **Accuracy** | 97.60% |
| **F1 Score** | 0.9683 |
| **Precision** | 0.9839 |
| **Recall** | 0.9531 |
| **Inference Time** | 146.16 μs/sample |
| **Training Time** | 55.39 sec |
| **Memory Usage** | 122.32 KB |

---

## 🔧 Data Format

Each line represents one sample:

```
<feature1> <feature2> ... <feature22> <label>
```

Example:
```
32883.0 1 167772164.0 167772163.0 111723.0 ... 1.0 0.0 0.0 1.0
```

**Feature Types:**
- Flow duration
- Protocol type
- Source/destination IPs (encoded)
- Port information
- Packet statistics
- Byte statistics
- Flag counts
- TCP/UDP specific features
- Time-based features

---

## 🚀 Quick Start

### Load Dataset in Code:

```cpp
#include "tsetlin_machine_esp32.h"

// Load training data
int n_train = 159732;
uint8_t** X_train = load_data("data/IoT_train.txt", n_train);
uint8_t* y_train = load_labels("data/IoT_train.txt", n_train);

// Load test data
int n_test = 39934;
uint8_t** X_test = load_data("data/IoT_test.txt", n_test);
uint8_t* y_test = load_labels("data/IoT_test.txt", n_test);

// Binarize features
StandardBinarizer binarizer;
binarizer_init(&binarizer, 22, 4); // 22 features, 4 bits
uint8_t** X_train_bin = binarize_dataset(&binarizer, X_train, n_train);

// Train Tsetlin Machine
TsetlinMachine tm;
tm_init(&tm, 88, 800, 15, 42); // 88 binary features, 800 clauses
tm_fit(&tm, X_train_bin, y_train, n_train, 30, 10);

// Evaluate
float accuracy = tm_evaluate(&tm, X_test_bin, y_test, n_test);
printf("Accuracy: %.2f%%\n", accuracy * 100);
```

---

## 📚 Dataset Source

This dataset is derived from **IoT network traffic captures** and preprocessed for machine learning tasks.

**Characteristics:**
- Real-world IoT device traffic
- Multiple attack types
- Balanced classes
- Preprocessed and normalized

---

## 🔬 Research Use

This dataset is used for:
- **Intrusion Detection Systems (IDS)**
- **Anomaly Detection**
- **IoT Security Research**
- **Machine Learning on Embedded Systems**
- **Tsetlin Machine Benchmarking**

---

## 📊 Feature Statistics

| Feature | Description | Example Range |
|---------|-------------|---------------|
| 1 | Flow duration | 0 - 50000 |
| 2 | Protocol type | 1, 2, 3 |
| 3-4 | Source/Dest IP | Encoded integers |
| 5-6 | Port numbers | 0 - 65535 |
| 7-8 | Total bytes | 0 - 10^9 |
| 9 | Protocol | TCP/UDP/ICMP |
| 10-15 | Packet counts | Various |
| 16-22 | TCP flags & features | 0/1 binary |

---

## ⚙️ Data Preprocessing

**Already Applied:**
- ✅ Missing value handling
- ✅ Feature normalization
- ✅ Label encoding
- ✅ Train/test split (80/20)

**Applied in Code:**
- Binarization (StandardBinarizer)
- Feature scaling (if needed)
- Data type conversion (float → uint8_t)

---

## 📝 Citation

If you use this dataset in research, please cite:

```
ESP32-Optimized Tsetlin Machine for IoT Intrusion Detection
Repository: https://github.com/Rakesh2109/ESP32-TM-Optimized
```

---

## 🔗 Related Files

- **Main Test:** `tests/main_test.cpp`
- **Binarizer:** `src/standard_binarizer.cpp`
- **TM Implementation:** `src/tsetlin_machine_esp32.cpp`
- **Results:** `FINAL_RESULTS.txt`
- **Analysis:** `NOVELTY_ANALYSIS.md`

---

## 📦 File Integrity

| File | MD5 Checksum | Lines |
|------|-------------|--------|
| IoT_train.txt | (compute locally) | 159,732 |
| IoT_test.txt | (compute locally) | 39,934 |
| IoT_data.txt | (compute locally) | 199,666 |

Verify integrity:
```bash
md5sum *.txt
wc -l *.txt
```

---

## 🎓 Learning Resources

- **Tsetlin Machine Paper:** [arXiv:1804.01508](https://arxiv.org/abs/1804.01508)
- **Project Documentation:** See root README.md
- **Performance Analysis:** NOVELTY_ANALYSIS.md
- **Results Summary:** FINAL_RESULTS.txt

---

## ⚠️ Important Notes

1. **File Size:** These are large files (~62MB total). Git can handle them, but cloning will take time.
2. **Memory Requirements:** Loading full dataset requires ~200MB RAM.
3. **ESP32 Usage:** For actual ESP32 deployment, use smaller subset or streaming.
4. **Data Format:** Space-separated, ensure proper parsing.

---

## 🔄 Dataset Updates

- **Version:** 1.0
- **Last Updated:** October 10, 2025
- **Status:** Stable, production-ready
- **Changes:** None (initial version)

---

**Repository:** https://github.com/Rakesh2109/ESP32-TM-Optimized  
**Dataset Location:** `/data/`  
**License:** Same as repository license

