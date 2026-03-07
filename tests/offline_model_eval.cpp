#include "models/bnn_types.h"
#include "models/efdt_model.h"
#include "models/hat_model.h"
#include "models/hoeffding_tree_model.h"
#include "models/sgt_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Dataset {
    std::vector<std::vector<float>> x;
    std::vector<int> y;
};

std::string trim(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

int parse_label(const std::string& raw, std::unordered_map<std::string, int>& dynamic_map) {
    const std::string token = trim(raw);

    try {
        const float numeric = std::stof(token);
        return static_cast<int>(std::lround(numeric));
    } catch (...) {
    }

    std::string lowered;
    lowered.reserve(token.size());
    for (char c : token) {
        lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lowered == "attack" || lowered == "malicious" || lowered == "anomaly" || lowered == "abnormal") {
        return 1;
    }
    if (lowered == "natural" || lowered == "normal" || lowered == "benign") {
        return 0;
    }

    auto it = dynamic_map.find(lowered);
    if (it != dynamic_map.end()) {
        return it->second;
    }

    const int next_value = static_cast<int>(dynamic_map.size());
    dynamic_map.emplace(lowered, next_value);
    return next_value;
}

Dataset load_csv_dataset(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open dataset: " + path);
    }

    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Dataset is empty: " + path);
    }

    const std::vector<std::string> header = split_csv_line(header_line);
    if (header.size() < 2) {
        throw std::runtime_error("Dataset must have at least one feature and one label column");
    }

    int label_idx = static_cast<int>(header.size()) - 1;
    for (std::size_t i = 0; i < header.size(); ++i) {
        std::string lowered;
        lowered.reserve(header[i].size());
        for (char c : header[i]) {
            lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
        if (lowered == "label" || lowered == "target" || lowered == "class" || lowered == "marker") {
            label_idx = static_cast<int>(i);
            break;
        }
    }

    Dataset dataset;
    std::unordered_map<std::string, int> dynamic_label_map;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> cols = split_csv_line(line);
        if (cols.size() != header.size()) {
            continue;
        }

        std::vector<float> features;
        features.reserve(cols.size() - 1);

        bool valid = true;
        for (std::size_t i = 0; i < cols.size(); ++i) {
            if (static_cast<int>(i) == label_idx) {
                continue;
            }
            try {
                features.push_back(std::stof(cols[i]));
            } catch (...) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        dataset.x.push_back(std::move(features));
        dataset.y.push_back(parse_label(cols[static_cast<std::size_t>(label_idx)], dynamic_label_map));
    }

    if (dataset.x.empty()) {
        throw std::runtime_error("No valid rows parsed from dataset: " + path);
    }

    return dataset;
}

std::pair<Dataset, Dataset> split_train_test(Dataset all, float train_frac, unsigned seed) {
    if (train_frac <= 0.0f || train_frac >= 1.0f) {
        throw std::runtime_error("train_frac must be in (0,1)");
    }

    std::vector<std::size_t> idx(all.x.size());
    std::iota(idx.begin(), idx.end(), 0u);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    const std::size_t train_n = static_cast<std::size_t>(std::floor(static_cast<double>(idx.size()) * train_frac));

    Dataset train;
    Dataset test;
    train.x.reserve(train_n);
    train.y.reserve(train_n);
    test.x.reserve(idx.size() - train_n);
    test.y.reserve(idx.size() - train_n);

    for (std::size_t i = 0; i < idx.size(); ++i) {
        const std::size_t src = idx[i];
        if (i < train_n) {
            train.x.push_back(std::move(all.x[src]));
            train.y.push_back(all.y[src]);
        } else {
            test.x.push_back(std::move(all.x[src]));
            test.y.push_back(all.y[src]);
        }
    }

    return {std::move(train), std::move(test)};
}

template <typename Model>
float evaluate_accuracy(Model& model, const Dataset& ds) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < ds.x.size(); ++i) {
        if (model.predict(ds.x[i]) == ds.y[i]) {
            correct += 1u;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(ds.x.size());
}

void print_metric(const std::string& name, float train_acc, float test_acc) {
    std::cout << std::left << std::setw(8) << name
              << " train_acc=" << std::fixed << std::setprecision(4) << train_acc
              << " test_acc=" << std::fixed << std::setprecision(4) << test_acc
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <csv_path> [train_frac=0.8] [seed=42] [max_train=0] [max_test=0]\\n";
            return 1;
        }

        const std::string csv_path = argv[1];
        const float train_frac = (argc >= 3) ? std::stof(argv[2]) : 0.8f;
        const unsigned seed = (argc >= 4) ? static_cast<unsigned>(std::stoul(argv[3])) : 42u;
        const std::size_t max_train = (argc >= 5) ? static_cast<std::size_t>(std::stoull(argv[4])) : 0u;
        const std::size_t max_test = (argc >= 6) ? static_cast<std::size_t>(std::stoull(argv[5])) : 0u;

        Dataset all = load_csv_dataset(csv_path);
        auto split = split_train_test(std::move(all), train_frac, seed);
        Dataset train = std::move(split.first);
        Dataset test = std::move(split.second);

        if (max_train > 0u && train.x.size() > max_train) {
            train.x.resize(max_train);
            train.y.resize(max_train);
        }
        if (max_test > 0u && test.x.size() > max_test) {
            test.x.resize(max_test);
            test.y.resize(max_test);
        }

        int max_label = 0;
        for (int y : train.y) {
            max_label = std::max(max_label, y);
        }
        for (int y : test.y) {
            max_label = std::max(max_label, y);
        }
        const std::size_t num_classes = static_cast<std::size_t>(max_label + 1);
        const std::size_t input_dim = train.x.front().size();

        std::cout << "Dataset: " << csv_path << '\n';
        std::cout << "Train: " << train.x.size() << " Test: " << test.x.size()
                  << " Features: " << input_dim << " Classes: " << num_classes << '\n';

        tree::EfdtConfig efdt_cfg;
        efdt_cfg.num_classes = num_classes;
        efdt_cfg.delta = 1e-7f;
        efdt_cfg.grace_period = 32;
        efdt_cfg.min_samples_split = 64;
        efdt_cfg.tie_threshold = 0.05f;
        efdt_cfg.reevaluate_period = 64;

        tree::EfdtModel efdt(input_dim, efdt_cfg);
        std::cout << "Training EFDT...\n";
        for (std::size_t i = 0; i < train.x.size(); ++i) {
            bnn::Sample sample{train.x[i], train.y[i]};
            efdt.train_sample(sample);
            if ((i + 1) % 500 == 0) {
                std::cout << "  Processed " << (i + 1) << " samples\n";
            }
        }
        const float efdt_train_acc = evaluate_accuracy(efdt, train);
        const float efdt_test_acc = evaluate_accuracy(efdt, test);
        print_metric("EFDT", efdt_train_acc, efdt_test_acc);

        tree::HatConfig hat_cfg;
        hat_cfg.num_classes = num_classes;
        hat_cfg.delta = 1e-7f;
        hat_cfg.grace_period = 32;
        hat_cfg.min_samples_split = 64;
        hat_cfg.tie_threshold = 0.05f;
        hat_cfg.num_threshold_bins = 16;
        hat_cfg.max_depth = 16;
        hat_cfg.bootstrap_sampling = true;
        hat_cfg.drift_window_threshold = 300;
        hat_cfg.switch_significance = 0.05f;
        hat_cfg.seed = seed;

        tree::HatModel hat(input_dim, hat_cfg);
        std::cout << "Training HAT...\n";
        for (std::size_t i = 0; i < train.x.size(); ++i) {
            bnn::Sample sample{train.x[i], train.y[i]};
            hat.train_sample(sample);
            if ((i + 1) % 500 == 0) {
                std::cout << "  Processed " << (i + 1) << " samples\n";
            }
        }
        const float hat_train_acc = evaluate_accuracy(hat, train);
        const float hat_test_acc = evaluate_accuracy(hat, test);
        print_metric("HAT", hat_train_acc, hat_test_acc);

        tree::SgtConfig sgt_cfg;
        sgt_cfg.delta = 1e-7f;
        sgt_cfg.grace_period = 32;
        sgt_cfg.min_samples_split = 64;
        sgt_cfg.tie_threshold = 0.05f;
        sgt_cfg.num_threshold_bins = 16;
        sgt_cfg.max_depth = 16;
        sgt_cfg.init_pred = 0.0f;
        sgt_cfg.lambda_value = 0.1f;
        sgt_cfg.gamma = 1.0f;
        sgt_cfg.std_prop = 1.0f;
        sgt_cfg.warm_start = 20;

        tree::SgtModel sgt(input_dim, sgt_cfg);
        std::cout << "Training SGT...\n";
        for (std::size_t i = 0; i < train.x.size(); ++i) {
            bnn::Sample sample{train.x[i], train.y[i]};
            sgt.train_sample(sample);
            if ((i + 1) % 500 == 0) {
                std::cout << "  Processed " << (i + 1) << " samples\n";
            }
        }
        const float sgt_train_acc = evaluate_accuracy(sgt, train);
        const float sgt_test_acc = evaluate_accuracy(sgt, test);
        print_metric("SGT", sgt_train_acc, sgt_test_acc);

        tree::HoeffdingTreeConfig ht_cfg;
        ht_cfg.num_classes = num_classes;
        ht_cfg.delta = 1e-7f;
        ht_cfg.grace_period = 32;
        ht_cfg.min_samples_split = 64;
        ht_cfg.tie_threshold = 0.05f;
        ht_cfg.num_threshold_bins = 16;
        ht_cfg.max_depth = 16;
        ht_cfg.max_size_mib = 10.0f;

        tree::HoeffdingTreeModel ht(input_dim, ht_cfg);
        std::cout << "Training HOEFFDING...\n";
        for (std::size_t i = 0; i < train.x.size(); ++i) {
            bnn::Sample sample{train.x[i], train.y[i]};
            ht.train_sample(sample);
            if ((i + 1) % 500 == 0) {
                std::cout << "  Processed " << (i + 1) << " samples\n";
            }
        }
        const float ht_train_acc = evaluate_accuracy(ht, train);
        const float ht_test_acc = evaluate_accuracy(ht, test);
        print_metric("HOEFF", ht_train_acc, ht_test_acc);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }
}
