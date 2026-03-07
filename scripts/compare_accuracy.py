#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare C++ HoeffdingTreeModel vs River HoeffdingTreeClassifier accuracy.

Usage:
  python3 scripts/compare_accuracy.py data/UKMNCT_IIoT_FDIA.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Tuple

try:
    from river import tree
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "river is required for this script. Install with: pip install river"
    ) from exc


def parse_label(raw: str, dynamic_map: Dict[str, int]) -> int:
    token = raw.strip()
    try:
        numeric = float(token)
        return int(round(numeric))
    except Exception:
        pass

    lowered = token.lower()
    if lowered in {"attack", "malicious", "anomaly", "abnormal"}:
        return 1
    if lowered in {"natural", "normal", "benign"}:
        return 0

    if lowered in dynamic_map:
        return dynamic_map[lowered]

    next_value = len(dynamic_map)
    dynamic_map[lowered] = next_value
    return next_value


def load_csv_dataset(path: str) -> Tuple[List[Dict[str, float]], List[int]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("Dataset must have at least one feature and one label column")

        label_idx = len(header) - 1
        for i, name in enumerate(header):
            lowered = name.strip().lower()
            if lowered in {"label", "target", "class", "marker"}:
                label_idx = i
                break

        dynamic_map: Dict[str, int] = {}
        features: List[Dict[str, float]] = []
        labels: List[int] = []

        for row in reader:
            if not row or len(row) != len(header):
                continue
            try:
                label = parse_label(row[label_idx], dynamic_map)
            except Exception:
                continue

            record: Dict[str, float] = {}
            valid = True
            for i, value in enumerate(row):
                if i == label_idx:
                    continue
                try:
                    record[header[i]] = float(value)
                except Exception:
                    valid = False
                    break
            if not valid:
                continue

            features.append(record)
            labels.append(label)

    if not features:
        raise ValueError("No valid rows parsed from dataset")

    return features, labels


def split_train_test(x: List[Dict[str, float]], y: List[int], train_frac: float, seed: int):
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")

    idx = list(range(len(x)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    split = int(math.floor(len(idx) * train_frac))
    tr_idx = idx[:split]
    te_idx = idx[split:]

    x_train = [x[i] for i in tr_idx]
    y_train = [y[i] for i in tr_idx]
    x_test = [x[i] for i in te_idx]
    y_test = [y[i] for i in te_idx]

    return (x_train, y_train), (x_test, y_test)


def evaluate_accuracy(model, x: List[Dict[str, float]], y: List[int]) -> float:
    correct = 0
    for xi, yi in zip(x, y):
        pred = model.predict_one(xi)
        if pred == yi:
            correct += 1
    return correct / float(len(y)) if y else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare River HoeffdingTree accuracy.")
    parser.add_argument("csv_path")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    x, y = load_csv_dataset(args.csv_path)
    (x_train, y_train), (x_test, y_test) = split_train_test(x, y, args.train_frac, args.seed)

    if args.max_train > 0:
        x_train = x_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_test > 0:
        x_test = x_test[: args.max_test]
        y_test = y_test[: args.max_test]

    num_classes = max(y_train + y_test) + 1

    model = tree.HoeffdingTreeClassifier(
        grace_period=32,
        delta=1e-7,
        tau=0.05,
        max_depth=16,
        leaf_prediction="mc",
        split_criterion="info_gain",
    )

    for xi, yi in zip(x_train, y_train):
        model.learn_one(xi, yi)

    train_acc = evaluate_accuracy(model, x_train, y_train)
    test_acc = evaluate_accuracy(model, x_test, y_test)

    print(f"Dataset: {args.csv_path}")
    print(f"Train: {len(x_train)} Test: {len(x_test)} Classes: {num_classes}")
    print(f"RIVER HT train_acc={train_acc:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
