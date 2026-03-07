#!/usr/bin/env python3
"""
Run test_serial.py for multiple runtime-selectable models using one firmware flash.
Supports transport comparisons and fairness profiles in one sweep.

Examples:
  python run_all_models.py --models ALL -p COM12 --dataset IoT_clean
  python run_all_models.py --tm-only -p COM12 --dataset IoT_clean

  python run_all_models.py --models EFDT,HOEFFDING,HAT,SGT,BNN,TM_SPARSE,TM_VANILLA,TM_BO \
    -p COM12 -b 2000000 --dataset IoT_clean --runs 1 --train-samples 3000 --test-samples 1000 \
    --window 1024 --batch-records 1024 --tree-epochs 1 --nn-epochs 20

  python run_all_models.py --transport both --skip-flash --models TM_SPARSE,TM_BO \
    -p COM12 --dataset IoT_clean -- --final-logs

  # Explicit transport comparison (with and without --csv-bitpack)
  python run_all_models.py --compare-csv-bitpack --models ALL -p COM12 --dataset IoT_clean

  # Fairness presets:
  # online: all models use 1 epoch
  # equal-budget: all models use same epoch budget (default 20)
  # both: run both presets in one command
  python run_all_models.py --models ALL --fairness both --fair-epochs 20 -p COM12 --dataset IoT_clean
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_TO_RUNTIME: Dict[str, str] = {
    "EFDT": "efdt",
    "HOEFFDING": "hoeffding",
    "HAT": "hat",
    "SGT": "sgt",
    "BNN": "bnn",
    "TM_SPARSE": "tm_sparse",
    "TM_VANILLA": "tm_vanilla",
    "TM_BO": "tm_bo",
}

TREE_MODELS = {"EFDT", "HOEFFDING", "HAT", "SGT"}
TM_MODELS = ["TM_SPARSE", "TM_VANILLA", "TM_BO"]
TM_DEFAULT_SPECIFICITY = 8
TM_BO_DEFAULT_SPECIFICITY = 14
ResultRow = Dict[str, Any]

BOARD_PROFILES: Dict[str, Dict[str, Any]] = {
    "esp32_c3_mini": {
        "pio_env": "esp32_c3_mini_fast_tm_all",
        "baud": 2000000,
        "window": 64,
        "batch_records": 64,
    },
    "esp32_s3_n16r8": {
        "pio_env": "esp32_s3_n16r8_fast_tm_all",
        "baud": 2000000,
        "window": 1024,
        "batch_records": 1024,
    },
    "esp32_p4_firebeetle": {
        "pio_env": "esp32_p4_firebeetle_fast_tm_all",
        "baud": 2000000,
        "window": 1024,
        "batch_records": 1024,
    },
    "esp32_p4_nano": {
        "pio_env": "esp32_p4_nano_fast_tm_all",
        "baud": 6000000,
        "window": 1024,
        "batch_records": 1024,
    },
    "esp32_p4_nano_cp2102": {
        "pio_env": "esp32_p4_nano_fast_tm_all_cp2102",
        "baud": 115200,
        "window": 1024,
        "batch_records": 1024,
    },
}

BOARD_PROFILE_ALIASES: Dict[str, str] = {
    "c3": "esp32_c3_mini",
    "esp32_c3": "esp32_c3_mini",
    "s3": "esp32_s3_n16r8",
    "esp32_s3": "esp32_s3_n16r8",
    "firebeetle": "esp32_p4_firebeetle",
    "p4_firebeetle": "esp32_p4_firebeetle",
    "p4_nano": "esp32_p4_nano",
    "nano": "esp32_p4_nano",
    "esp32-p4_nano": "esp32_p4_nano",
    "p4nano": "esp32_p4_nano",
    "esp32_p4_nano_cp2102": "esp32_p4_nano_cp2102",
    "esp32-p4_nano_cp2102": "esp32_p4_nano_cp2102",
    "p4_nano_cp2102": "esp32_p4_nano_cp2102",
    "cp2102": "esp32_p4_nano_cp2102",
}


def normalize_board_profile(raw: str) -> str:
    key = (raw or "").strip().lower()
    return BOARD_PROFILE_ALIASES.get(key, key)


def apply_board_profile(args: "argparse.Namespace") -> None:
    if not args.board_profile:
        return
    profile_name = normalize_board_profile(args.board_profile)
    profile = BOARD_PROFILES.get(profile_name)
    if profile is None:
        valid = ", ".join(sorted(BOARD_PROFILES.keys()))
        raise SystemExit(f"Unknown --board-profile '{args.board_profile}'. Valid: {valid}")
    args.board_profile = profile_name
    args.pio_env = profile["pio_env"]
    args.baud = int(profile["baud"])
    args.window = int(profile["window"])
    args.batch_records = int(profile["batch_records"])


def parse_models(raw: str) -> List[str]:
    key = raw.strip().upper()
    if key == "ALL":
        return list(MODEL_TO_RUNTIME.keys())
    if key in ("TM_ONLY", "TM"):
        return TM_MODELS.copy()
    models = [m.strip().upper() for m in raw.split(",") if m.strip()]
    invalid = [m for m in models if m not in MODEL_TO_RUNTIME]
    if invalid:
        raise SystemExit(
            f"Unknown model(s): {', '.join(invalid)}. "
            f"Valid: {', '.join(MODEL_TO_RUNTIME.keys())}, ALL, or TM_ONLY"
        )
    return models


def run_cmd(cmd: List[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def build_common_test_args(args: argparse.Namespace) -> List[str]:
    out = [
        "-p", args.port,
        "-b", str(args.baud),
        "--dataset", args.dataset,
        "--runs", str(args.runs),
        "--window", str(args.window),
        "--batch-records", str(args.batch_records),
        "--throttle-ms", str(args.throttle_ms),
        "--status-every", str(args.status_every),
    ]
    if args.train_samples is not None:
        out += ["--train-samples", str(args.train_samples)]
    if args.test_samples is not None:
        out += ["--test-samples", str(args.test_samples)]
    if args.input_real:
        out.append("--input-real")
    if args.unsafe_no_data_checksum:
        out.append("--unsafe-no-data-checksum")
    if args.no_hw_reset:
        out.append("--no-hw-reset")
    if args.final_logs:
        out.append("--final-logs")
    if args.snapshot:
        out.append("--snapshot")
    if getattr(args, "js_power", False):
        out.append("--js-power")
    if getattr(args, "js_serial", None):
        out += ["--js-serial", str(args.js_serial)]
    if getattr(args, "js_poll_ms", None) is not None:
        out += ["--js-poll-ms", str(args.js_poll_ms)]
    return out


def build_tm_runtime_args(args: argparse.Namespace, model: str = "") -> List[str]:
    if getattr(args, "tm_specificity", None) is None:
        specificity = TM_BO_DEFAULT_SPECIFICITY if model == "TM_BO" else TM_DEFAULT_SPECIFICITY
    else:
        specificity = args.tm_specificity
    return [
        "--tm-clauses", str(args.tm_clauses),
        "--tm-threshold", str(args.tm_threshold),
        "--tm-specificity", str(specificity),
        "--tm-seed", str(args.tm_seed),
        "--tm-init-density", str(args.tm_init_density),
    ]


def epochs_for_model(model: str, args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return args.epochs
    if model in TREE_MODELS:
        return args.tree_epochs
    return args.nn_epochs


def fairness_profiles(args: argparse.Namespace) -> List[Tuple[str, Optional[int]]]:
    """
    Return fairness profiles as (profile_name, forced_epochs).
    forced_epochs=None means default mixed schedule (tree-epochs/nn-epochs or --epochs).
    """
    equal_budget_epochs = args.epochs if args.epochs is not None else args.fair_epochs
    if args.fairness == "none":
        return [("default", None)]
    if args.fairness == "online":
        return [("fair_online", 1)]
    if args.fairness == "equal-budget":
        return [("fair_budget", equal_budget_epochs)]
    return [("fair_online", 1), ("fair_budget", equal_budget_epochs)]


def epochs_for_profile(model: str, args: argparse.Namespace, forced_epochs: Optional[int]) -> int:
    if forced_epochs is not None:
        return int(forced_epochs)
    return epochs_for_model(model, args)


def transport_modes(args: argparse.Namespace) -> List[Tuple[str, List[str]]]:
    """
    Return selected transport variants as (mode_name, extra_cli_args) tuples.
    """
    if args.transport == "bitpack":
        return [("bitpack", ["--csv-bitpack"])]
    if args.transport == "raw":
        return [("raw", [])]
    return [("bitpack", ["--csv-bitpack"]), ("raw", [])]


def latest_result_for_tag(tag: str) -> Optional[Path]:
    pattern = f"results/experiment_{tag}_*.json"
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return (None, None)
    mean_v = statistics.mean(values)
    std_v = statistics.stdev(values) if len(values) > 1 else 0.0
    return (mean_v, std_v)


def _extract_stats_metric(stats: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[float]]:
    node = stats.get(key)
    if not isinstance(node, dict):
        return (None, None)
    mean_v = node.get("mean")
    std_v = node.get("std")
    mean_f = float(mean_v) if isinstance(mean_v, (int, float)) else None
    std_f = float(std_v) if isinstance(std_v, (int, float)) else None
    return (mean_f, std_f)


def _extract_nested_float(root: Dict[str, Any], path: List[str]) -> Optional[float]:
    cur: Any = root
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def extract_metrics(result_file: Path) -> Dict[str, Any]:
    data = json.loads(result_file.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        return {
            "throughput_mean": None,
            "throughput_std": None,
            "latency_mean_ms": None,
            "latency_std_ms": None,
            "final_test_mean": None,
            "final_test_std": None,
            "final_train_mean": None,
            "final_train_std": None,
            "best_acc_mean": None,
            "duration_mean_sec": None,
            "samples_sent": 0,
            "bytes_sent": 0,
            "bytes_per_sample": None,
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "precision_mean": None,
            "precision_std": None,
            "recall_mean": None,
            "recall_std": None,
            "f1_mean": None,
            "f1_std": None,
            "auc_mean": None,
            "auc_std": None,
        }

    throughputs: List[float] = []
    latencies: List[float] = []
    final_test_accs: List[float] = []
    final_train_accs: List[float] = []
    best_accs: List[float] = []
    durations: List[float] = []
    samples_sent = 0
    bytes_sent = 0
    successful_runs = 0

    for run in runs:
        summary = run.get("summary", {})
        if summary.get("failed", False):
            continue
        successful_runs += 1

        thr = summary.get("avg_throughput_sps")
        if isinstance(thr, (int, float)):
            thr_f = float(thr)
            throughputs.append(thr_f)
            lat = summary.get("avg_latency_ms")
            if isinstance(lat, (int, float)) and float(lat) > 0:
                latencies.append(float(lat))
            elif thr_f > 0:
                latencies.append(1000.0 / thr_f)

        fta = summary.get("final_test_accuracy")
        if isinstance(fta, (int, float)):
            final_test_accs.append(float(fta))

        fra = summary.get("final_train_accuracy")
        if isinstance(fra, (int, float)):
            final_train_accs.append(float(fra))

        best_acc = summary.get("best_accuracy")
        if isinstance(best_acc, (int, float)):
            best_accs.append(float(best_acc))

        dur = summary.get("total_duration_sec")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))

        n = summary.get("total_samples_sent")
        if isinstance(n, int):
            samples_sent = max(samples_sent, n)

        b = summary.get("total_bytes_sent")
        if isinstance(b, int):
            bytes_sent = max(bytes_sent, b)

    thr_mean, thr_std = _mean_std(throughputs)
    lat_mean, lat_std = _mean_std(latencies)
    test_mean, test_std = _mean_std(final_test_accs)
    train_mean, train_std = _mean_std(final_train_accs)
    best_mean, _ = _mean_std(best_accs)
    duration_mean, _ = _mean_std(durations)
    bytes_per_sample = (bytes_sent / samples_sent) if samples_sent > 0 else None

    stats = data.get("statistics", {})
    precision_mean, precision_std = _extract_stats_metric(stats, "precision")
    recall_mean, recall_std = _extract_stats_metric(stats, "recall")
    f1_mean, f1_std = _extract_stats_metric(stats, "f1_score")

    auc_mean: Optional[float] = None
    auc_std: Optional[float] = None
    for auc_key in ("auc", "roc_auc", "auroc", "pr_auc", "average_precision"):
        auc_mean, auc_std = _extract_stats_metric(stats, auc_key)
        if auc_mean is not None:
            break

    # Fallbacks from comprehensive analysis payload for single-run summaries.
    if precision_mean is None:
        precision_mean = _extract_nested_float(
            data, ["comprehensive_analysis", "performance_benchmarks", "precision"]
        )
        precision_std = 0.0 if precision_mean is not None else None
    if recall_mean is None:
        recall_mean = _extract_nested_float(
            data, ["comprehensive_analysis", "performance_benchmarks", "recall"]
        )
        recall_std = 0.0 if recall_mean is not None else None
    if f1_mean is None:
        f1_mean = _extract_nested_float(
            data, ["comprehensive_analysis", "performance_benchmarks", "f1_score"]
        )
        f1_std = 0.0 if f1_mean is not None else None
    if auc_mean is None:
        auc_mean = _extract_nested_float(
            data, ["comprehensive_analysis", "performance_benchmarks", "auc"]
        )
        auc_std = 0.0 if auc_mean is not None else None

    return {
        "throughput_mean": thr_mean,
        "throughput_std": thr_std,
        "latency_mean_ms": lat_mean,
        "latency_std_ms": lat_std,
        "final_test_mean": test_mean,
        "final_test_std": test_std,
        "final_train_mean": train_mean,
        "final_train_std": train_std,
        "best_acc_mean": best_mean,
        "duration_mean_sec": duration_mean,
        "samples_sent": samples_sent,
        "bytes_sent": bytes_sent,
        "bytes_per_sample": bytes_per_sample,
        "total_runs": len(runs),
        "successful_runs": successful_runs,
        "failed_runs": len(runs) - successful_runs,
        "precision_mean": precision_mean,
        "precision_std": precision_std,
        "recall_mean": recall_mean,
        "recall_std": recall_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
    }


def _git_read(args: List[str], fallback: str = "unknown") -> str:
    try:
        cp = subprocess.run(["git"] + args, capture_output=True, text=True, check=False)
    except Exception:
        return fallback
    if cp.returncode != 0:
        return fallback
    out = cp.stdout.strip()
    return out if out else fallback


def ensure_qa_file(path: Path) -> None:
    header = (
        "| Run UTC | Git SHA | Branch | Fairness | Model | Mode | Dataset | Epochs | Train/Test | "
        "Window | Batch | Runs | Throughput (smp/s) | Latency (ms) | Best Acc | Test Acc | Train Acc | "
        "Precision | Recall | F1 | AUC | Gen Gap | Duration (s) | Bytes/Sample | Samples | Status | Result File |"
    )
    separator = (
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    if not path.exists():
        content = """# QA Accuracy And Throughput Spec

## Purpose
Track throughput (`smp/s`) and accuracy for every benchmark run to prevent silent quality regressions.

## Non-Regression Policy
1. We must never regress on accuracy or throughput for the same configuration without explicit written approval.
2. Any regression is a release blocker until fixed or explicitly approved and documented.
3. Every benchmark run must be logged in the table below.
4. Comparisons must use the same configuration key:
   `model + mode + dataset + epochs + train/test samples + window + batch + checksum mode`.

## Required Metrics
1. `Throughput (smp/s)`
2. `Final test accuracy`
3. `Final train accuracy`
"""
        path.write_text(content, encoding="utf-8")

    existing = path.read_text(encoding="utf-8")
    if header not in existing:
        with path.open("a", encoding="utf-8") as f:
            if not existing.endswith("\n"):
                f.write("\n")
            f.write("\n## Run Log v2\n")
            f.write(header + "\n")
            f.write(separator + "\n")


def append_qa_log(qa_file: Path, args: argparse.Namespace, results: List[ResultRow]) -> None:
    ensure_qa_file(qa_file)
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sha = _git_read(["rev-parse", "--short", "HEAD"])
    branch = _git_read(["branch", "--show-current"])

    rows: List[str] = []
    for row in results:
        thr = row.get("throughput_sps")
        lat = row.get("latency_ms")
        test_acc = row.get("test_acc")
        train_acc = row.get("train_acc")
        precision = row.get("precision")
        recall = row.get("recall")
        f1 = row.get("f1")
        auc = row.get("auc")
        best_acc = row.get("best_acc")
        gen_gap = row.get("gen_gap")
        duration = row.get("duration_sec")
        bps = row.get("bytes_per_sample")
        samples = row.get("samples", 0)
        result_file = row.get("result_file")

        thr_s = f"{thr:.1f}" if isinstance(thr, (int, float)) else "-"
        lat_s = f"{lat:.4f}" if isinstance(lat, (int, float)) else "-"
        best_s = f"{best_acc:.4f}" if isinstance(best_acc, (int, float)) else "-"
        test_s = f"{test_acc:.4f}" if isinstance(test_acc, (int, float)) else "-"
        train_s = f"{train_acc:.4f}" if isinstance(train_acc, (int, float)) else "-"
        prec_s = f"{precision:.4f}" if isinstance(precision, (int, float)) else "-"
        recall_s = f"{recall:.4f}" if isinstance(recall, (int, float)) else "-"
        f1_s = f"{f1:.4f}" if isinstance(f1, (int, float)) else "-"
        auc_s = f"{auc:.4f}" if isinstance(auc, (int, float)) else "-"
        gap_s = f"{gen_gap:.4f}" if isinstance(gen_gap, (int, float)) else "-"
        dur_s = f"{duration:.3f}" if isinstance(duration, (int, float)) else "-"
        bps_s = f"{bps:.2f}" if isinstance(bps, (int, float)) else "-"
        rf = result_file.name if isinstance(result_file, Path) else "-"
        runs_total = row.get("runs_total", args.runs)
        runs_ok = row.get("runs_ok", 0)
        runs_s = f"{runs_ok}/{runs_total}"

        rows.append(
            f"| {run_utc} | `{sha}` | `{branch}` | {row.get('fairness')} | {row.get('model')} | "
            f"{row.get('mode')} | {args.dataset} | {row.get('epochs')} | "
            f"{args.train_samples}/{args.test_samples} | {args.window} | {args.batch_records} | {runs_s} | "
            f"{thr_s} | {lat_s} | {best_s} | {test_s} | {train_s} | {prec_s} | {recall_s} | {f1_s} | {auc_s} | "
            f"{gap_s} | {dur_s} | {bps_s} | {samples} | {row.get('status')} | `{rf}` |"
        )

    with qa_file.open("a", encoding="utf-8") as f:
        for line in rows:
            f.write(line + "\n")


def _fmt_float(v: Any, digits: int) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.{digits}f}"
    return "-"


def _fmt_std(mean: Any, std: Any, digits: int) -> str:
    if not isinstance(mean, (int, float)):
        return "-"
    if isinstance(std, (int, float)) and std > 0:
        return f"{mean:.{digits}f}+/-{std:.{digits}f}"
    return f"{mean:.{digits}f}"


def _print_summary(results: List[ResultRow]) -> None:
    print("\nSummary:")
    print(
        "FAIRNESS      MODEL        MODE      STATUS       "
        "THR(smp/s)     LAT(ms)   BEST_ACC   TEST_ACC   TRAIN_ACC   PREC      REC       F1       AUC      GAP    DUR(s)    B/S   RUNS  EPOCHS  FILE"
    )
    auc_available = False
    for row in results:
        thr_s = _fmt_std(row.get("throughput_sps"), row.get("throughput_std"), 1)
        lat_s = _fmt_std(row.get("latency_ms"), row.get("latency_std_ms"), 4)
        best_s = _fmt_float(row.get("best_acc"), 4)
        test_s = _fmt_std(row.get("test_acc"), row.get("test_acc_std"), 4)
        train_s = _fmt_std(row.get("train_acc"), row.get("train_acc_std"), 4)
        prec_s = _fmt_std(row.get("precision"), row.get("precision_std"), 4)
        rec_s = _fmt_std(row.get("recall"), row.get("recall_std"), 4)
        f1_s = _fmt_std(row.get("f1"), row.get("f1_std"), 4)
        auc_s = _fmt_std(row.get("auc"), row.get("auc_std"), 4)
        if isinstance(row.get("auc"), (int, float)):
            auc_available = True
        gap_s = _fmt_float(row.get("gen_gap"), 4)
        dur_s = _fmt_float(row.get("duration_sec"), 3)
        bps_s = _fmt_float(row.get("bytes_per_sample"), 2)
        file_s = row["result_file"].name if isinstance(row.get("result_file"), Path) else "-"
        runs_s = f"{row.get('runs_ok', 0)}/{row.get('runs_total', 0)}"
        print(
            f"{str(row.get('fairness')):13s} {str(row.get('model')):12s} {str(row.get('mode')):9s} "
            f"{str(row.get('status')):12s} {thr_s:>12s}  {lat_s:>10s}  {best_s:>9s}  {test_s:>9s}  "
            f"{train_s:>10s}  {prec_s:>8s}  {rec_s:>8s}  {f1_s:>8s}  {auc_s:>8s}  "
            f"{gap_s:>7s}  {dur_s:>8s}  {bps_s:>5s}  {runs_s:>4s}  {int(row.get('epochs', 0)):6d}  {file_s}"
        )
    if not auc_available:
        print("NOTE: AUC is not present in current experiment JSON payloads; showing '-' until producer exports AUC.")

def _status_ok(status: str) -> bool:
    return status in ("ok", "dry_run")


def _build_tag(tag_prefix: str, fairness: str, runtime_model: str, mode_name: str) -> str:
    if fairness == "default":
        return f"{tag_prefix}{runtime_model}_{mode_name}"
    return f"{tag_prefix}{fairness}_{runtime_model}_{mode_name}"


def _result_template(
    *,
    fairness: str,
    model: str,
    mode_name: str,
    epochs: int,
    status: str,
    rc: int = 0,
    result_file: Optional[Path] = None,
) -> ResultRow:
    return {
        "fairness": fairness,
        "model": model,
        "mode": mode_name,
        "status": status,
        "rc": rc,
        "result_file": result_file,
        "throughput_sps": None,
        "throughput_std": None,
        "latency_ms": None,
        "latency_std_ms": None,
        "test_acc": None,
        "test_acc_std": None,
        "train_acc": None,
        "train_acc_std": None,
        "precision": None,
        "precision_std": None,
        "recall": None,
        "recall_std": None,
        "f1": None,
        "f1_std": None,
        "auc": None,
        "auc_std": None,
        "best_acc": None,
        "duration_sec": None,
        "samples": 0,
        "bytes_sent": 0,
        "bytes_per_sample": None,
        "gen_gap": None,
        "runs_ok": 0,
        "runs_total": 0,
        "epochs": epochs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all models using runtime model selection")
    ap.add_argument("--models", type=str, default="ALL", help="Comma list of models, ALL, or TM_ONLY")
    ap.add_argument("--tm-only", action="store_true",
                    help="Shortcut for --models TM_ONLY (TM_SPARSE, TM_VANILLA, TM_BO)")
    ap.add_argument("--board-profile", type=str, default=None,
                    help="Apply board defaults (pio env, baud, window, batch). "
                         "Options: esp32_c3_mini, esp32_s3_n16r8, esp32_p4_firebeetle, "
                         "esp32_p4_nano, esp32_p4_nano_cp2102 (aliases: c3, s3, nano, firebeetle, cp2102)")
    _default_port = "COM12" if sys.platform == "win32" else "/dev/ttyUSB0"
    ap.add_argument("-p", "--port", type=str, default=_default_port, help="Serial port")
    ap.add_argument("-b", "--baud", type=int, default=2000000, help="Serial baud")
    ap.add_argument("--dataset", type=str, default="IoT_clean", help="Dataset name")
    ap.add_argument("--runs", type=int, default=1, help="Runs per model")
    ap.add_argument("--train-samples", type=int, default=None, help="Training samples per epoch cap (default: full train set)")
    ap.add_argument("--test-samples", type=int, default=None, help="Test samples per epoch cap (default: full test set)")
    ap.add_argument("--window", type=int, default=1024, help="Transport window size")
    ap.add_argument("--batch-records", type=int, default=1024, help="Batch records per frame")
    ap.add_argument("--throttle-ms", type=int, default=0, help="Sender throttle in ms")
    ap.add_argument("--status-every", type=float, default=0.0, help="Status interval seconds")
    ap.add_argument("--epochs", type=int, default=None, help="Force same epochs for all models")
    ap.add_argument("--tree-epochs", type=int, default=1, help="Epochs for tree models")
    ap.add_argument("--nn-epochs", type=int, default=20, help="Epochs for BNN/TM models")
    ap.add_argument("--tm-clauses", type=int, default=100, help="Runtime TM clauses (host config, no reflash)")
    ap.add_argument("--tm-threshold", type=int, default=10, help="Runtime TM threshold T (host config, no reflash)")
    ap.add_argument("--tm-specificity", type=int, default=None,
                    help=f"Runtime TM specificity s. Default: {TM_DEFAULT_SPECIFICITY} for TM_SPARSE/TM_VANILLA, {TM_BO_DEFAULT_SPECIFICITY} for TM_BO")
    ap.add_argument("--tm-seed", type=int, default=1, help="Runtime TM seed (host config, no reflash)")
    ap.add_argument("--tm-init-density", type=int, default=100,
                    help="Runtime TM sparse/BO init literal density pct [0..100] (host config, no reflash)")
    ap.add_argument("--fairness", choices=["none", "online", "equal-budget", "both"], default="none",
                    help="Fairness scheduling: none (mixed), online (all=1), equal-budget (all same), both")
    ap.add_argument("--fair-epochs", type=int, default=20,
                    help="Epoch budget used by equal-budget/both (unless --epochs is set)")
    ap.add_argument("--transport", choices=["both", "bitpack", "raw"], default="both",
                    help="Transport mode sweep: both (default), bitpack-only, or raw-only")
    ap.add_argument("--csv-bitpack", dest="transport", action="store_const", const="bitpack",
                    help="Compatibility alias for --transport bitpack")
    ap.add_argument("--no-csv-bitpack", dest="transport", action="store_const", const="raw",
                    help="Compatibility alias for --transport raw")
    ap.add_argument("--compare-csv-bitpack", dest="transport", action="store_const", const="both",
                    help="Run both transport modes: with and without --csv-bitpack")
    ap.add_argument("--input-real", action="store_true", help="Use input-real path")
    ap.add_argument("--unsafe-no-data-checksum", action="store_true", default=False,
                    help="Disable data checksum (default: off)")
    ap.add_argument("--safe-data-checksum", action="store_false", dest="unsafe_no_data_checksum",
                    help="Enable data checksum")
    ap.add_argument("--no-hw-reset", action="store_true", help="Pass --no-hw-reset to test_serial.py")
    ap.add_argument("--final-logs", action="store_true", help="Pass --final-logs to test_serial.py")
    ap.add_argument("--snapshot", action="store_true", help="Pass --snapshot to test_serial.py")
    ap.add_argument("--js-power", action="store_true", help="Enable Joulescope train/test phase metrics")
    ap.add_argument("--js-serial", type=str, default=None, help="Optional Joulescope serial number")
    ap.add_argument("--js-poll-ms", type=float, default=100.0, help="Joulescope poll interval ms")
    ap.add_argument("--seed-mode", choices=["fixed", "random"], default="random",
                    help="Seed policy for per-model runs: fixed (reproducible) or random (default)")
    ap.add_argument("--base-seed", type=int, default=42,
                    help="Base shuffle seed used when --seed-mode fixed")
    ap.add_argument("--pio-env", type=str, default="esp32_p4_nano_fast_tm_all", help="PlatformIO env to flash once")
    ap.add_argument("--upload-port", type=str, default=None, help="Optional explicit upload port for pio")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    ap.add_argument("--skip-flash", action="store_true", help="Skip firmware upload")
    ap.add_argument("--tag-prefix", type=str, default="runtime_suite_", help="Prefix for model-tag in result files")
    ap.add_argument("--qa-file", type=str, default="QA_ACCURACY.MD", help="QA ledger markdown file path")
    ap.add_argument("--no-qa-log", action="store_true", help="Disable QA ledger append")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    baud_explicit = "-b" in sys.argv or "--baud" in sys.argv
    user_baud = args.baud
    apply_board_profile(args)
    if baud_explicit:
        args.baud = user_baud

    if args.tm_only:
        args.models = "TM_ONLY"
    models = parse_models(args.models)

    test_script = Path("test_serial.py")
    if not test_script.exists():
        raise SystemExit("test_serial.py not found in current directory")

    common_test_args = build_common_test_args(args)
    print("Common test_serial args:")
    print("  " + " ".join(common_test_args))
    if passthrough:
        print("Extra passthrough args:")
        print("  " + " ".join(passthrough))
    print(
        "TM runtime args (for TM models): "
        f"C={args.tm_clauses}, T={args.tm_threshold}, s={args.tm_specificity}, "
        f"seed={args.tm_seed}, init_density={args.tm_init_density}"
    )

    if not args.skip_flash:
        flash_cmd = ["pio", "run", "-e", args.pio_env, "-t", "upload"]
        if args.upload_port:
            flash_cmd += ["--upload-port", args.upload_port]
        print("=" * 80)
        print(f"FLASH ONCE: {args.pio_env}")
        print("=" * 80)
        if args.dry_run:
            print(f"$ {' '.join(flash_cmd)}")
        else:
            rc = run_cmd(flash_cmd)
            if rc != 0:
                print(f"[FAIL] Flash failed (rc={rc})")
                return 1
            time.sleep(1.0)

    modes = transport_modes(args)
    profiles = fairness_profiles(args)
    print("Transport modes:")
    print("  " + ", ".join(name for name, _ in modes))
    print("Fairness profiles:")
    print("  " + ", ".join(name for name, _ in profiles))

    results: List[ResultRow] = []
    failed = False

    for fairness_name, forced_epochs in profiles:
        print("\n" + "*" * 80)
        print(f"FAIRNESS PROFILE: {fairness_name}")
        print("*" * 80)

        for mode_name, mode_args in modes:
            print("\n" + "#" * 80)
            print(f"TRANSPORT MODE: {mode_name}")
            print("#" * 80)

            for model in models:
                runtime_model = MODEL_TO_RUNTIME[model]
                per_model_epochs = epochs_for_profile(model, args, forced_epochs)
                tag = _build_tag(args.tag_prefix, fairness_name, runtime_model, mode_name)

                print("=" * 80)
                print(
                    f"MODEL: {model} "
                    f"(runtime='{runtime_model}', epochs={per_model_epochs}, mode={mode_name}, fairness={fairness_name})"
                )
                print("=" * 80)

                cmd = [
                    args.python,
                    str(test_script),
                    "--model", runtime_model,
                    "--model-tag", tag,
                    "--epochs", str(per_model_epochs),
                ] + common_test_args + mode_args + passthrough
                if model in TM_MODELS:
                    cmd += build_tm_runtime_args(args)

                if args.dry_run:
                    print(f"$ {' '.join(cmd)}")
                    row = _result_template(
                        fairness=fairness_name,
                        model=model,
                        mode_name=mode_name,
                        epochs=per_model_epochs,
                        status="dry_run",
                        rc=0,
                    )
                    row["runs_total"] = args.runs
                    results.append(row)
                    continue

                rc = run_cmd(cmd)
                if rc != 0:
                    failed = True
                    row = _result_template(
                        fairness=fairness_name,
                        model=model,
                        mode_name=mode_name,
                        epochs=per_model_epochs,
                        status="run_failed",
                        rc=rc,
                    )
                    row["runs_total"] = args.runs
                    results.append(row)
                    continue

                result_file = latest_result_for_tag(tag)
                if result_file is None:
                    failed = True
                    row = _result_template(
                        fairness=fairness_name,
                        model=model,
                        mode_name=mode_name,
                        epochs=per_model_epochs,
                        status="missing_result",
                        rc=2,
                    )
                    row["runs_total"] = args.runs
                    results.append(row)
                    continue

                metrics = extract_metrics(result_file)
                train_acc = metrics["final_train_mean"]
                test_acc = metrics["final_test_mean"]
                gen_gap = None
                if isinstance(train_acc, (int, float)) and isinstance(test_acc, (int, float)):
                    gen_gap = float(train_acc) - float(test_acc)

                status = "ok" if metrics["successful_runs"] > 0 else "all_failed"
                if status != "ok":
                    failed = True

                row = _result_template(
                    fairness=fairness_name,
                    model=model,
                    mode_name=mode_name,
                    epochs=per_model_epochs,
                    status=status,
                    rc=0,
                    result_file=result_file,
                )
                row["throughput_sps"] = metrics["throughput_mean"]
                row["throughput_std"] = metrics["throughput_std"]
                row["latency_ms"] = metrics["latency_mean_ms"]
                row["latency_std_ms"] = metrics["latency_std_ms"]
                row["test_acc"] = test_acc
                row["test_acc_std"] = metrics["final_test_std"]
                row["train_acc"] = train_acc
                row["train_acc_std"] = metrics["final_train_std"]
                row["precision"] = metrics["precision_mean"]
                row["precision_std"] = metrics["precision_std"]
                row["recall"] = metrics["recall_mean"]
                row["recall_std"] = metrics["recall_std"]
                row["f1"] = metrics["f1_mean"]
                row["f1_std"] = metrics["f1_std"]
                row["auc"] = metrics["auc_mean"]
                row["auc_std"] = metrics["auc_std"]
                row["best_acc"] = metrics["best_acc_mean"]
                row["duration_sec"] = metrics["duration_mean_sec"]
                row["samples"] = metrics["samples_sent"]
                row["bytes_sent"] = metrics["bytes_sent"]
                row["bytes_per_sample"] = metrics["bytes_per_sample"]
                row["gen_gap"] = gen_gap
                row["runs_ok"] = metrics["successful_runs"]
                row["runs_total"] = metrics["total_runs"]
                results.append(row)

    _print_summary(results)
    for row in results:
        if not _status_ok(str(row.get("status"))):
            failed = True
            rc = row.get("rc")
            if rc not in (0, None):
                print(f"  rc={rc}")

    if not args.no_qa_log and not args.dry_run:
        qa_file = Path(args.qa_file)
        append_qa_log(qa_file, args, results)
        print(f"\nQA log updated: {qa_file}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
