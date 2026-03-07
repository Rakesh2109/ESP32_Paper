#!/usr/bin/env python3
"""
Parallel TM hyperparameter sweep on native localhost, then optional device verification.

Workflow:
1) Start N native socket servers (one per port).
2) Run exhaustive grid search in parallel across servers.
3) Rank by test accuracy (then throughput).
4) Optionally verify top-K configs on ESP32 device.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class TMConfig:
    clauses: int
    threshold: int
    specificity: int
    density: int
    seed: int

    def tag_suffix(self) -> str:
        return f"c{self.clauses}_t{self.threshold}_s{self.specificity}_d{self.density}_seed{self.seed}"


def parse_csv_ints(raw: str) -> List[int]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(int(tok))
    if not vals:
        raise ValueError("empty integer list")
    return vals


def latest_result_for_tag(tag: str) -> Optional[Path]:
    pattern = str(RESULTS_DIR / f"experiment_{tag}_*.json")
    hits = [Path(p) for p in glob.glob(pattern)]
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def extract_metrics(path: Path) -> Dict[str, Optional[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs", [])
    if not runs:
        return {
            "test_acc": None,
            "train_acc": None,
            "best_acc": None,
            "throughput": None,
            "latency_ms": None,
            "duration_s": None,
        }
    summary = runs[0].get("summary", {})
    return {
        "test_acc": float(summary["final_test_accuracy"]) if isinstance(summary.get("final_test_accuracy"), (int, float)) else None,
        "train_acc": float(summary["final_train_accuracy"]) if isinstance(summary.get("final_train_accuracy"), (int, float)) else None,
        "best_acc": float(summary["best_accuracy"]) if isinstance(summary.get("best_accuracy"), (int, float)) else None,
        "throughput": float(summary["avg_throughput_sps"]) if isinstance(summary.get("avg_throughput_sps"), (int, float)) else None,
        "latency_ms": float(summary["avg_latency_ms"]) if isinstance(summary.get("avg_latency_ms"), (int, float)) else None,
        "duration_s": float(summary["total_duration_sec"]) if isinstance(summary.get("total_duration_sec"), (int, float)) else None,
    }


def run_cmd(cmd: Sequence[str], timeout_s: Optional[int] = None, log_file: Optional[Path] = None) -> int:
    if log_file is None:
        return subprocess.run(cmd, cwd=str(ROOT), timeout=timeout_s).returncode
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        cp = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=timeout_s,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return cp.returncode


def build_test_cmd(
    *,
    python_exe: str,
    port: str,
    model: str,
    dataset: str,
    epochs: int,
    runs: int,
    train_samples: int,
    test_samples: int,
    window: int,
    batch_records: int,
    cfg: TMConfig,
    model_tag: str,
) -> List[str]:
    return [
        python_exe,
        "test_serial.py",
        "--port", port,
        "--baud", "2000000",
        "--dataset", dataset,
        "--model", model,
        "--epochs", str(epochs),
        "--runs", str(runs),
        "--train-samples", str(train_samples),
        "--test-samples", str(test_samples),
        "--csv-bitpack",
        "--unsafe-no-data-checksum",
        "--window", str(window),
        "--batch-records", str(batch_records),
        "--status-every", "0",
        "--model-tag", model_tag,
        "--tm-clauses", str(cfg.clauses),
        "--tm-threshold", str(cfg.threshold),
        "--tm-specificity", str(cfg.specificity),
        "--tm-seed", str(cfg.seed),
        "--tm-init-density", str(cfg.density),
    ]


def start_native_server(exe_path: Path, port: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HAL_SOCKET_PORT"] = str(port)
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [str(exe_path)],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_processes(procs: Iterable[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    t0 = time.time()
    for p in procs:
        if p.poll() is None:
            while time.time() - t0 < 3.0 and p.poll() is None:
                time.sleep(0.1)
            if p.poll() is None:
                p.kill()


def run_worker(
    worker_id: int,
    port: int,
    cfgs: Sequence[TMConfig],
    args: argparse.Namespace,
    phase_name: str,
    out_rows: List[Dict[str, object]],
) -> None:
    socket_port = f"socket://localhost:{port}"
    for idx, cfg in enumerate(cfgs, start=1):
        tag = f"{phase_name}_{args.model}_{cfg.tag_suffix()}_w{worker_id}"
        cmd = build_test_cmd(
            python_exe=args.python,
            port=socket_port,
            model=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            runs=args.runs,
            train_samples=args.train_samples,
            test_samples=args.test_samples,
            window=args.window,
            batch_records=args.batch_records,
            cfg=cfg,
            model_tag=tag,
        )
        t0 = time.time()
        run_log = RESULTS_DIR / "search_logs" / f"{tag}.log"
        rc = run_cmd(cmd, timeout_s=args.timeout_s, log_file=run_log)
        dt = time.time() - t0

        row: Dict[str, object] = {
            "phase": phase_name,
            "worker": worker_id,
            "port": port,
            "index": idx,
            "clauses": cfg.clauses,
            "threshold": cfg.threshold,
            "specificity": cfg.specificity,
            "density": cfg.density,
            "seed": cfg.seed,
            "rc": rc,
            "wall_s": dt,
            "result_file": "",
            "run_log": run_log.name,
            "test_acc": None,
            "train_acc": None,
            "best_acc": None,
            "throughput": None,
            "latency_ms": None,
            "duration_s": None,
        }

        if rc == 0:
            rf = latest_result_for_tag(tag)
            if rf is not None:
                m = extract_metrics(rf)
                row.update(m)
                row["result_file"] = rf.name
        out_rows.append(row)


def rank_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    ok = [r for r in rows if r.get("rc") == 0 and isinstance(r.get("test_acc"), (int, float))]
    ok.sort(
        key=lambda r: (
            float(r["test_acc"]),  # type: ignore[index]
            float(r["best_acc"] if isinstance(r.get("best_acc"), (int, float)) else -1.0),
            float(r["throughput"] if isinstance(r.get("throughput"), (int, float)) else -1.0),
        ),
        reverse=True,
    )
    return ok


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "phase", "worker", "port", "index",
        "clauses", "threshold", "specificity", "density", "seed",
        "rc", "wall_s", "test_acc", "train_acc", "best_acc", "throughput", "latency_ms", "duration_s", "result_file", "run_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def split_round_robin(items: Sequence[TMConfig], n: int) -> List[List[TMConfig]]:
    buckets: List[List[TMConfig]] = [[] for _ in range(n)]
    for i, it in enumerate(items):
        buckets[i % n].append(it)
    return buckets


def build_grid(args: argparse.Namespace) -> List[TMConfig]:
    out: List[TMConfig] = []
    for c in parse_csv_ints(args.clauses):
        for t in parse_csv_ints(args.thresholds):
            for s in parse_csv_ints(args.specificities):
                for d in parse_csv_ints(args.densities):
                    out.append(TMConfig(c, t, s, d, args.seed))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel native TM hyperparameter sweep + device verification")
    ap.add_argument("--model", default="tm_bo", choices=["tm_sparse", "tm_vanilla", "tm_bo"])
    ap.add_argument("--dataset", default="IoT_clean")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-port", type=int, default=5555)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--train-samples", type=int, default=3000)
    ap.add_argument("--test-samples", type=int, default=1000)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--batch-records", type=int, default=1024)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clauses", default="64,100,128,160,192")
    ap.add_argument("--thresholds", default="8,10,12,16")
    ap.add_argument("--specificities", default="2,3,4,5")
    ap.add_argument("--densities", default="25,35,50")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--verify-device", action="store_true")
    ap.add_argument("--device-port", default="COM12")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args)
    if args.shuffle:
        random.shuffle(grid)

    exe = ROOT / ".pio" / "build" / "native" / "program.exe"
    if not exe.exists():
        print("native binary missing. run: pio run -e native")
        return 2

    print(f"Grid size: {len(grid)}")
    print(f"Workers: {args.workers} (ports {args.base_port}..{args.base_port + args.workers - 1})")
    print(
        f"Search run config: model={args.model} dataset={args.dataset} "
        f"epochs={args.epochs} train={args.train_samples} test={args.test_samples}"
    )

    procs: List[subprocess.Popen] = []
    try:
        for w in range(args.workers):
            port = args.base_port + w
            log_path = RESULTS_DIR / f"native_server_{port}.log"
            p = start_native_server(exe, port, log_path)
            procs.append(p)
        time.sleep(1.5)

        buckets = split_round_robin(grid, args.workers)
        from concurrent.futures import ThreadPoolExecutor

        all_rows: List[Dict[str, object]] = []
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for w, cfgs in enumerate(buckets):
                port = args.base_port + w
                futures.append(pool.submit(run_worker, w, port, cfgs, args, "native", all_rows))
            for f in futures:
                f.result()

        ranked = rank_rows(all_rows)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = RESULTS_DIR / f"tm_native_search_{args.model}_{ts}.csv"
        write_csv(csv_path, all_rows)
        print(f"\nSaved native sweep: {csv_path}")
        print(f"Completed: {len(all_rows)} / {len(grid)}")
        print(f"Successful with metrics: {len(ranked)}")
        print("\nTop 10 (native):")
        for i, r in enumerate(ranked[:10], start=1):
            print(
                f"{i:2d}. C={r['clauses']} T={r['threshold']} s={r['specificity']} d={r['density']} "
                f"acc={float(r['test_acc']):.4f} thr={float(r['throughput']):.1f} file={r['result_file']}"
            )

        if not args.verify_device or not ranked:
            return 0

        top = ranked[: max(1, min(args.topk, len(ranked)))]
        print(f"\nVerifying top {len(top)} on device {args.device_port}...")
        verify_rows: List[Dict[str, object]] = []
        for i, r in enumerate(top, start=1):
            cfg = TMConfig(
                clauses=int(r["clauses"]),
                threshold=int(r["threshold"]),
                specificity=int(r["specificity"]),
                density=int(r["density"]),
                seed=int(r["seed"]),
            )
            tag = f"verify_{args.model}_{i}_{cfg.tag_suffix()}"
            cmd = build_test_cmd(
                python_exe=args.python,
                port=args.device_port,
                model=args.model,
                dataset=args.dataset,
                epochs=args.epochs,
                runs=args.runs,
                train_samples=args.train_samples,
                test_samples=args.test_samples,
                window=args.window,
                batch_records=args.batch_records,
                cfg=cfg,
                model_tag=tag,
            )
            verify_log = RESULTS_DIR / "search_logs" / f"{tag}.log"
            rc = run_cmd(cmd, timeout_s=args.timeout_s, log_file=verify_log)
            row = {
                "rank_native": i,
                "clauses": cfg.clauses,
                "threshold": cfg.threshold,
                "specificity": cfg.specificity,
                "density": cfg.density,
                "seed": cfg.seed,
                "rc": rc,
                "native_test_acc": r.get("test_acc"),
                "native_thr": r.get("throughput"),
                "device_test_acc": None,
                "device_thr": None,
                "device_file": "",
                "run_log": "",
            }
            row["run_log"] = verify_log.name
            if rc == 0:
                rf = latest_result_for_tag(tag)
                if rf is not None:
                    m = extract_metrics(rf)
                    row["device_test_acc"] = m["test_acc"]
                    row["device_thr"] = m["throughput"]
                    row["device_file"] = rf.name
            verify_rows.append(row)

        verify_csv = RESULTS_DIR / f"tm_verify_device_{args.model}_{ts}.csv"
        with verify_csv.open("w", newline="", encoding="utf-8") as f:
            cols = [
                "rank_native", "clauses", "threshold", "specificity", "density", "seed",
                "rc", "native_test_acc", "native_thr", "device_test_acc", "device_thr", "device_file", "run_log",
            ]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in verify_rows:
                w.writerow({k: row.get(k) for k in cols})
        print(f"\nSaved device verification: {verify_csv}")
        print("\nTop config parity:")
        for row in verify_rows:
            print(
                f"rank={row['rank_native']} C={row['clauses']} T={row['threshold']} s={row['specificity']} d={row['density']} "
                f"native_acc={row['native_test_acc']} device_acc={row['device_test_acc']} "
                f"native_thr={row['native_thr']} device_thr={row['device_thr']}"
            )
        return 0
    finally:
        stop_processes(procs)


if __name__ == "__main__":
    raise SystemExit(main())
