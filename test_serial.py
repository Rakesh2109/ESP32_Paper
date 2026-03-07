#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 tree-model v2 binary streamer - RELIABLE
- Keeps frame/data formats identical.
- Robust ACK wait: tolerates interleaved STAT/READY/DONE/ERROR.
- Soft backoff of window; periodic STAT pings optional.
- Bounded batching to avoid serial buffer overfill.

Preprocessing: No host-side normalization. For EFDT, stream raw float32 rows
with --input-real or bit-packed binary features.

Models (C++, include/models/): EFDT and HAT-ADWIN. Offline eval: tests/offline_model_eval.cpp with CSV.
"""

from __future__ import annotations
import argparse, glob, struct, time
import contextlib
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import serial
from tqdm import tqdm
import os
import sys
import statistics
import json
from datetime import datetime
from loguru import logger

# Optional dependencies for CSV/RAW streaming
try:
    import numpy as np
    import pandas as pd
except Exception:
    np = None
    pd = None

try:
    from sklearn.preprocessing import StandardScaler
except Exception:
    StandardScaler = None

# Optional Joulescope for power/energy measurement (pip install joulescope)
try:
    import joulescope
    _JOULESCOPE_AVAILABLE = True
except Exception:
    joulescope = None
    _JOULESCOPE_AVAILABLE = False


class JoulescopeSampler:
    """Sample current, voltage, power during a phase; compute energy (J) = integral of power."""

    def __init__(self, poll_ms: float = 100.0, serial_number: Optional[str] = None,
                 suppress_usb_warnings: bool = True):
        self.poll_ms = max(10.0, float(poll_ms))
        self.serial_number = serial_number
        self.suppress_usb_warnings = bool(suppress_usb_warnings)
        self._device = None
        self._samples: List[Tuple[float, float, float, float]] = []
        self._thread = None
        self._stop = False

    @staticmethod
    @contextlib.contextmanager
    def _silence_stdio_fd(enabled: bool):
        if not enabled:
            yield
            return
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        devnull_file = None
        saved_fds: List[Optional[int]] = [None, None]
        devnull_fd: Optional[int] = None
        try:
            devnull_file = open(os.devnull, "w")
            sys.stdout = devnull_file
            sys.stderr = devnull_file
            try:
                saved_fds[0] = os.dup(1)
                saved_fds[1] = os.dup(2)
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
            except OSError:
                pass
            yield
        except Exception:
            yield
        finally:
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
            if devnull_file:
                try: devnull_file.close()
                except Exception: pass
            for i, fd_num in enumerate((1, 2)):
                if saved_fds[i] is not None:
                    try: os.dup2(saved_fds[i], fd_num); os.close(saved_fds[i])
                    except OSError: pass
            if devnull_fd is not None:
                try: os.close(devnull_fd)
                except OSError: pass

    def _sample_loop(self) -> None:
        while not self._stop and self._device is not None:
            try:
                with self._silence_stdio_fd(self.suppress_usb_warnings):
                    data = self._device.read(contiguous_duration=0.001)
                if data is not None and len(data) > 0:
                    i_a, v_v = float(data[-1, 0]), float(data[-1, 1])
                    self._samples.append((time.perf_counter(), i_a, v_v, i_a * v_v))
            except Exception:
                pass
            time.sleep(self.poll_ms / 1000.0)

    def open_device(self) -> bool:
        if not _JOULESCOPE_AVAILABLE:
            return False
        if self._device is not None:
            return True
        for attempt in range(1, 4):
            try:
                devices = joulescope.scan()
                if not devices:
                    logger.warning("Joulescope: no device found")
                    return False
                if self.serial_number:
                    devices = [d for d in devices if self.serial_number in str(d)]
                    if not devices:
                        return False
                self._device = joulescope.scan_require_one(config="auto")
                with self._silence_stdio_fd(self.suppress_usb_warnings):
                    self._device.open()
                logger.info("Joulescope: device opened for power/energy measurement")
                return True
            except Exception as e:
                try:
                    if self._device: self._device.close()
                except Exception: pass
                self._device = None
                if attempt < 3:
                    time.sleep(0.5 * attempt)
                else:
                    logger.warning("Joulescope open failed after 3 attempts: {}", e)
        return False

    def close_device(self) -> None:
        try:
            if self._device:
                with self._silence_stdio_fd(self.suppress_usb_warnings):
                    self._device.close()
                self._device = None
        except Exception:
            pass

    def start_phase(self) -> None:
        self._samples = []
        self._stop = False
        if self._device is not None:
            self._thread = __import__("threading").Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

    def stop_phase(self) -> Dict[str, Optional[float]]:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        out: Dict[str, Optional[float]] = {"current_mean_a": None, "voltage_mean_v": None,
                                            "power_mean_w": None, "energy_j": None,
                                            "duration_sec": None, "sample_count": 0}
        if not self._samples:
            return out
        times = [s[0] for s in self._samples]
        currents = [s[1] for s in self._samples]
        voltages = [s[2] for s in self._samples]
        powers = [s[3] for s in self._samples]
        out["sample_count"] = len(self._samples)
        out["current_mean_a"] = statistics.mean(currents)
        out["voltage_mean_v"] = statistics.mean(voltages)
        out["power_mean_w"] = statistics.mean(powers)
        out["duration_sec"] = times[-1] - times[0] if len(times) > 1 else 0.0
        energy_j = 0.0
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            if dt > 0:
                energy_j += (powers[i] + powers[i + 1]) / 2.0 * dt
        out["energy_j"] = energy_j
        return out

# ============================================================================
# ResultsFormatter - Generates human-readable text reports from JSON data
# ============================================================================

class ResultsFormatter:
    """Formats experimental results into human-readable text reports."""
    
    @staticmethod
    def format_full_report(results: dict) -> str:
        """Generate complete experiment report from results dictionary."""
        lines = []
        
        # Header
        lines.append("="*70)
        lines.append("EXPERIMENT RESULTS")
        lines.append("="*70)
        lines.append("")
        
        # Metadata
        lines.append("METADATA:")
        for key, val in results['metadata'].items():
            lines.append(f"  {key:20s}: {val}")
        lines.append("")
        
        # Configuration
        lines.append("CONFIGURATION:")
        for key, val in results['configuration'].items():
            lines.append(f"  {key:20s}: {val}")
        lines.append("")
        
        # Runs
        lines.append("="*70)
        lines.append(f"EXPERIMENTAL RUNS ({len(results['runs'])})")
        lines.append("="*70)
        lines.append("")
        
        for run_data in results['runs']:
            lines.extend(ResultsFormatter._format_run(run_data))
        
        # Statistics
        if results['statistics']:
            lines.extend(ResultsFormatter._format_statistics(results['statistics']))
        
        # Comprehensive analysis
        if results['comprehensive_analysis']:
            lines.append("="*70)
            lines.append("COMPREHENSIVE ANALYSIS FROM ESP32")
            lines.append("="*70)
            lines.append("")
            
            ca = results['comprehensive_analysis']
            
            # Format structured data if present
            if isinstance(ca, dict):
                # Memory Efficiency
                if ca.get('memory_efficiency'):
                    me = ca['memory_efficiency']
                    lines.append("MEMORY EFFICIENCY ANALYSIS:")
                    lines.append(f"  Model Memory:           {me['tm_memory']} bytes ({me['tm_memory']/1024:.2f} KB)")
                    lines.append(f"  Memory per Feature:     {me['memory_per_feature']:.2f} bytes")
                    lines.append(f"  Fragmentation:          {me['fragmentation']:.2f}%")
                    lines.append(f"  Current Allocated:      {me['current_allocated']} bytes ({me['current_allocated']/1024:.2f} KB)")
                    lines.append(f"  Peak Allocated:         {me['peak_allocated']} bytes ({me['peak_allocated']/1024:.2f} KB)")
                    lines.append(f"  Memory Efficiency:      {me['memory_efficiency']:.2f}%")
                    lines.append("")
                
                # Performance Benchmarks
                if ca.get('performance_benchmarks'):
                    pb = ca['performance_benchmarks']
                    lines.append("PERFORMANCE BENCHMARKS:")
                    lines.append(f"  Train Throughput:       {pb['train_throughput']:.2f} smp/s")
                    lines.append(f"  Test Throughput:        {pb['test_throughput']:.2f} smp/s")
                    lines.append(f"  Memory/Train Sample:    {pb['mem_per_train_sample']:.2f} bytes")
                    lines.append(f"  Memory/Test Sample:     {pb['mem_per_test_sample']:.2f} bytes")
                    lines.append(f"  Model Size:             {pb['tm_core_memory']} bytes ({pb['tm_core_memory']/1024:.2f} KB)")
                    lines.append(f"  Performance Score:      {pb['performance_score']:.2f}%")
                    lines.append(f"  Efficiency Ratio:       {pb['efficiency_ratio']:.2f}")
                    lines.append(f"  Precision:              {pb['precision']:.4f}")
                    lines.append(f"  Recall:                 {pb['recall']:.4f}")
                    lines.append(f"  F1 Score:               {pb['f1_score']:.4f}")
                    lines.append("")
                
                # Protocol Statistics
                if ca.get('protocol_statistics'):
                    ps = ca['protocol_statistics']
                    lines.append("PROTOCOL STATISTICS:")
                    lines.append(f"  Frames Received:        {ps['frames_received']}")
                    lines.append(f"  Frames Valid:           {ps['frames_valid']}")
                    lines.append(f"  Rejected (Size):        {ps['rejected_size']}")
                    lines.append(f"  Rejected (Checksum):    {ps['rejected_checksum']}")
                    lines.append(f"  Rejected (Format):      {ps['rejected_format']}")
                    lines.append(f"  Rejection Rate:         {ps['rejection_rate']:.2f}%")
                    lines.append("")
                
                # Log output (bounds monitoring JSON, etc.)
                if ca.get('log_output'):
                    lines.append("ADDITIONAL LOG OUTPUT:")
                    lines.append(ca['log_output'])
                    lines.append("")
        
        # Footer
        lines.append("="*70)
        lines.append("END OF REPORT")
        lines.append("="*70)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_run(run_data: dict) -> list:
        """Format a single run's data."""
        lines = []
        run_num = run_data['run_number']
        summary = run_data['summary']
        
        status = "✅" if not summary['failed'] else "❌"
        lines.append(f"{status} RUN {run_num}:")
        lines.append(f"  Duration:        {summary['total_duration_sec']:.1f}s")
        lines.append(f"  Samples sent:    {summary['total_samples_sent']}")
        lines.append(f"  Throughput:      {summary['avg_throughput_sps']:.1f} smp/s")
        # Calculate and display latency (time per sample in milliseconds)
        latency_ms = summary.get('avg_latency_ms', 0)
        if latency_ms == 0 and summary['avg_throughput_sps'] > 0:
            latency_ms = 1000.0 / summary['avg_throughput_sps']
        lines.append(f"  Latency:         {latency_ms:.2f} ms/sample")
        if summary.get('train_latency_ms', 0) > 0:
            lines.append(f"  Train latency:   {summary['train_latency_ms']:.2f} ms/sample")
        if summary.get('test_latency_ms', 0) > 0:
            lines.append(f"  Test latency:    {summary['test_latency_ms']:.2f} ms/sample")
        if summary.get('train_energy_j') is not None:
            lines.append(f"  Train energy:    {summary['train_energy_j']:.4f} J")
        if summary.get('train_energy_uj_per_sample') is not None:
            lines.append(f"  Train energy:    {summary['train_energy_uj_per_sample']:.2f} \u03bcJ/sample")
        if summary.get('test_energy_j') is not None:
            lines.append(f"  Test energy:     {summary['test_energy_j']:.4f} J")
        if summary.get('test_energy_uj_per_sample') is not None:
            lines.append(f"  Test energy:     {summary['test_energy_uj_per_sample']:.2f} \u03bcJ/sample")
        if summary.get('train_power_mean_w') is not None:
            lines.append(f"  Train power:     {summary['train_power_mean_w']:.4f} W (mean)")
        if summary.get('test_power_mean_w') is not None:
            lines.append(f"  Test power:      {summary['test_power_mean_w']:.4f} W (mean)")
        if summary.get('train_current_mean_a') is not None:
            lines.append(f"  Train current:   {summary['train_current_mean_a']:.6f} A (mean)")
        if summary.get('test_current_mean_a') is not None:
            lines.append(f"  Test current:    {summary['test_current_mean_a']:.6f} A (mean)")
        if summary.get('train_voltage_mean_v') is not None:
            lines.append(f"  Train voltage:  {summary['train_voltage_mean_v']:.4f} V (mean)")
        if summary.get('test_voltage_mean_v') is not None:
            lines.append(f"  Test voltage:   {summary['test_voltage_mean_v']:.4f} V (mean)")
        lines.append(f"  Best accuracy:   {summary['best_accuracy']:.4f} (epoch {summary['best_epoch']})")
        lines.append(f"  Final train acc: {summary['final_train_accuracy']}")
        lines.append(f"  Final test acc:  {summary['final_test_accuracy']}")
        
        # Epoch details
        if run_data['epochs']:
            lines.append(f"\n  Epochs ({len(run_data['epochs'])}):")
            for epoch_data in run_data['epochs']:
                ep = epoch_data['epoch']
                train_acc = epoch_data['train'].get('accuracy', 'n/a')
                test_acc = epoch_data['test'].get('accuracy', 'n/a')
                line = f"    Epoch {ep}: train_acc={train_acc}, test_acc={test_acc}"
                tr_e = epoch_data.get('train', {}).get('energy_j')
                te_e = epoch_data.get('test', {}).get('energy_j')
                tr_n = epoch_data.get('train', {}).get('samples_sent')
                te_n = epoch_data.get('test', {}).get('samples_sent')
                tr_e_uj = ((float(tr_e) * 1e6) / float(tr_n)) if (tr_e is not None and isinstance(tr_n, int) and tr_n > 0) else None
                te_e_uj = ((float(te_e) * 1e6) / float(te_n)) if (te_e is not None and isinstance(te_n, int) and te_n > 0) else None
                if tr_e is not None or te_e is not None:
                    line += f" | train_E={tr_e:.4f}J" if tr_e is not None else " | train_E=n/a"
                    line += f" test_E={te_e:.4f}J" if te_e is not None else " test_E=n/a"
                if tr_e_uj is not None or te_e_uj is not None:
                    line += f" train_E\u03bc={tr_e_uj:.2f}\u03bcJ/sample" if tr_e_uj is not None else ""
                    line += f" test_E\u03bc={te_e_uj:.2f}\u03bcJ/sample" if te_e_uj is not None else ""
                lines.append(line)
        
        # Memory profile
        if run_data['memory_profile']:
            mp = run_data['memory_profile']
            lines.append(f"\n  Memory Profile:")
            lines.append(f"    Heap free:    {mp['free_heap']}")
            lines.append(f"    Heap min:     {mp['min_heap']}")
            lines.append(f"    Max alloc:    {mp['max_alloc']}")
            lines.append(f"    Utils peak:   {mp['util_peak']}")
        
        lines.append("")
        return lines
    
    @staticmethod
    def _format_statistics(stats: dict) -> list:
        """Format statistical analysis section."""
        lines = []
        
        lines.append("="*70)
        lines.append("STATISTICAL ANALYSIS")
        lines.append("="*70)
        lines.append("")
        
        lines.append(f"Total runs:       {stats['total_runs']}")
        lines.append(f"Successful:       {stats['successful_runs']}")
        lines.append(f"Failed:           {stats['failed_runs']}")
        lines.append("")
        
        if stats['test_accuracy_best']['mean'] is not None:
            lines.append("Test Accuracy (Best per run):")
            lines.append(f"  Mean:   {stats['test_accuracy_best']['mean']:.4f}")
            lines.append(f"  Std:    {stats['test_accuracy_best']['std']:.4f}")
            lines.append(f"  Min:    {stats['test_accuracy_best']['min']:.4f}")
            lines.append(f"  Max:    {stats['test_accuracy_best']['max']:.4f}")
            lines.append(f"  Median: {stats['test_accuracy_best']['median']:.4f}")
            if stats['test_accuracy_best']['ci_95'] is not None:
                lines.append(f"  95% CI: {stats['test_accuracy_best']['mean']:.4f} ± {stats['test_accuracy_best']['ci_95']:.4f}")
        lines.append("")
        
        if stats.get('precision') and stats['precision']['mean'] is not None:
            lines.append("Precision:")
            lines.append(f"  Mean:   {stats['precision']['mean']:.4f}")
            lines.append(f"  Std:    {stats['precision']['std']:.4f}")
            if stats['precision']['min'] is not None:
                lines.append(f"  Min:    {stats['precision']['min']:.4f}")
            if stats['precision']['max'] is not None:
                lines.append(f"  Max:    {stats['precision']['max']:.4f}")
            if stats['precision']['median'] is not None:
                lines.append(f"  Median: {stats['precision']['median']:.4f}")
            lines.append("")
        
        if stats.get('recall') and stats['recall']['mean'] is not None:
            lines.append("Recall:")
            lines.append(f"  Mean:   {stats['recall']['mean']:.4f}")
            lines.append(f"  Std:    {stats['recall']['std']:.4f}")
            if stats['recall']['min'] is not None:
                lines.append(f"  Min:    {stats['recall']['min']:.4f}")
            if stats['recall']['max'] is not None:
                lines.append(f"  Max:    {stats['recall']['max']:.4f}")
            if stats['recall']['median'] is not None:
                lines.append(f"  Median: {stats['recall']['median']:.4f}")
            lines.append("")
        
        if stats.get('f1_score') and stats['f1_score']['mean'] is not None:
            lines.append("F1 Score:")
            lines.append(f"  Mean:   {stats['f1_score']['mean']:.4f}")
            lines.append(f"  Std:    {stats['f1_score']['std']:.4f}")
            if stats['f1_score']['min'] is not None:
                lines.append(f"  Min:    {stats['f1_score']['min']:.4f}")
            if stats['f1_score']['max'] is not None:
                lines.append(f"  Max:    {stats['f1_score']['max']:.4f}")
            if stats['f1_score']['median'] is not None:
                lines.append(f"  Median: {stats['f1_score']['median']:.4f}")
            lines.append("")
        
        if stats.get('throughput') and stats['throughput']['mean'] is not None:
            lines.append("Throughput (samples/sec):")
            lines.append(f"  Mean:   {stats['throughput']['mean']:.1f}")
            lines.append(f"  Std:    {stats['throughput']['std']:.1f}")
            if stats['throughput']['min'] is not None:
                lines.append(f"  Min:    {stats['throughput']['min']:.1f}")
            if stats['throughput']['max'] is not None:
                lines.append(f"  Max:    {stats['throughput']['max']:.1f}")
            if stats['throughput']['median'] is not None:
                lines.append(f"  Median: {stats['throughput']['median']:.1f}")
            lines.append("")
        
        if stats.get('latency') and stats['latency']['mean'] is not None:
            lines.append("Latency (ms per sample):")
            lines.append(f"  Mean:   {stats['latency']['mean']:.2f}")
            lines.append(f"  Std:    {stats['latency']['std']:.2f}")
            if stats['latency']['min'] is not None:
                lines.append(f"  Min:    {stats['latency']['min']:.2f}")
            if stats['latency']['max'] is not None:
                lines.append(f"  Max:    {stats['latency']['max']:.2f}")
            if stats['latency']['median'] is not None:
                lines.append(f"  Median: {stats['latency']['median']:.2f}")
            lines.append("")
        
        return lines
    
    @staticmethod
    def format_statistics_report(stats: dict, config: dict, timestamp: str) -> str:
        """Generate statistics-only report (for multi-run experiments)."""
        lines = []
        
        lines.append(f"Statistical Analysis - {config['runs']} Runs")
        lines.append("="*70)
        lines.append(f"Configuration: window={config['window']}, throttle={config['throttle_ms']}ms, epochs={config['epochs']}")
        lines.append(f"Timestamp: {timestamp}")
        lines.append("")
        
        lines.append("Per-Run Results:")
        lines.append("-" * 70)
        # Note: Per-run details need to be passed separately or extracted from full results
        lines.append("")
        
        lines.append("="*70)
        lines.append("Summary Statistics:")
        lines.append("-" * 70)
        lines.append("")
        
        if stats['test_accuracy_best']['mean'] is not None:
            lines.append("Test Accuracy (Best per run):")
            lines.append(f"  Mean:   {stats['test_accuracy_best']['mean']:.4f}")
            lines.append(f"  Std:    {stats['test_accuracy_best']['std']:.4f}")
            lines.append(f"  Min:    {stats['test_accuracy_best']['min']:.4f}")
            lines.append(f"  Max:    {stats['test_accuracy_best']['max']:.4f}")
            lines.append(f"  Median: {stats['test_accuracy_best']['median']:.4f}")
            lines.append(f"  Values: {[f'{x:.4f}' for x in stats['test_accuracy_best']['values']]}")
            
            if stats['test_accuracy_best']['ci_95'] is not None:
                lines.append(f"  95% CI: {stats['test_accuracy_best']['mean']:.4f} ± {stats['test_accuracy_best']['ci_95']:.4f}")
        
        lines.append(f"\nRuns completed: {stats['successful_runs']}/{stats['total_runs']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_epoch_report(epoch_data: dict, epoch_num: int, total_epochs: int) -> str:
        """Generate per-epoch report."""
        lines = []
        
        lines.append(f"Epoch {epoch_num}/{total_epochs} Results")
        lines.append("=" * 70)
        
        if epoch_data['train']:
            train = epoch_data['train']
            lines.append("\nTraining:")
            lines.append(f"  Samples:    {train['samples_sent']}")
            lines.append(f"  Duration:   {train['duration_sec']:.1f}s")
            lines.append(f"  Throughput: {train['throughput_sps']:.1f} smp/s")
            if train.get('accuracy') is not None:
                lines.append(f"  Accuracy:   {train['accuracy']:.4f}")
                if train.get('ema_accuracy') is not None:
                    lines.append(f"  EMA Acc:    {train['ema_accuracy']:.3f}")
        
        if epoch_data['test']:
            test = epoch_data['test']
            lines.append("\nTesting:")
            lines.append(f"  Samples:    {test['samples_sent']}")
            lines.append(f"  Duration:   {test['duration_sec']:.1f}s")
            if test.get('accuracy') is not None:
                lines.append(f"  Accuracy:   {test['accuracy']:.4f}")
        
        if epoch_data.get('snapshot'):
            snap = epoch_data['snapshot']
            lines.append("\nSnapshot:")
            for k, v in snap.items():
                lines.append(f"  {k}: {v}")
        
        if epoch_data.get('error'):
            lines.append(f"\n❌ Error: {epoch_data['error']}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)

# ============================================================================
# Protocol Constants
# ============================================================================

FRAME_MAGIC=0xA55A
FRAME_TYPE_RECORD=0x01; FRAME_TYPE_CMD=0x02
FRAME_TYPE_ACK=0x81; FRAME_TYPE_STAT=0x82; FRAME_TYPE_READY=0x83
FRAME_TYPE_DONE=0x84; FRAME_TYPE_ERROR=0x85; FRAME_TYPE_METRIC=0x86
FRAME_TYPE_MEMPROF=0x87; FRAME_TYPE_LOG=0x88
FRAME_TYPE_MEM_EFFICIENCY=0x89; FRAME_TYPE_PERF_BENCH=0x8A; FRAME_TYPE_PROTOCOL_STATS=0x8B
# Extended types
FRAME_TYPE_SAMPLE_RAW=0x06  # Raw real-valued sample row (float32)
FRAME_TYPE_RECORD_MC=0x07   # Bit-packed multiclass record
FRAME_TYPE_RECORD_BATCH=0x09  # Batch of packed binary records (unsafe fast path)
FLAG_FLETCHER=0x01
OPC_STATUS=0x3F; 
OPC_SYNC=0x59;
OPC_RESET=0x52;  # Full hardware reset
OPC_RESET_STATE=0x72;  # Soft reset to READY state
OPC_CONFIG=0x43       # Feature/configuration
OPC_MODEL_SELECT=0x4C # Runtime model selection
OPC_START_TRAIN=0x53; OPC_START_TEST=0x56; OPC_BUFFER=0x42; OPC_SNAPSHOT=0x4D; OPC_MEMPROF=0x50; OPC_SHOW_FINAL=0x46

CFG_TWINE = 0x01
CFG_INPUT_REAL = 0x02
CFG_MODEL_DENSE = 0x08
CFG_TM_RUNTIME = 0x10

PREPROC_NONE = 0
PREPROC_ONLINE_STANDARDIZE = 1
PREPROC_TWINE = 2

MODEL_NAME_TO_ID = {
    "efdt": 1,
    "hoeffding": 2,
    "hat": 3,
    "sgt": 4,
    "bnn": 5,
    "tm_sparse": 6,
    "tm_vanilla": 7,
    "tm_bo": 8,
}

PREPROC_NAME_TO_ID = {
    "none": PREPROC_NONE,
    "standardize": PREPROC_ONLINE_STANDARDIZE,
    "twine": PREPROC_TWINE,
}
PREPROC_ID_TO_NAME = {v: k for k, v in PREPROC_NAME_TO_ID.items()}


def _clamp_twine_bits(bits: int) -> int:
    return max(1, min(8, int(bits)))


def resolve_preproc(mode_name: str, twine_bits: int) -> Tuple[int, int]:
    mode_key = (mode_name or "none").strip().lower()
    if mode_key not in PREPROC_NAME_TO_ID:
        raise ValueError(f"Unknown preprocessor '{mode_name}'")
    preproc_id = PREPROC_NAME_TO_ID[mode_key]
    param0 = _clamp_twine_bits(twine_bits) if preproc_id == PREPROC_TWINE else 0
    return preproc_id, param0


def build_config_payload(*, input_real: bool, preproc_id: int, preproc_param0: int = 0,
                         dense_model: bool = False,
                         tm_runtime: Optional[Dict[str, int]] = None) -> Tuple[int, bytes]:
    flags = 0
    if input_real:
        flags |= CFG_INPUT_REAL
    if dense_model:
        flags |= CFG_MODEL_DENSE
    # Legacy compatibility bit for older firmwares that only inspect CFG_TWINE.
    if preproc_id == PREPROC_TWINE:
        flags |= CFG_TWINE
    if tm_runtime is not None:
        flags |= CFG_TM_RUNTIME
        clauses = max(2, int(tm_runtime.get("clauses", 100)))
        threshold = max(1, int(tm_runtime.get("threshold", 10)))
        specificity = max(2, int(tm_runtime.get("specificity", 4)))
        seed = int(tm_runtime.get("seed", 1)) & 0xFFFFFFFF
        init_density = max(0, min(100, int(tm_runtime.get("init_density_pct", 100))))
        payload = struct.pack(
            "<BBHHhhIB",
            flags & 0xFF,
            preproc_param0 & 0xFF,
            preproc_id & 0xFFFF,
            clauses & 0xFFFF,
            threshold,
            specificity,
            seed,
            init_density & 0xFF,
        )
    else:
        payload = struct.pack("<BBH", flags & 0xFF, preproc_param0 & 0xFF, preproc_id & 0xFFFF)
    return flags, payload


def resolve_tm_runtime_config(args: argparse.Namespace) -> Optional[Dict[str, int]]:
    raw = {
        "clauses": args.tm_clauses,
        "threshold": args.tm_threshold,
        "specificity": args.tm_specificity,
        "seed": args.tm_seed,
        "init_density_pct": args.tm_init_density,
    }
    if all(v is None for v in raw.values()):
        return None

    cfg = {
        "clauses": 100,
        "threshold": 10,
        "specificity": 4,
        "seed": 1,
        "init_density_pct": 100,
    }
    for key, value in raw.items():
        if value is not None:
            cfg[key] = int(value)
    return cfg

def fletcher16(data: bytes, seed: int=0) -> int:
    s1, s2 = seed & 0xFF, (seed >> 8) & 0xFF
    if not data:
        return ((s2 & 0xFF) << 8) | (s1 & 0xFF)

    # Numpy fast path avoids Python byte loops for larger payloads.
    if np is not None and len(data) >= 64:
        arr = np.frombuffer(data, dtype=np.uint8)
        i = 0
        # Chunking keeps intermediate sums bounded and cache-friendly.
        while i < arr.size:
            chunk = arr[i:i + 2048].astype(np.uint32, copy=False)
            csum = np.cumsum(chunk, dtype=np.uint64)
            s2 = int((s2 + (chunk.size * s1) + int(csum.sum(dtype=np.uint64))) % 255)
            s1 = int((s1 + int(csum[-1])) % 255)
            i += chunk.size
        return ((s2 & 0xFF) << 8) | (s1 & 0xFF)

    for b in data:
        s1 += b
        s1 -= 255 if s1 >= 255 else 0
        s2 += s1
        s2 -= 255 if s2 >= 255 else 0
    return ((s2 & 0xFF) << 8) | (s1 & 0xFF)

def parse_line_to_bits(line: str):
    s=line.strip()
    if not s or s.startswith("#"): raise ValueError("skip")
    v=[int(t) for t in s.split()]
    if any(x not in (0,1) for x in v): raise ValueError("non-binary")
    return v[:-1], v[-1]

def parse_line_to_bits_mc(line: str):
    s=line.strip()
    if not s or s.startswith("#"): raise ValueError("skip")
    t=s.split()
    if len(t)<2: raise ValueError("short")
    *feat_str, lbl_str = t
    feats=[int(x) for x in feat_str]
    if any(x not in (0,1) for x in feats): raise ValueError("non-binary-feat")
    lbl=int(lbl_str)
    if lbl<0: raise ValueError("neg-label")
    return feats, lbl

def pack_feature_bits(bits: List[int]) -> bytes:
    out, cur, n = bytearray(), 0, 0
    for b in bits:
        cur = ((cur<<1)|(b&1)) & 0xFF; n+=1
        if n==8: out.append(cur); cur=0; n=0
    if n: out.append((cur<<(8-n)) & 0xFF)
    return bytes(out)

def pack_record(feats: List[int], label: int) -> bytes:
    N=len(feats); header=((1 if label else 0)<<31)|(N & 0x7FFFFFFF)
    return struct.pack("<I",header)+pack_feature_bits(feats)

def pack_record_from_real_row(feats, label: int) -> bytes:
    # Fast host-side binarization for transport-efficient CSV streaming.
    # Values > 0 become 1, else 0.
    bits = (np.asarray(feats) > 0).astype(np.uint8, copy=False)
    nfeat = int(bits.shape[0])
    packed = np.packbits(bits, bitorder='big').tobytes()
    header = ((1 if label else 0) << 31) | (nfeat & 0x7FFFFFFF)
    return struct.pack("<I", header) + packed

def pack_records_from_real_matrix(X, y, limit: Optional[int] = None) -> List[bytes]:
    """Vectorized binarize+pack for CSV bit-pack mode."""
    if np is None:
        rows = X if limit is None else X[:limit]
        labels = y if limit is None else y[:limit]
        return [pack_record_from_real_row(rows[i], int(labels[i])) for i in range(rows.shape[0])]

    rows = X if limit is None else X[:limit]
    labels = y if limit is None else y[:limit]
    if rows.shape[0] == 0:
        return []

    bits = (rows > 0).astype(np.uint8, copy=False)
    packed_rows = np.packbits(bits, axis=1, bitorder='big')
    nfeat = int(rows.shape[1]) & 0x7FFFFFFF
    out: List[bytes] = []
    append = out.append
    for i in range(rows.shape[0]):
        header = ((1 if int(labels[i]) else 0) << 31) | nfeat
        append(struct.pack("<I", header) + packed_rows[i].tobytes())
    return out

def prepare_labeled_bitpack_rows(X, y, limit: Optional[int] = None):
    """
    Vectorized host-side bit-pack prep:
      returns (labeled_rows, nfeat)
      labeled_rows shape: [rows, 1 + packed_bytes], dtype=uint8
      first byte per row is label (0/1), rest are packed feature bits.
    """
    if np is None:
        return None, 0

    rows = X if limit is None else X[:limit]
    labels = y if limit is None else y[:limit]
    n_rows = int(rows.shape[0])
    if n_rows == 0:
        return np.empty((0, 1), dtype=np.uint8), int(rows.shape[1]) if rows.ndim == 2 else 0

    bits = (rows > 0).astype(np.uint8, copy=False)
    packed_rows = np.packbits(bits, axis=1, bitorder='big')
    lbl = (np.asarray(labels).astype(np.uint8, copy=False) & 1).reshape(-1, 1)

    labeled_rows = np.empty((n_rows, 1 + packed_rows.shape[1]), dtype=np.uint8)
    labeled_rows[:, 0:1] = lbl
    labeled_rows[:, 1:] = packed_rows
    return labeled_rows, int(rows.shape[1])

def pack_record_batch_payload(records: List[bytes]) -> bytes:
    """
    Convert packed RECORD payloads into batch payload:
      <count:u16><nfeat:u16> + count * (<label:u8><packed_bits>)
    All records must share nfeat.
    """
    if not records:
        return b""
    hdr0 = struct.unpack_from("<I", records[0], 0)[0]
    nfeat = int(hdr0 & 0x7FFFFFFF)
    nbytes = (nfeat + 7) // 8
    out = bytearray(struct.pack("<HH", len(records), nfeat))
    for rec in records:
        if len(rec) < 4:
            raise ValueError("short packed record payload")
        hdr = struct.unpack_from("<I", rec, 0)[0]
        lbl = 1 if ((hdr >> 31) & 0x1) else 0
        rfeat = int(hdr & 0x7FFFFFFF)
        if rfeat != nfeat:
            raise ValueError("mixed nfeat in batch")
        bits = rec[4:]
        if len(bits) != nbytes:
            raise ValueError("packed bit-length mismatch in batch")
        out.append(lbl)
        out.extend(bits)
    return bytes(out)

def pack_record_batch_payload_from_labeled_rows(labeled_rows, nfeat: int, start: int, count: int) -> bytes:
    if np is None:
        raise RuntimeError("numpy required for labeled row batch payload path")
    if count <= 0:
        return b""
    body = labeled_rows[start:start + count]
    return struct.pack("<HH", int(count), int(nfeat)) + body.tobytes(order='C')

def pack_record_mc(feats: List[int], label: int) -> bytes:
    n=len(feats)
    header=struct.pack("<HBB", n, label & 0xFF, 0)
    return header + pack_feature_bits(feats)

def iter_records(path: Path):
    with path.open("r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            try: feats, lbl = parse_line_to_bits(line)
            except ValueError: continue
            yield struct.pack("<I", ((1 if lbl else 0)<<31)|(len(feats)&0x7FFFFFFF)) + pack_feature_bits(feats)

def iter_records_mc(path: Path):
    with path.open("r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            try: feats, lbl = parse_line_to_bits_mc(line)
            except ValueError: continue
            yield pack_record_mc(feats, lbl)

def count_records(paths: List[Path]) -> int:
    total=0
    for p in paths:
        with p.open("r",encoding="utf-8",errors="ignore") as f:
            for s in f:
                s=s.strip()
                if s and not s.startswith("#"): total+=1
    return total

# ---------------- CSV/RAW helpers ----------------
def load_csv_dataset(csv_path: Path, label_col: Optional[str], train_frac: float = 0.8,
                     shuffle: bool = True, seed: int = 42):
    if pd is None or np is None:
        raise SystemExit("pandas/numpy required for --csv. Install via requirements.txt")
    df = pd.read_csv(csv_path)
    if label_col is None:
        label_col = df.columns[-1]
    if label_col not in df.columns:
        raise SystemExit(f"Label column '{label_col}' not found in CSV")
    y = df[label_col].values
    X = df.drop(columns=[label_col]).values
    X = X.astype(np.float32, copy=False)
    # Ensure binary labels {0,1}
    if y.dtype.kind not in ('i','u'):
        uniq = pd.unique(y)
        if len(uniq) == 2:
            mapping = {uniq[0]:0, uniq[1]:1}
            y = np.array([mapping.get(v, 0) for v in y], dtype=np.uint8)
        else:
            y = pd.to_numeric(y, errors='coerce').fillna(0).astype(np.int32).clip(0,1).values
    y = (np.asarray(y).astype(np.uint8)) & 1
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_train = int(n * train_frac)
    tr_idx = idx[:n_train]
    te_idx = idx[n_train:]
    X_train = X[tr_idx].astype(np.float32, copy=False)
    X_test = X[te_idx].astype(np.float32, copy=False)

    return (X_train, y[tr_idx]), (X_test, y[te_idx]), None


def load_txt_dataset_as_array(paths: List[Path], *, multiclass: bool = False) -> Tuple["np.ndarray", "np.ndarray"]:
    """
    Load one or more whitespace-delimited binary feature text files into numpy arrays.
    Each line must contain feature bits followed by the label.
    """
    if np is None:
        raise SystemExit("numpy required for --input-real with text datasets (install via requirements.txt)")

    parse_fn = parse_line_to_bits_mc if multiclass else parse_line_to_bits

    samples: List[List[int]] = []
    labels: List[int] = []
    expected_dim: Optional[int] = None

    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    feats, lbl = parse_fn(line)
                except ValueError:
                    continue
                if expected_dim is None:
                    expected_dim = len(feats)
                elif len(feats) != expected_dim:
                    raise SystemExit(f"Inconsistent feature length detected in {path} (expected {expected_dim}, got {len(feats)})")
                samples.append(feats)
                labels.append(lbl)

    if not samples:
        raise SystemExit("No valid samples parsed from provided text dataset paths.")

    X = np.asarray(samples, dtype=np.float32)
    y_dtype = np.int32 if multiclass else np.uint8
    y = np.asarray(labels, dtype=y_dtype)
    return X, y

def iter_rows_raw(X, y):
    for i in range(X.shape[0]):
        yield X[i], int(y[i])

def pack_raw_sample_row(feats, label: int) -> bytes:
    nfeat = feats.shape[0]
    header = struct.pack("<HBB", nfeat, (1 if label else 0) & 0xFF, 1)  # dtype=1 float32
    return header + feats.astype(np.float32, copy=False).tobytes(order='C')

def make_frame(ftype:int, payload:bytes, seq:int, use_chk:bool=True)->bytes:
    flags = FLAG_FLETCHER if use_chk else 0
    hdr = struct.pack("<HBBHI", FRAME_MAGIC, ftype, flags, len(payload), seq & 0xFFFFFFFF)
    if not use_chk: return hdr+payload
    seed = fletcher16(hdr[2:])
    chk  = fletcher16(payload, seed) if payload else seed
    return hdr + payload + struct.pack("<H", chk)

def append_frame(buf: bytearray, ftype: int, payload: bytes, seq: int, use_chk: bool = True) -> int:
    """
    Append a framed payload directly into an existing bytearray.
    Returns number of bytes appended.
    """
    flags = FLAG_FLETCHER if use_chk else 0
    hdr = struct.pack("<HBBHI", FRAME_MAGIC, ftype, flags, len(payload), seq & 0xFFFFFFFF)
    buf += hdr
    if payload:
        buf += payload
    if use_chk:
        seed = fletcher16(hdr[2:])
        chk = fletcher16(payload, seed) if payload else seed
        buf += struct.pack("<H", chk)
    return 10 + len(payload) + (2 if use_chk else 0)

def append_record_batch_frame_from_labeled_rows(
    buf: bytearray,
    labeled_rows,
    nfeat: int,
    start: int,
    count: int,
    seq: int,
    use_chk: bool = True,
) -> int:
    """
    Zero-extra-copy append for FRAME_TYPE_RECORD_BATCH from precomputed
    labeled uint8 rows ([label | packed_bits]).
    """
    if np is None:
        raise RuntimeError("numpy required for labeled row batch append path")
    if count <= 0:
        return 0

    rows = labeled_rows[start:start + count]
    if rows.size == 0:
        return 0
    if not rows.flags.c_contiguous:
        rows = np.ascontiguousarray(rows)

    row_bytes = int(rows.shape[1])
    payload_len = 4 + int(count) * row_bytes
    flags = FLAG_FLETCHER if use_chk else 0
    hdr = struct.pack("<HBBHI", FRAME_MAGIC, FRAME_TYPE_RECORD_BATCH, flags, payload_len, seq & 0xFFFFFFFF)
    batch_hdr = struct.pack("<HH", int(count), int(nfeat))
    body_mv = memoryview(rows).cast("B")

    buf += hdr
    buf += batch_hdr
    buf += body_mv

    if use_chk:
        seed = fletcher16(hdr[2:])
        seed = fletcher16(batch_hdr, seed)
        chk = fletcher16(body_mv, seed)
        buf += struct.pack("<H", chk)

    return 10 + payload_len + (2 if use_chk else 0)

class Frame:
    __slots__=("type","flags","seq","payload","checksum_ok")
    def __init__(self,t,f,s,p,ok): self.type=t;self.flags=f;self.seq=s;self.payload=p;self.checksum_ok=ok

def read_exact(ser:serial.Serial, n:int, timeout_s:float)->Optional[bytes]:
    end=time.time()+timeout_s; buf=bytearray()
    while len(buf)<n:
        b=ser.read(n-len(buf))
        if b: buf+=b
        elif time.time()>end: return None
        else: time.sleep(0.001)
    return bytes(buf)

def find_magic(ser:serial.Serial, timeout_s:float)->bool:
    import serial.serialutil
    end=time.time()+timeout_s; state=0
    while time.time()<end:
        try:
            b=ser.read(1)
            if not b: 
                time.sleep(0.001); 
                continue
            c=b[0]
            if state==0: state=1 if c==0x5A else 0
            else:
                if c==0xA5: return True
                state=1 if c==0x5A else 0
        except (serial.serialutil.SerialException, OSError) as e:
            # Serial port error (device disconnected, etc.)
            # Return False to let caller handle it
            return False
    return False

def read_frame(ser: serial.Serial, timeout_s: float = 5.0) -> Optional[Frame]:
    if not find_magic(ser, timeout_s): return None
    hdr = read_exact(ser, 1 + 1 + 2 + 4, timeout_s=timeout_s)
    if hdr is None: return None
    ftype = hdr[0]; flags=hdr[1]
    length = hdr[2] | (hdr[3] << 8)
    seq = struct.unpack_from("<I", hdr, 4)[0]
    payload = read_exact(ser, length, timeout_s=timeout_s) if length else b""
    if payload is None: return None
    checksum_ok = True
    if flags & FLAG_FLETCHER:
        rx = read_exact(ser, 2, timeout_s=timeout_s)
        if rx is None: return None
        rx_chk = struct.unpack("<H", rx)[0]
        seed = fletcher16(bytes([ftype, flags]) + struct.pack("<H", length) + struct.pack("<I", seq))
        calc = fletcher16(payload, seed)
        checksum_ok = (rx_chk == calc)
    return Frame(ftype, flags, seq, payload, checksum_ok)

def write_all(ser: serial.Serial, data: bytes, timeout_s: float = 5.0) -> None:
    if not data:
        return
    mv = memoryview(data)
    sent = 0
    deadline = time.time() + timeout_s
    while sent < len(mv):
        n = ser.write(mv[sent:])
        if n is None:
            n = 0
        if n > 0:
            sent += n
            continue
        if time.time() >= deadline:
            raise TimeoutError(f"serial write timeout ({sent}/{len(mv)} bytes sent)")
        time.sleep(0.001)

def send_cmd(ser:serial.Serial, seq:int, opcode:int, extra:bytes=b"")->int:
    write_all(ser, make_frame(FRAME_TYPE_CMD, bytes([opcode])+extra, seq, True), timeout_s=1.0)
    return (seq+1)&0xFFFFFFFF

def tune_serial_link(ser: serial.Serial, network_link: bool = False) -> None:
    """
    Favor low-latency frame handling over long blocking reads/writes.
    This keeps ACK/STAT processing responsive at high sample rates.
    """
    try:
        ser.timeout = 0.05 if network_link else 0.01
    except Exception:
        pass
    try:
        ser.write_timeout = 1.0
    except Exception:
        pass
    if not network_link:
        try:
            # Best-effort on supported drivers (Windows/pyserial).
            ser.set_buffer_size(rx_size=1 << 20, tx_size=1 << 20)
        except Exception:
            pass

def wait_ready(ser:serial.Serial, timeout_s:float=12.0):
    """Wait for READY frame. Returns (ready_ok, trained_count, tested_count)."""
    logger.info(f"🔍 Waiting for READY frame (timeout: {timeout_s:.0f}s)...")
    log_frames_received = 0
    end=time.time()+timeout_s
    while time.time()<end:
        fr=read_frame(ser,timeout_s=min(1.0,max(0.05,end-time.time())))
        if not fr: continue
        if fr.checksum_ok and fr.type==FRAME_TYPE_READY:
            # Parse READY payload: (major:u8, minor:u8, pad:u16, trained:u32, tested:u32)
            trained, tested = 0, 0
            if len(fr.payload) >= 12:
                trained = struct.unpack_from("<I", fr.payload, 4)[0]  # offset 4 (after major, minor, pad)
                tested = struct.unpack_from("<I", fr.payload, 8)[0]   # offset 8
            
            logger.success(f"✅ READY received (received {log_frames_received} LOG frames during startup)")
            if trained > 0 or tested > 0:
                logger.info(f"📊 Device state: trained={trained}, tested={tested} samples since last reset")
            return (True, trained, tested)
        elif fr.type==FRAME_TYPE_LOG:
            log_frames_received += 1
            try:
                msg = fr.payload.decode('utf-8', errors='replace').rstrip()
                if not msg.startswith("CONFIG:"):
                    logger.debug(f"[ESP32] {msg}")
            except:
                pass
        # Ignore other frames (ACK/DONE/STAT) during initial sync
    logger.error("⏰ Timed out waiting for READY")
    return (False, 0, 0)

def select_runtime_model(ser: serial.Serial, model_name: str, timeout_s: float = 5.0) -> None:
    model_key = (model_name or "").strip().lower()
    if model_key not in MODEL_NAME_TO_ID:
        raise ValueError(f"Unknown model '{model_name}'")
    model_id = MODEL_NAME_TO_ID[model_key]
    seq = 1
    logger.info(f"🧠 Selecting runtime model: {model_key.upper()} (id={model_id})")
    seq = send_cmd(ser, seq, OPC_MODEL_SELECT, bytes([model_id]))
    ready_ok, trained, tested = wait_ready(ser, timeout_s=timeout_s)
    if not ready_ok:
        raise TimeoutError(f"READY not received after model select ({model_key})")
    logger.success(f"✅ Runtime model set to {model_key.upper()} (trained={trained}, tested={tested})")

def wait_ready_brief(ser: serial.Serial, timeout_s: float = 0.8) -> bool:
    """Best-effort short wait for READY (used for guarded phase setup)."""
    end = time.time() + timeout_s
    while time.time() < end:
        fr = read_frame(ser, timeout_s=min(0.2, max(0.02, end - time.time())))
        if not fr or not fr.checksum_ok:
            continue
        if fr.type == FRAME_TYPE_READY:
            return True
        if fr.type == FRAME_TYPE_LOG:
            try:
                msg = fr.payload.decode('utf-8', errors='replace').rstrip()
                if not msg.startswith("CONFIG:"):
                    logger.debug(f"[ESP32] {msg}")
            except Exception:
                pass
    return False

def parse_stat_payload(payload:bytes):
    if len(payload)!=12: return None
    tr=struct.unpack_from("<I",payload,0)[0]
    te=struct.unpack_from("<I",payload,4)[0]
    ac=struct.unpack_from("<f",payload,8)[0]
    return (tr,te,ac)

def wait_for_ack_until(ser:serial.Serial, target:int, timeout_s:float=10.0):
    """Wait until device has ACKed >= target records. Returns (acked_count, last_stat, saw_error)."""
    end=time.time()+timeout_s; last_stat=None; last_ack=None; saw_error=False
    while time.time()<end:
        fr=read_frame(ser, timeout_s=min(0.5, max(0.05, end-time.time())))
        if not fr or not fr.checksum_ok:
            continue
        if fr.type==FRAME_TYPE_ACK and len(fr.payload)>=8:
            cnt=struct.unpack_from("<I",fr.payload,0)[0]
            last_ack=cnt
            if cnt>=target: return cnt,last_stat,saw_error
        elif fr.type==FRAME_TYPE_STAT:
            ps=parse_stat_payload(fr.payload)
            if ps: last_stat=ps
        elif fr.type==FRAME_TYPE_DONE and len(fr.payload)>=8:
            cnt=struct.unpack_from("<I",fr.payload,0)[0]
            last_ack = max(last_ack or 0, cnt)
            if cnt>=target:
                return cnt,last_stat,saw_error
            # else keep waiting; DONE is just a beacon now
        elif fr.type==FRAME_TYPE_ERROR and fr.payload:
            code=fr.payload[0]
            saw_error=True
            logger.warning(f"⚠️  Device ERROR code={code}; continuing...")
        elif fr.type==FRAME_TYPE_READY:
            # Beacon; ignore
            pass
        elif fr.type==FRAME_TYPE_LOG:
            # Device log message - print it
            try:
                msg = fr.payload.decode('utf-8', errors='replace').rstrip()
                if not msg.startswith("CONFIG:"):
                    logger.debug(f"[ESP32] {msg}")
            except:
                pass
    return last_ack,last_stat,saw_error

def sync_get_done_count(ser: serial.Serial, seq: int, timeout_s: float = 8.0):
    """
    Issue OPC_SYNC and return (next_seq, done_count, last_stat).
    done_count is the absolute ingest counter from firmware.
    """
    seq = send_cmd(ser, seq, OPC_SYNC)
    deadline = time.time() + timeout_s
    last_stat = None
    while time.time() < deadline:
        fr = read_frame(ser, timeout_s=min(0.2, max(0.02, deadline - time.time())))
        if not fr or not fr.checksum_ok:
            continue
        if fr.type == FRAME_TYPE_DONE and len(fr.payload) >= 8:
            done_cnt = struct.unpack_from("<I", fr.payload, 0)[0]
            return seq, done_cnt, last_stat
        if fr.type == FRAME_TYPE_STAT:
            ps = parse_stat_payload(fr.payload)
            if ps:
                last_stat = ps
    return seq, None, last_stat

def drain_for_stats(ser:serial.Serial, drain_time_s:float=0.05):
    last=None; end=time.time()+drain_time_s
    while time.time()<end:
        fr=read_frame(ser,timeout_s=0.05)
        if not fr or not fr.checksum_ok: continue
        if fr.type==FRAME_TYPE_STAT:
            ps=parse_stat_payload(fr.payload)
            if ps: last=ps
        elif fr.type==FRAME_TYPE_ERROR and fr.payload:
            logger.warning(f"⚠️  Device ERROR code={fr.payload[0]} (stat drain)")
        elif fr.type==FRAME_TYPE_LOG:
            try:
                msg = fr.payload.decode('utf-8', errors='replace').rstrip()
                if not msg.startswith("CONFIG:"):
                    logger.debug(f"[ESP32] {msg}")
            except:
                pass
    return last

def send_phase_setup_guarded(ser: serial.Serial, seq: int, mode: str, window: int,
                             cfg_payload: bytes) -> int:
    """Conservative phase setup to avoid startup frame races in packed mode."""
    _ = drain_for_stats(ser, 0.02)
    seq = send_cmd(ser, seq, OPC_CONFIG, cfg_payload)
    # Keep this non-blocking-ish; long waits here can dominate phase time.
    _ = wait_ready_brief(ser, timeout_s=0.05)
    seq = send_cmd(ser, seq, OPC_BUFFER, struct.pack("<H", max(1, int(window))))
    seq = send_cmd(ser, seq, OPC_START_TRAIN if mode == "train" else OPC_START_TEST)
    _ = drain_for_stats(ser, 0.02)
    return seq

def request_snapshot(ser:serial.Serial, seq:int):
    seq=send_cmd(ser,seq,OPC_SNAPSHOT)
    t0=time.time()
    while time.time()-t0<3.0:
        fr=read_frame(ser,timeout_s=0.5)
        if fr and fr.checksum_ok and fr.type==FRAME_TYPE_METRIC and len(fr.payload)==(4+4+4+4+4+1+3):
            trained=struct.unpack_from("<I",fr.payload,0)[0]
            train_correct=struct.unpack_from("<I",fr.payload,4)[0]
            ema_acc=struct.unpack_from("<f",fr.payload,8)[0]
            ema_score=struct.unpack_from("<f",fr.payload,12)[0]
            last_score=struct.unpack_from("<i",fr.payload,16)[0]
            last_pred=fr.payload[20]
            return seq, {"trained":trained,"train_correct":train_correct,
                         "ema_train_acc":ema_acc,"ema_train_score":ema_score,
                         "last_score":last_score,"last_pred":last_pred}
    return seq,None

def request_memprof(ser:serial.Serial, seq:int):
    seq=send_cmd(ser,seq,OPC_MEMPROF)
    t0=time.time()
    while time.time()-t0<3.0:
        fr=read_frame(ser,timeout_s=0.5)
        if fr and fr.checksum_ok and fr.type==FRAME_TYPE_MEMPROF and len(fr.payload)>= (4*3 + 4*3 + 8*2 + 4*3 + 4 + 4 + 4):
            off=0
            trained = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            tested  = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            acc     = struct.unpack_from("<f",fr.payload,off)[0]; off+=4
            freehp  = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            minhp   = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            maxal   = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            util_cur= struct.unpack_from("<Q",fr.payload,off)[0]; off+=8
            util_pk = struct.unpack_from("<Q",fr.payload,off)[0]; off+=8
            alloc_c = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            free_c  = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            alive_c = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            total_e = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            sent_e  = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
            trunc_f = fr.payload[off]; off+=4  # truncated flag (1 byte) + pad (3)

            entries=[]
            for _ in range(sent_e):
                name_len = struct.unpack_from("<H",fr.payload,off)[0]; off+=2
                name = fr.payload[off:off+name_len].decode('utf-8','replace'); off+=name_len
                total_us = struct.unpack_from("<Q",fr.payload,off)[0]; off+=8
                count    = struct.unpack_from("<I",fr.payload,off)[0]; off+=4
                max_us   = struct.unpack_from("<Q",fr.payload,off)[0]; off+=8
                last_us  = struct.unpack_from("<Q",fr.payload,off)[0]; off+=8
                avg_us   = (total_us / count) if count else 0.0
                entries.append((name,total_us,count,avg_us,max_us,last_us))

            return seq, {
                "trained":trained,"tested":tested,"acc":acc,
                "free_heap":freehp,"min_heap":minhp,"max_alloc":maxal,
                "util_current":util_cur,"util_peak":util_pk,
                "util_alloc_count":alloc_c,"util_free_count":free_c,"util_active":alive_c,
                "entries_total":total_e,"entries_sent":sent_e,"truncated":bool(trunc_f),
                "entries":entries
            }
    return seq,None

def decode_mem_efficiency(payload: bytes) -> dict:
    """Decode memory efficiency analysis payload"""
    off = 0
    tm_mem = struct.unpack_from("<I", payload, off)[0]; off += 4
    mem_per_clause = struct.unpack_from("<f", payload, off)[0]; off += 4
    mem_per_feat = struct.unpack_from("<f", payload, off)[0]; off += 4
    mem_per_cf = struct.unpack_from("<f", payload, off)[0]; off += 4
    free_heap = struct.unpack_from("<I", payload, off)[0]; off += 4
    max_alloc = struct.unpack_from("<I", payload, off)[0]; off += 4
    frag = struct.unpack_from("<f", payload, off)[0]; off += 4
    cur_alloc = struct.unpack_from("<Q", payload, off)[0]; off += 8
    peak_alloc = struct.unpack_from("<Q", payload, off)[0]; off += 8
    mem_eff = struct.unpack_from("<f", payload, off)[0]; off += 4
    alloc_cnt = struct.unpack_from("<I", payload, off)[0]; off += 4
    free_cnt = struct.unpack_from("<I", payload, off)[0]; off += 4
    total_alloc = struct.unpack_from("<Q", payload, off)[0]; off += 8
    avg_alloc = struct.unpack_from("<f", payload, off)[0]; off += 4
    alloc_freq = struct.unpack_from("<f", payload, off)[0]; off += 4
    
    return {
        "tm_memory": tm_mem,
        "memory_per_clause": mem_per_clause,
        "memory_per_feature": mem_per_feat,
        "memory_per_cf": mem_per_cf,
        "free_heap": free_heap,
        "max_alloc": max_alloc,
        "fragmentation": frag,
        "current_allocated": cur_alloc,
        "peak_allocated": peak_alloc,
        "memory_efficiency": mem_eff,
        "alloc_count": alloc_cnt,
        "free_count": free_cnt,
        "total_allocated": total_alloc,
        "avg_alloc_size": avg_alloc,
        "alloc_frequency": alloc_freq,
    }

def decode_perf_bench(payload: bytes) -> dict:
    """Decode performance benchmarks payload"""
    off = 0
    train_tput = struct.unpack_from("<f", payload, off)[0]; off += 4
    test_tput = struct.unpack_from("<f", payload, off)[0]; off += 4
    mem_train = struct.unpack_from("<f", payload, off)[0]; off += 4
    mem_test = struct.unpack_from("<f", payload, off)[0]; off += 4
    tput_ratio = struct.unpack_from("<f", payload, off)[0]; off += 4
    chip_used = struct.unpack_from("<I", payload, off)[0]; off += 4
    tm_mem = struct.unpack_from("<I", payload, off)[0]; off += 4
    sys_ovh = struct.unpack_from("<I", payload, off)[0]; off += 4
    tm_pct = struct.unpack_from("<f", payload, off)[0]; off += 4
    ovh_pct = struct.unpack_from("<f", payload, off)[0]; off += 4
    avg_alloc = struct.unpack_from("<f", payload, off)[0]; off += 4
    alloc_ovh = struct.unpack_from("<f", payload, off)[0]; off += 4
    total_ovh = struct.unpack_from("<I", payload, off)[0]; off += 4
    ta_pct = struct.unpack_from("<f", payload, off)[0]; off += 4
    mem_util = struct.unpack_from("<f", payload, off)[0]; off += 4
    perf_score = struct.unpack_from("<f", payload, off)[0]; off += 4
    eff_ratio = struct.unpack_from("<f", payload, off)[0]; off += 4
    precision = struct.unpack_from("<f", payload, off)[0]; off += 4
    recall = struct.unpack_from("<f", payload, off)[0]; off += 4
    f1_score = struct.unpack_from("<f", payload, off)[0]; off += 4
    
    return {
        "train_throughput": train_tput,
        "test_throughput": test_tput,
        "mem_per_train_sample": mem_train,
        "mem_per_test_sample": mem_test,
        "throughput_mem_ratio": tput_ratio,
        "chip_used_heap": chip_used,
        "tm_core_memory": tm_mem,
        "sys_overhead": sys_ovh,
        "tm_mem_percent": tm_pct,
        "overhead_percent": ovh_pct,
        "avg_alloc_size": avg_alloc,
        "alloc_overhead_pct": alloc_ovh,
        "total_overhead": total_ovh,
        "ta_percentage": ta_pct,
        "memory_utilization": mem_util,
        "performance_score": perf_score,
        "efficiency_ratio": eff_ratio,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }

def decode_protocol_stats(payload: bytes) -> dict:
    """Decode protocol statistics payload"""
    off = 0
    frames_rx = struct.unpack_from("<I", payload, off)[0]; off += 4
    frames_valid = struct.unpack_from("<I", payload, off)[0]; off += 4
    rej_size = struct.unpack_from("<I", payload, off)[0]; off += 4
    rej_chk = struct.unpack_from("<I", payload, off)[0]; off += 4
    rej_fmt = struct.unpack_from("<I", payload, off)[0]; off += 4
    rej_rate = struct.unpack_from("<f", payload, off)[0]; off += 4
    
    return {
        "frames_received": frames_rx,
        "frames_valid": frames_valid,
        "rejected_size": rej_size,
        "rejected_checksum": rej_chk,
        "rejected_format": rej_fmt,
        "rejection_rate": rej_rate,
    }

def stream_phase(ser:serial.Serial, paths:List[Path], mode:str, window:int,
                 samples_cap:Optional[int], throttle_ms:int, status_every_s:float=2.0,
                 quiet: bool = False, multiclass: bool = False,
                 use_data_checksum: bool = True,
                 preproc_id: int = PREPROC_NONE,
                 preproc_param0: int = 0,
                 tm_runtime_cfg: Optional[Dict[str, int]] = None):
    assert mode in ("train","test")
    phase_start = time.time()
    logger.debug(f"⏱️ [{mode.upper()}] Phase starting...")
    total_in=count_records(paths)
    target=min(samples_cap,total_in) if samples_cap is not None else total_in
    if not quiet:
        logger.info(f"{mode.upper()}: files={len(paths)} found={total_in} target={target}")
    seq=1
    
    # No pre-switch STAT needed anymore; we use low-latency OPC_SYNC after streaming
    
    w=max(1,int(window));
    if not quiet:
        logger.info(f"window={w}")
    if not quiet:
        logger.info(f"mode: {'TRAIN' if mode=='train' else 'TEST'}")
    # Reassert bit-packed input mode at each phase boundary for robustness.
    _, cfg_payload = build_config_payload(
        input_real=False,
        preproc_id=preproc_id,
        preproc_param0=preproc_param0,
        tm_runtime=tm_runtime_cfg,
    )
    seq=send_cmd(ser, seq, OPC_CONFIG, cfg_payload)
    seq=send_cmd(ser,seq,OPC_BUFFER,struct.pack("<H",w))
    seq=send_cmd(ser,seq, OPC_START_TRAIN if mode=="train" else OPC_START_TEST)
    seq, phase_base_count, _ = sync_get_done_count(ser, seq, timeout_s=2.0)
    if phase_base_count is None:
        phase_base_count = 0

    sent=0; total_bytes=0; batch=0; last_stat=None; last_ping=time.time(); send_buf=bytearray()
    BUF_FLUSH_THRESHOLD = 64*1024  # Avoid huge host-side buffer.
    ACK_TIMEOUT_S = 2.0
    # Keep per-window ACK pacing for reliability.
    ACK_BATCH_WINDOWS = 1
    ack_batch = max(w * ACK_BATCH_WINDOWS, w)
    pbar=tqdm(total=target,desc=(f"{mode}"),unit="smp",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
              leave=False,
              disable=quiet,
              file=sys.stderr)  # Write to stderr to match Loguru output
    try:
        for p in paths:
            row_iter = iter_records_mc(p) if multiclass else iter_records(p)
            for rec in row_iter:
                if sent>=target: break
                now=time.time()
                if status_every_s and status_every_s>0 and (now-last_ping)>=status_every_s:
                    seq=send_cmd(ser,seq,OPC_STATUS)
                    last_ping=now

                frame_len = append_frame(
                    send_buf,
                    FRAME_TYPE_RECORD_MC if multiclass else FRAME_TYPE_RECORD,
                    rec,
                    seq,
                    use_data_checksum,
                )
                seq=(seq+1)&0xFFFFFFFF
                sent+=1; batch+=1; total_bytes+=frame_len; pbar.update(1)

                if throttle_ms>0: time.sleep(throttle_ms/1000.0)

                # Flush when we hit a window or large buffer
                if batch>=ack_batch or len(send_buf)>=BUF_FLUSH_THRESHOLD:
                    if send_buf:
                        write_all(ser, send_buf, timeout_s=5.0); send_buf.clear()
                    ack_target = phase_base_count + sent
                    ack,stat,saw_err=wait_for_ack_until(ser,ack_target,timeout_s=ACK_TIMEOUT_S)
                    if ack is None or ack < ack_target:
                        # Soft resync: request STAT and reduce window to relieve backpressure
                        try:
                            seq=send_cmd(ser,seq,OPC_STATUS)
                            _=drain_for_stats(ser,0.1)
                        except Exception:
                            pass
                        if w>1:
                            new_w=max(1,w//2)
                            if new_w!=w:
                                seq=send_cmd(ser,seq,OPC_BUFFER,struct.pack("<H",new_w))
                                w=new_w
                                ack_batch = max(w * ACK_BATCH_WINDOWS, w)
                        # Re-wait for the same ACK target after backoff. Do not advance sender
                        # state until the in-flight batch is actually acknowledged.
                        ack2,stat2,_=wait_for_ack_until(ser,ack_target,timeout_s=8.0)
                        if ack2 is None or ack2 < ack_target:
                            local_ack = (ack2 - phase_base_count) if ack2 is not None else None
                            raise RuntimeError(f"{mode.upper()} ACK stalled at {local_ack} / {sent}")
                        if stat2:
                            last_stat=stat2
                        batch=0
                        continue
                    if saw_err and w > 1:
                        # Error seen but ACK target was reached; gently back off window without blocking.
                        new_w = max(1, w // 2)
                        if new_w != w:
                            try:
                                seq = send_cmd(ser, seq, OPC_BUFFER, struct.pack("<H", new_w))
                                w = new_w
                                ack_batch = max(w * ACK_BATCH_WINDOWS, w)
                            except Exception:
                                pass
                    if stat:
                        last_stat=stat; tr,te,ac=stat
                        if not quiet:
                            pbar.set_postfix_str((f"tested={te} acc={ac:.3f}") if mode=='test' else (f"trained={tr}"))
                    batch=0
            if sent>=target: break
        if batch>0:
            if send_buf:
                write_all(ser, send_buf, timeout_s=5.0); send_buf.clear()
            # Do not block on partial-window ACK here; OPC_SYNC below verifies drain.
            stat=drain_for_stats(ser,0.05)
            if stat:
                last_stat=stat; tr,te,ac=stat
                if mode=='test' and not quiet: pbar.set_postfix_str(f"tested={te} acc={ac:.3f}")
    finally:
        pbar.close()
    # Ensure device has processed all sent samples (compute drained queue)
    # Use explicit SYNC command for low-latency drain completion
    try:
        wait_start = time.time()
        logger.debug(f"[{mode.upper()}] SYNC: waiting for device queue drain...")
        seq = send_cmd(ser, seq, OPC_SYNC)
        # Wait for DONE emitted by firmware on drain complete
        end = time.time() + 8.0
        while time.time() < end:
            fr = read_frame(ser, timeout_s=0.05)
            if not fr:
                continue
            if fr.type == FRAME_TYPE_DONE and fr.checksum_ok:
                done_cnt = None
                if len(fr.payload) >= 8:
                    done_cnt = struct.unpack_from("<I", fr.payload, 0)[0]
                # Accept SYNC only when DONE count confirms all sent records were ingested.
                done_target = phase_base_count + sent
                if done_cnt is not None and done_cnt >= done_target:
                    wait_time = (time.time() - wait_start) * 1000
                    logger.debug(f"[{mode.upper()}] ✓ SYNC DONE in {wait_time:.0f}ms (count={done_cnt - phase_base_count}/{sent})")
                    break
                continue
            if fr.type == FRAME_TYPE_STAT and fr.checksum_ok:
                ps = parse_stat_payload(fr.payload)
                if ps:
                    last_stat = ps
        else:
            logger.warning(f"[{mode.upper()}] ⚠️ SYNC timeout (queue drain) after 8.0s")
    except Exception as e:
        logger.debug(f"[{mode.upper()}] Exception during SYNC: {e}")
    
    # Fetch final STAT after drain to report tested count and accuracy
    try:
        seq = send_cmd(ser, seq, OPC_STATUS)
        st = drain_for_stats(ser, 0.1)
        if st:
            last_stat = st
    except Exception:
        pass

    phase_total = time.time() - phase_start
    logger.debug(f"⏱️ [{mode.upper()}] Phase completed in {phase_total:.3f}s total")
    return sent,total_bytes,last_stat

def stream_phase_raw(ser:serial.Serial, X, y, mode:str, window:int,
                     samples_cap:Optional[int], throttle_ms:int, status_every_s:float=2.0,
                     quiet: bool = False, csv_bitpack: bool = False,
                     use_data_checksum: bool = True, batch_records: int = 1,
                     preproc_id: int = PREPROC_NONE, preproc_param0: int = 0,
                     tm_runtime_cfg: Optional[Dict[str, int]] = None):
    assert mode in ("train","test")
    phase_start = time.time()
    logger.debug(f"⏱️ [{mode.upper()}] Phase starting...")
    total_in = X.shape[0]
    target = min(samples_cap,total_in) if samples_cap is not None else total_in
    if not quiet:
        logger.info(f"{mode.upper()}: rows={total_in} target={target}")
    seq=1
    w=max(1,int(window));
    if not quiet:
        logger.info(f"window={w}")
    # Reassert feature mode at each phase boundary.
    # raw float32 path: INPUT_REAL=1, csv-bitpack path: INPUT_REAL=0.
    phase_preproc_id = PREPROC_NONE if csv_bitpack else preproc_id
    phase_preproc_param0 = 0 if csv_bitpack else preproc_param0
    _, cfg_payload = build_config_payload(
        input_real=(not csv_bitpack),
        preproc_id=phase_preproc_id,
        preproc_param0=phase_preproc_param0,
        tm_runtime=tm_runtime_cfg,
    )
    if not quiet:
        logger.info(f"mode: {'TRAIN' if mode=='train' else 'TEST'}")
    # Fast setup path for high-throughput unsafe batch mode.
    # Guarded setup keeps extra READY/stat drains for robustness.
    use_fast_setup = csv_bitpack and (not use_data_checksum) and (batch_records > 1)
    if csv_bitpack and not use_fast_setup:
        seq = send_phase_setup_guarded(ser, seq, mode, w, cfg_payload)
    else:
        seq=send_cmd(ser, seq, OPC_CONFIG, cfg_payload)
        seq=send_cmd(ser,seq,OPC_BUFFER,struct.pack("<H",w))
        seq=send_cmd(ser,seq, OPC_START_TRAIN if mode=="train" else OPC_START_TEST)
    seq, phase_base_count, _ = sync_get_done_count(ser, seq, timeout_s=2.0)
    if phase_base_count is None:
        phase_base_count = 0

    bitpack_payloads = None
    labeled_bitpack_rows = None
    bitpack_nfeat = 0
    if csv_bitpack:
        labeled_bitpack_rows, bitpack_nfeat = prepare_labeled_bitpack_rows(X, y, target)
        if labeled_bitpack_rows is None:
            bitpack_payloads = pack_records_from_real_matrix(X, y, target)

    # Unsafe batch mode can tolerate less frequent host-side ACK waits.
    can_use_batch = csv_bitpack and (not use_data_checksum) and batch_records > 1

    sent=0; total_bytes=0; batch=0; last_stat=None; last_ping=time.time(); send_buf=bytearray()
    BUF_FLUSH_THRESHOLD = 64*1024
    ACK_TIMEOUT_S = 2.0
    # In unsafe batch mode we can amortize ACK wait overhead across more samples.
    ACK_BATCH_WINDOWS = 8 if can_use_batch else 1
    ack_batch = max(w * ACK_BATCH_WINDOWS, w)
    pbar=tqdm(total=target,desc=(f"{mode}"),unit="smp",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
              leave=False,
              disable=quiet,
              file=sys.stderr)
    try:
        def flush_and_wait_ack() -> None:
            nonlocal seq, w, ack_batch, batch, last_stat
            if send_buf:
                write_all(ser, send_buf, timeout_s=5.0)
                send_buf.clear()
            ack_target = phase_base_count + sent
            ack,stat,saw_err = wait_for_ack_until(ser, ack_target, timeout_s=ACK_TIMEOUT_S)
            if ack is None or ack < ack_target:
                try:
                    seq = send_cmd(ser, seq, OPC_STATUS)
                    _ = drain_for_stats(ser, 0.1)
                except Exception:
                    pass
                if w > 1:
                    new_w = max(1, w // 2)
                    if new_w != w:
                        seq = send_cmd(ser, seq, OPC_BUFFER, struct.pack("<H", new_w))
                        w = new_w
                        ack_batch = max(w * ACK_BATCH_WINDOWS, w)
                ack2,stat2,_ = wait_for_ack_until(ser, ack_target, timeout_s=8.0)
                if ack2 is None or ack2 < ack_target:
                    local_ack = (ack2 - phase_base_count) if ack2 is not None else None
                    raise RuntimeError(f"{mode.upper()} ACK stalled at {local_ack} / {sent}")
                if stat2:
                    last_stat = stat2
                batch = 0
                return
            if saw_err and w > 1:
                new_w = max(1, w // 2)
                if new_w != w:
                    try:
                        seq = send_cmd(ser, seq, OPC_BUFFER, struct.pack("<H", new_w))
                        w = new_w
                        ack_batch = max(w * ACK_BATCH_WINDOWS, w)
                    except Exception:
                        pass
            if stat:
                last_stat = stat
                tr,te,ac = stat
                if mode == 'test' and not quiet:
                    pbar.set_postfix_str(f"tested={te} acc={ac:.3f}")
            batch = 0

        # Unsafe fast path: pack multiple bit-packed rows into one batch frame.
        if can_use_batch:
            idx = 0
            total_rows = int(labeled_bitpack_rows.shape[0]) if labeled_bitpack_rows is not None else len(bitpack_payloads)
            while idx < total_rows and sent < target:
                now = time.time()
                if status_every_s and status_every_s > 0 and (now - last_ping) >= status_every_s:
                    seq = send_cmd(ser, seq, OPC_STATUS)
                    last_ping = now

                n_take = min(int(batch_records), total_rows - idx, target - sent)
                if labeled_bitpack_rows is not None:
                    frame_len = append_record_batch_frame_from_labeled_rows(
                        send_buf, labeled_bitpack_rows, bitpack_nfeat, idx, n_take, seq, use_data_checksum
                    )
                else:
                    chunk = bitpack_payloads[idx:idx + n_take]
                    payload = pack_record_batch_payload(chunk)
                    frame_len = append_frame(send_buf, FRAME_TYPE_RECORD_BATCH, payload, seq, use_data_checksum)
                seq = (seq + 1) & 0xFFFFFFFF
                sent += n_take
                batch += n_take
                total_bytes += frame_len
                pbar.update(n_take)
                idx += n_take

                if throttle_ms > 0:
                    time.sleep(throttle_ms / 1000.0)
                if batch >= ack_batch or len(send_buf) >= BUF_FLUSH_THRESHOLD:
                    flush_and_wait_ack()
        else:
            row_iter = ((None, None) for _ in range(0))  # placeholder
            if csv_bitpack:
                if labeled_bitpack_rows is not None:
                    row_iter = ((i, None) for i in range(int(labeled_bitpack_rows.shape[0])))
                else:
                    row_iter = ((payload, None) for payload in bitpack_payloads)
            else:
                row_iter = iter_rows_raw(X, y)

            for item0, item1 in row_iter:
                if sent>=target: break
                now=time.time()
                if status_every_s and status_every_s>0 and (now-last_ping)>=status_every_s:
                    seq=send_cmd(ser,seq,OPC_STATUS)
                    last_ping=now
                if csv_bitpack:
                    if labeled_bitpack_rows is not None:
                        ridx = int(item0)
                        lbl = int(labeled_bitpack_rows[ridx, 0] & 1)
                        header = ((1 if lbl else 0) << 31) | (bitpack_nfeat & 0x7FFFFFFF)
                        payload = struct.pack("<I", header) + labeled_bitpack_rows[ridx, 1:].tobytes(order='C')
                    else:
                        payload = item0
                    frame_len = append_frame(send_buf, FRAME_TYPE_RECORD, payload, seq, use_data_checksum)
                else:
                    feats,label = item0, item1
                    payload = pack_raw_sample_row(feats, label)
                    frame_len = append_frame(send_buf, FRAME_TYPE_SAMPLE_RAW, payload, seq, use_data_checksum)
                seq=(seq+1)&0xFFFFFFFF
                sent+=1; batch+=1; total_bytes+=frame_len; pbar.update(1)
                if throttle_ms>0: time.sleep(throttle_ms/1000.0)
                if batch>=ack_batch or len(send_buf)>=BUF_FLUSH_THRESHOLD:
                    flush_and_wait_ack()
        if batch>0:
            if send_buf:
                write_all(ser, send_buf, timeout_s=5.0); send_buf.clear()
            stat=drain_for_stats(ser,0.05)
            if stat:
                last_stat=stat; tr,te,ac=stat
                if mode=='test' and not quiet: pbar.set_postfix_str(f"tested={te} acc={ac:.3f}")
    finally:
        pbar.close()

    # SYNC and final STAT
    logger.debug(f"[{mode.upper()}] SYNC: waiting for device queue drain...")
    seq = send_cmd(ser, seq, OPC_SYNC)
    sync_done = False
    sync_timeout = time.time() + 8.0
    while time.time() < sync_timeout:
        fr = read_frame(ser, timeout_s=0.05)
        if fr and fr.checksum_ok and fr.type == FRAME_TYPE_DONE:
            done_cnt = None
            if len(fr.payload) >= 8:
                done_cnt = struct.unpack_from("<I", fr.payload, 0)[0]
            done_target = phase_base_count + sent
            if done_cnt is not None and done_cnt >= done_target:
                sync_done = True
                break
            continue
        elif fr and fr.checksum_ok and fr.type == FRAME_TYPE_STAT:
            last_stat = parse_stat_payload(fr.payload)
    if not sync_done:
        logger.warning(f"[{mode.upper()}] ⚠️ SYNC timed out")
    seq = send_cmd(ser, seq, OPC_STATUS)
    final_deadline = time.time() + 0.3
    while time.time() < final_deadline:
        fr = read_frame(ser, timeout_s=0.01)
        if fr and fr.checksum_ok and fr.type == FRAME_TYPE_STAT:
            last_stat = parse_stat_payload(fr.payload)
            break
    phase_total = time.time() - phase_start
    logger.debug(f"⏱️ [{mode.upper()}] Phase completed in {phase_total:.3f}s total")
    return sent,total_bytes,last_stat

def auto_reset_and_wait_ready(ser:serial.Serial, timeout_s:float=15.0):
    """Reset ESP32 and wait for READY, handling serial port disconnection gracefully."""
    import serial.serialutil
    
    port = ser.port
    baudrate = ser.baudrate
    
    try:
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        time.sleep(0.05)
        
        seq = 1
        logger.info("🔁 Sending binary RESET (OPC_RESET)...")
        seq = send_cmd(ser, seq, OPC_RESET)
        ser.flush()
        time.sleep(0.2)
        
        # After reset, the USB serial may disconnect momentarily on Windows
        # Try to read, but if we get a serial error, close and reopen
        try:
            ser.reset_input_buffer()
        except (serial.serialutil.SerialException, OSError):
            logger.warning("⚠️  Serial port disconnected during reset (expected on Windows)")
            logger.info("🔄 Waiting for device to reconnect...")
            ser.close()
            time.sleep(2.0)  # Wait for USB re-enumeration
            
            # Reopen the port
            for attempt in range(5):
                try:
                    ser.open()
                    logger.success(f"✅ Serial port reconnected (attempt {attempt + 1})")
                    time.sleep(0.5)
                    break
                except (serial.serialutil.SerialException, OSError):
                    if attempt < 4:
                        time.sleep(1.0)
                    else:
                        raise TimeoutError("Failed to reconnect to serial port after reset")
    except Exception as e:
        logger.warning(f"⚠️  Error during reset: {e}")
        # Try to recover by reopening
        if not ser.is_open:
            try:
                ser.open()
            except:
                pass
    
    # Clear any stale frames before waiting for fresh READY
    time.sleep(0.3)  # Give ESP32 time to boot and send first READY
    ser.reset_input_buffer()  # Flush any buffered data
    time.sleep(0.1)
    
    ready_ok, _, _ = wait_ready(ser, timeout_s)
    if not ready_ok:
        raise TimeoutError("READY not received after RESET")

def discover_data_files(data_dir:Path):
    train=sorted(Path(p) for p in glob.glob(str(data_dir/"*train*.txt")))
    test =sorted(Path(p) for p in glob.glob(str(data_dir/"*test*.txt")))
    if not train:
        p=data_dir/"train.txt"
        if p.exists(): train=[p]
    if not test:
        p=data_dir/"test.txt"
        if p.exists(): test=[p]
    return train,test

def ensure_mnist_available(base_dir: Path) -> bool:
    """Ensure MNIST txt files exist under data/mnist; download if missing."""
    mnist_dir = base_dir / "mnist"
    train_txt = mnist_dir / "mnist_train.txt"
    test_txt  = mnist_dir / "mnist_test.txt"
    if train_txt.exists() and test_txt.exists():
        return True
    try:
        logger.info("🧲 MNIST not found; downloading via scripts/download_mnist.py --pool avg4 ...")
        cmd = [sys.executable, str(Path("scripts")/"download_mnist.py"), "--pool", "avg4"]
        rc = subprocess.run(cmd, cwd=str(Path.cwd())).returncode
        if rc != 0:
            logger.error(f"MNIST download script failed (rc={rc})")
            return False
    except Exception as e:
        logger.error(f"MNIST download error: {e}")
        return False
    ok = train_txt.exists() and test_txt.exists()
    if ok:
        logger.success("✅ MNIST prepared under data/mnist/")
    else:
        logger.error("❌ MNIST files still missing after download")
    return ok

def main():
    ap=argparse.ArgumentParser(description="ESP32 TM v2 binary streamer")

    # Connection
    grp_conn = ap.add_argument_group("Connection")
    grp_conn.add_argument("-p","--port", default="COM3")
    grp_conn.add_argument("-b","--baud", type=int, default=2000000)
    grp_conn.add_argument("--no-hw-reset", action="store_true", help="Skip hardware reset at startup (use software reset only)")
    grp_conn.add_argument("--wokwi", action="store_true", help="Connect to Wokwi simulator (uses rfc2217://localhost:4000)")
    grp_conn.add_argument("--desktop", action="store_true", help="Connect to desktop native build (uses socket://localhost:5555)")
    grp_conn.add_argument("--js-power", action="store_true", help="Measure current, voltage, power and energy via Joulescope (separate train/test)")
    grp_conn.add_argument("--js-serial", type=str, default=None, help="Joulescope device serial (default: use first device)")
    grp_conn.add_argument("--js-poll-ms", type=float, default=100.0, help="Joulescope sampling interval in ms (default: 100)")
    grp_conn.add_argument("--js-keep-usb-warnings", action="store_true",
                          help="Do not suppress low-level USB bulk warnings emitted by the Joulescope driver")

    # Data
    grp_data = ap.add_argument_group("Data")
    grp_data.add_argument("--data-dir", default="data")
    grp_data.add_argument("--dataset", type=str, default=None,
                          help="Dataset name: loads data/<name>/*.txt or auto-detects data/<name>.csv")
    # Streaming
    grp_stream = ap.add_argument_group("Streaming")
    grp_stream.add_argument("--window", type=int, default=256, help="In-flight frames window size")
    grp_stream.add_argument("--batch-records", type=int, default=32,
                            help="Records per batch frame in unsafe csv-bitpack mode")
    grp_stream.add_argument("--throttle-ms", type=int, default=0, help="Sender throttle between records (ms)")
    grp_stream.add_argument("--status-every", type=float, default=0.0, help="Send STAT every N seconds during TRAIN")

    # Experiment
    grp_exp = ap.add_argument_group("Experiment")
    grp_exp.add_argument("--epochs", type=int, default=1, help="Number of training epochs to run (cumulative training)")
    grp_exp.add_argument("--runs", type=int, default=1, help="Number of experimental runs for statistical analysis")
    grp_exp.add_argument("--shuffle-seed", type=int, default=42,
                         help="Base seed for deterministic per-run train/test shuffling")
    grp_exp.add_argument("--no-run-shuffle", action="store_true",
                         help="Disable per-run shuffling (use fixed dataset order each run)")
    grp_exp.add_argument("--train-samples", type=int, default=None, help="Cap training samples (per epoch)")
    grp_exp.add_argument("--test-samples", type=int, default=None, help="Cap test samples (per epoch)")
    grp_exp.add_argument("--eval-samples", type=int, default=None, help="Test samples per epoch (default: --test-samples)")
    grp_exp.add_argument("--train-frac", type=float, default=0.8, help="Training fraction for train/test split (default: 0.8 = 80%% train, 20%% test)")
    grp_exp.add_argument("--save-epoch-results", action="store_true", help="Save intermediate results after each epoch")
    grp_exp.add_argument("--save-run-stats", action="store_true", help="Save detailed statistics across all runs")
    grp_exp.add_argument("--snapshot", action="store_true", help="Request training snapshot metrics each epoch")
    grp_exp.add_argument("--dump-prof", action="store_true", help="Print per-function profiler table after training")
    grp_exp.add_argument("--final-logs", action="store_true", help="Collect verbose LOG frames during final analysis (slower)")
    grp_exp.add_argument("--unsafe-no-data-checksum", action="store_true",
                         help="UNSAFE: disable Fletcher checksum on data frames for max throughput")

    # Model & Preprocessing
    grp_prep = ap.add_argument_group("Preprocessing")
    grp_prep.add_argument("--model", type=str, default=None,
                          choices=sorted(MODEL_NAME_TO_ID.keys()),
                          help="Runtime model selection (single firmware supports all models)")
    grp_prep.add_argument("--input-real", action="store_true", help="Send real-valued rows (float32) instead of bit-packed features")
    grp_prep.add_argument("--csv-bitpack", action="store_true", help="For CSV datasets: binarize (>0) and bit-pack features for higher transport throughput")
    grp_prep.add_argument("--online-preproc", choices=sorted(PREPROC_NAME_TO_ID.keys()), default="none",
                          help="On-device preprocessing: none, standardize, or twine")
    grp_prep.add_argument("--twine-bits", type=int, default=1,
                          help="TWINE quantization bits (1-8) when --online-preproc twine")
    grp_prep.add_argument("--tm-clauses", type=int, default=None,
                          help="Runtime TM clause count (sent via OPC_CONFIG, no reflash)")
    grp_prep.add_argument("--tm-threshold", type=int, default=None,
                          help="Runtime TM threshold T (sent via OPC_CONFIG, no reflash)")
    grp_prep.add_argument("--tm-specificity", type=int, default=None,
                          help="Runtime TM specificity s (sent via OPC_CONFIG, no reflash)")
    grp_prep.add_argument("--tm-seed", type=int, default=None,
                          help="Runtime TM RNG seed (sent via OPC_CONFIG, no reflash)")
    grp_prep.add_argument("--tm-init-density", type=int, default=None,
                          help="Runtime TM sparse/BO init literal density pct [0..100] (sent via OPC_CONFIG)")
    grp_prep.add_argument("--host-standardize", dest="host_standardize", action="store_true", default=True,
                          help="Apply StandardScaler on host before streaming raw samples (default: enabled)")
    grp_prep.add_argument("--no-host-standardize", dest="host_standardize", action="store_false",
                          help="Disable host-side StandardScaler")
    grp_prep.add_argument("--model-tag", type=str, default="", help="Optional model label stored in logs/results filenames")
    args=ap.parse_args()
    
    # Override port settings if using Wokwi simulator
    if args.wokwi:
        args.port = "rfc2217://localhost:4000"
        args.no_hw_reset = True  # Always skip hardware reset for simulator
        logger.info("🎮 Wokwi mode: connecting to rfc2217://localhost:4000")
    
    # Override port settings if using desktop native build
    if args.desktop:
        args.port = "socket://localhost:5555"
        args.no_hw_reset = True  # Always skip hardware reset for desktop
        logger.info("💻 Desktop mode: connecting to socket://localhost:5555")
    elif args.port.startswith("socket://"):
        args.no_hw_reset = True
        logger.info(f"💻 Socket port specified ({args.port}); skipping hardware reset")

    selected_preproc_id, selected_preproc_param0 = resolve_preproc(args.online_preproc, args.twine_bits)
    host_standardize_effective = bool(args.host_standardize) and (selected_preproc_id == PREPROC_NONE)
    if args.host_standardize and not host_standardize_effective:
        logger.info("🧪 Host StandardScaler disabled because on-device preprocessor is enabled")
    tm_runtime_cfg = resolve_tm_runtime_config(args)
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Generate unique filename with timestamp and configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = (args.model_tag or "").strip().replace(" ", "_")
    if model_tag:
        config_name = f"{model_tag}_w{args.window}_t{args.throttle_ms}"
    else:
        config_name = f"w{args.window}_t{args.throttle_ms}"
    log_file = results_dir / f"experiment_log_{config_name}_{timestamp}.txt"
    
    # Configure Loguru
    logger.remove()  # Remove default handler
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",  # Changed to DEBUG to see debug messages
        colorize=True
    )
    # Add file handler (plain text, no colors)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        level="INFO",
        encoding="utf-8"
    )
    
    logger.info(f"📁 Results directory: {results_dir}")
    logger.info(f"🕐 Timestamp: {timestamp}")
    logger.info(f"⚙️  Configuration: window={args.window}, throttle={args.throttle_ms}ms")
    if args.model:
        logger.info(f"🧠 Runtime model: {args.model.upper()}")
    if tm_runtime_cfg is not None:
        logger.info(
            "🧠 TM runtime config: "
            f"clauses={tm_runtime_cfg['clauses']} "
            f"T={tm_runtime_cfg['threshold']} "
            f"s={tm_runtime_cfg['specificity']} "
            f"seed={tm_runtime_cfg['seed']} "
            f"init_density={tm_runtime_cfg['init_density_pct']}"
        )
    twine_info = f", bits={selected_preproc_param0}" if selected_preproc_id == PREPROC_TWINE else ""
    logger.info(f"🧩 Online preprocessor: {args.online_preproc}{twine_info}")
    logger.info(f"🧮 Host StandardScaler: {'on' if host_standardize_effective else 'off'}")
    if args.unsafe_no_data_checksum:
        logger.warning("⚠️ UNSAFE mode enabled: data-frame checksums are disabled")
    if args.batch_records > 1:
        logger.info(f"📦 Batch records per frame: {args.batch_records}")
    if args.wokwi:
        logger.info("🎮 Mode: Wokwi Simulator (rfc2217://localhost:4000)")
    elif args.desktop:
        logger.info("💻 Mode: Desktop Native Build (socket://localhost:5555)")
    logger.info("=" * 70)

    using_csv = False
    using_memory_dataset = False
    Xtr=Ytr=Xte=Yte=None
    # Ensure these exist for both modes
    train_paths: List[Path] = []
    test_paths: List[Path] = []
    csv_meta = None
    text_meta = None

    # Resolve dataset directory (e.g., data/mnist or data/iiot)
    base_dir = Path(args.data_dir)
    dataset_dir = base_dir / args.dataset if args.dataset else base_dir

    # Auto-prepare MNIST if requested and missing
    if args.dataset and args.dataset.lower() == "mnist":
        if not ensure_mnist_available(base_dir):
            raise SystemExit("Failed to prepare MNIST dataset (download_mnist.py)")

    # If dataset name is provided, try CSV auto-detection: data/<name>.csv or data/<name>/<name>.csv
    csv_candidate = None
    if args.dataset:
        candidates = [base_dir / f"{args.dataset}.csv", dataset_dir / f"{args.dataset}.csv"]
        for c in candidates:
            if c.exists():
                csv_candidate = c
                break
        # If not found yet, look for a single CSV under data/<name>
        if not csv_candidate and dataset_dir.exists() and dataset_dir.is_dir():
            csv_files = list(dataset_dir.glob("*.csv"))
            if len(csv_files) == 1:
                csv_candidate = csv_files[0]

    if csv_candidate is not None:
        using_csv = True
        csv_path = csv_candidate
        logger.info(f"📄 CSV dataset detected: {csv_path}")
        (Xtr, Ytr), (Xte, Yte), norm_info = load_csv_dataset(csv_path, None, args.train_frac, True, 42)
        
        if host_standardize_effective:
            if StandardScaler is not None:
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr).astype(np.float32)
                Xte = scaler.transform(Xte).astype(np.float32)
                logger.info("✨ StandardScaler applied on host (fit on train, transform on train+test)")
            else:
                logger.warning("⚠️  sklearn not available, skipping normalization")
        else:
            logger.info("🧮 Host-side normalization disabled for this run")
        
        if args.csv_bitpack:
            args.input_real = False
            logger.info("🧮 CSV bit-pack mode: host binarize (>0) + packed transport")
        else:
            if not args.input_real:
                args.input_real = True
            logger.info("🧮 Streaming CSV rows as raw float32 samples (input_real enabled)")
            logger.info("🧠 Quantizer: disabled (streaming raw float32 samples)")
        csv_meta = {
            'path': str(csv_path),
        }
    else:
        # Fallback to text file discovery under (possibly named) dataset directory
        data_dir = dataset_dir
        train_paths, test_paths = discover_data_files(data_dir)
        if not train_paths: raise SystemExit("No training files found.")
        if not test_paths:  raise SystemExit("No test files found.")
        for p in train_paths+test_paths:
            if not p.exists(): raise SystemExit(f"Input not found: {p}")

        if args.input_real:
            Xtr, Ytr = load_txt_dataset_as_array(train_paths, multiclass=False)
            Xte, Yte = load_txt_dataset_as_array(test_paths, multiclass=False)
            using_memory_dataset = True
            
            if host_standardize_effective:
                if StandardScaler is not None:
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
                    Xte = scaler.transform(Xte).astype(np.float32)
                    logger.info("✨ StandardScaler applied on host (fit on train, transform on train+test)")
                else:
                    logger.warning("⚠️  sklearn not available, skipping normalization")
            else:
                logger.info("🧮 Host-side normalization disabled for this run")
            
            logger.info("🧮 Streaming text dataset as raw float32 samples (input_real enabled)")
            text_meta = {
                'train_files': [str(p) for p in train_paths],
                'test_files': [str(p) for p in test_paths],
            }

    effective_preproc_id = selected_preproc_id
    effective_preproc_param0 = selected_preproc_param0
    if not args.input_real and effective_preproc_id != PREPROC_NONE:
        logger.warning("⚠️  On-device preprocessors require input_real/raw samples; disabling preprocessor for packed mode")
        effective_preproc_id = PREPROC_NONE
        effective_preproc_param0 = 0
    effective_preproc_name = PREPROC_ID_TO_NAME.get(effective_preproc_id, "none")

    logger.info(f"🔌 Opening {args.port} @ {args.baud}...")
    
    # Handle network URLs (for Wokwi simulator) using serial_for_url
    if args.port.startswith("rfc2217://") or args.port.startswith("socket://"):
        # Use serial_for_url which can handle various URL schemes
        # Note: rfc2217:// and socket:// don't support write_timeout
        # Use longer timeout for network connections (can have more latency)
        ser = serial.serial_for_url(args.port, baudrate=args.baud, timeout=0.2)
        tune_serial_link(ser, network_link=True)
    else:
        # Regular COM port or /dev/ttyX
        ser = serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=1)
        tune_serial_link(ser, network_link=False)
    
    with ser:
        if not args.no_hw_reset:
            # Perform hardware reset at startup
            try:
                auto_reset_and_wait_ready(ser,15.0)
            except Exception as e:
                logger.warning(f"⚠️  Reset failed: {e}")
                logger.info("🔄 Falling back to waiting for existing READY beacon...")
                # Close and reopen to ensure clean state
                ser.close()
                time.sleep(1.0)
                ser.open()
                ready_ok, _, _ = wait_ready(ser, 10.0)
                if not ready_ok:
                    raise TimeoutError("Failed to receive READY after reset fallback")
        else:
            # No hardware reset (no DTR/RTS): keep USB-UART link up and reset runtime state only.
            timeout = 60.0 if (args.wokwi or args.desktop) else 15.0
            logger.debug(f"Debug: args.wokwi={args.wokwi}, args.desktop={args.desktop}, timeout={timeout}")
            logger.info(f"⏩ No hardware reset: sending OPC_RESET_STATE, then waiting for READY (timeout: {timeout}s)...")
            for attempt in range(2):
                try:
                    ser.reset_input_buffer()
                    if attempt == 0:
                        ser.reset_output_buffer()
                    time.sleep(0.05 if attempt == 0 else 0.5)
                    send_cmd(ser, 1, OPC_RESET_STATE)
                    ser.flush()
                    # Do not flush RX after reset command: READY may arrive almost immediately.
                    time.sleep(0.05)
                    break
                except (OSError, serial.SerialException) as e:
                    errno = getattr(e, "errno", None)
                    if errno == 6 or "Device not configured" in str(e) or "not configured" in str(e).lower():
                        if attempt == 0:
                            logger.debug("Software reset write failed (adapter/driver), reopening port and retrying...")
                            try:
                                if ser.is_open:
                                    ser.close()
                                time.sleep(0.3)
                                ser.open()
                                tune_serial_link(ser, network_link=False)
                            except Exception as reopen_err:
                                logger.debug(f"Reopen after write failure did not fully recover: {reopen_err}")
                        else:
                            logger.warning("⚠️  Software reset send failed (e.g. Device not configured); waiting for READY anyway.")
                    else:
                        logger.warning(f"⚠️  Software reset send failed: {e}")
                        break
                except Exception as e:
                    logger.warning(f"⚠️  Software reset send failed: {e}")
                    break
            ready_ok, dev_trained, dev_tested = wait_ready(ser, timeout)
            if not ready_ok:
                logger.error("Failed to receive READY - device may not be running or wrong baud/port")
                raise TimeoutError("READY not received (--no-hw-reset: software reset was sent)")

        # Optional runtime algorithm selection (single flashed firmware can host all models)
        if args.model:
            try:
                select_runtime_model(ser, args.model, timeout_s=5.0)
            except Exception as e:
                logger.error(f"Failed to select runtime model '{args.model}': {e}")
                return

        # Send feature/config flags once (before runs)
        try:
            seq = 1
            flags, payload = build_config_payload(
                input_real=args.input_real,
                preproc_id=effective_preproc_id,
                preproc_param0=effective_preproc_param0,
                tm_runtime=tm_runtime_cfg,
            )
            seq = send_cmd(ser, seq, OPC_CONFIG, payload)
            config_details = []
            if args.input_real:
                config_details.append("input=raw-float")
            config_details.append(f"preproc={effective_preproc_name}")
            if effective_preproc_id == PREPROC_TWINE:
                config_details.append(f"twine_bits={effective_preproc_param0}")
            if tm_runtime_cfg is not None:
                config_details.append(
                    "tm="
                    f"C{tm_runtime_cfg['clauses']}/"
                    f"T{tm_runtime_cfg['threshold']}/"
                    f"s{tm_runtime_cfg['specificity']}/"
                    f"seed{tm_runtime_cfg['seed']}/"
                    f"dens{tm_runtime_cfg['init_density_pct']}"
                )
            if not config_details:
                config_details.append("defaults")
            logger.info(f"🧩 Config sent: {', '.join(config_details)} (flags=0x{flags:02X})")
        except Exception as e:
            logger.warning(f"⚠️  Failed to send config: {e}")

        # Multi-run experimental framework for statistical analysis
        all_runs_results = []  # Store results from each run

        # Joulescope power/energy measurement (one device for all runs)
        js_sampler = None
        if getattr(args, "js_power", False):
            js_sampler = JoulescopeSampler(
                poll_ms=getattr(args, "js_poll_ms", 100.0),
                serial_number=getattr(args, "js_serial", None),
                suppress_usb_warnings=not getattr(args, "js_keep_usb_warnings", False),
            )
            if not js_sampler.open_device():
                js_sampler = None

        # Create master results dictionary for JSON export
        if using_csv and csv_meta is not None:
            data_files_payload = {'csv': csv_meta}
        elif using_memory_dataset and text_meta is not None:
            data_files_payload = text_meta
        else:
            data_files_payload = {
                'train_files': [str(p) for p in train_paths],
                'test_files': [str(p) for p in test_paths],
            }

        master_results = {
            'metadata': {
                'timestamp': timestamp,
                'port': args.port,
                'baud': args.baud,
                'mode': 'wokwi' if args.wokwi else ('desktop' if args.desktop else 'hardware'),
                'dataset': args.dataset,
                'data_dir': str((Path(args.data_dir) / args.dataset) if args.dataset else Path(args.data_dir)),
                'log_file': str(log_file.name),
            },
            'configuration': {
                'model_tag': model_tag if model_tag else None,
                'model': args.model,
                'window': args.window,
                'batch_records': args.batch_records,
                'throttle_ms': args.throttle_ms,
                'epochs': args.epochs,
                'runs': args.runs,
                'shuffle_seed': args.shuffle_seed,
                'no_run_shuffle': args.no_run_shuffle,
                'train_samples': args.train_samples,
                'test_samples': args.test_samples,
                'eval_samples': args.eval_samples,
                'status_every': args.status_every,
                'no_hw_reset': args.no_hw_reset,
                'final_logs': args.final_logs,
                'csv_bitpack': args.csv_bitpack,
                'unsafe_no_data_checksum': args.unsafe_no_data_checksum,
                'online_preproc': effective_preproc_name,
                'twine_bits': effective_preproc_param0 if effective_preproc_id == PREPROC_TWINE else None,
                'host_standardize': host_standardize_effective,
                'tm_runtime': tm_runtime_cfg,
                'js_power': getattr(args, 'js_power', False),
                'js_poll_ms': getattr(args, 'js_poll_ms', 100.0),
            },
            'data_files': data_files_payload,
            'runs': [],
            'statistics': {},
            'comprehensive_analysis': None,
        }
        
        Xtr_base = Xtr
        Ytr_base = Ytr
        Xte_base = Xte
        Yte_base = Yte
        if using_csv or using_memory_dataset:
            # Preserve original split order; per-run permutations derive from this stable base.
            Xtr_base = Xtr.copy()
            Ytr_base = Ytr.copy()
            Xte_base = Xte.copy()
            Yte_base = Yte.copy()

        for run in range(1, args.runs+1):
            logger.debug(f"[DEBUG] === STARTING RUN {run}/{args.runs} ===")
            if args.runs > 1:
                logger.info("="*70)
                logger.info(f"🔬 EXPERIMENTAL RUN {run}/{args.runs}")
                logger.info("="*70)
                
                # Reset device for each run (except first run if continuing from initial state)
                if run > 1:
                    logger.info("🔄 Resetting device for new experimental run...")
                    seq = 1
                    seq = send_cmd(ser, seq, OPC_RESET_STATE)
                    time.sleep(0.5)
                    
                    ready_timeout = 10.0 if (args.wokwi or args.desktop) else 5.0
                    ready_ok, dev_trained, dev_tested = wait_ready(ser, ready_timeout)
                    if not ready_ok:
                        logger.error(f"❌ Failed to receive READY for run {run}")
                        continue
                    
                    # Verify reset was successful (counters should be 0)
                    if dev_trained == 0 and dev_tested == 0:
                        logger.success(f"✅ Device reset complete (counters zeroed)")
                    else:
                        logger.warning(f"⚠️  Device reset received but counters not zeroed: trained={dev_trained}, tested={dev_tested}")
                        logger.warning(f"⚠️  Continuing anyway - results may be affected")
            
            best_acc=-1.0; best_epoch=0
            total_sent=0; total_bytes=0; total_time=0.0
            epoch_test_acc=[]
            epoch_train_acc=[]
            epoch_failed = False
            epochs_data = []  # Store detailed epoch data
            run_start_time = time.time()
            
            logger.debug(f"[DEBUG] Run {run}: Initialized run variables, starting epochs loop")

            # Apply reproducible per-run shuffling from a fixed base split.
            if using_csv or using_memory_dataset:
                if args.no_run_shuffle:
                    Xtr = Xtr_base
                    Ytr = Ytr_base
                    Xte = Xte_base
                    Yte = Yte_base
                    logger.debug(f"[DEBUG] Run {run}: Using fixed data order (--no-run-shuffle)")
                else:
                    run_seed = int(args.shuffle_seed) + run
                    tr_rng = np.random.default_rng(run_seed)
                    te_rng = np.random.default_rng(run_seed ^ 0x9E3779B9)
                    indices_tr = tr_rng.permutation(Xtr_base.shape[0])
                    indices_te = te_rng.permutation(Xte_base.shape[0])
                    Xtr = Xtr_base[indices_tr]
                    Ytr = Ytr_base[indices_tr]
                    Xte = Xte_base[indices_te]
                    Yte = Yte_base[indices_te]
                    logger.debug(f"[DEBUG] Run {run}: Data shuffled with seed={run_seed}")

            # Per-run finalization (mem/prof summary, accuracy, store results)
            def _finalize_current_run():
                memprof_data = None
                seq = 1
                seq, mp = request_memprof(ser, seq)
                if mp:
                    memprof_data = mp
                    logger.info("🧮 MEMORY & PROFILING:")
                    logger.info(f"  Samples:  trained={mp['trained']}, tested={mp['tested']}, acc={mp['acc']:.4f}")
                    logger.info(f"  Heap:     free={mp['free_heap']}, min={mp['min_heap']}, maxAlloc={mp['max_alloc']}")
                    logger.info(f"  Utils:    cur={mp['util_current']}, peak={mp['util_peak']}, allocs={mp['util_alloc_count']}, frees={mp['util_free_count']}, alive={mp['util_active']}")
                    logger.info(f"  Profiler: entries={mp['entries_total']}, sent={mp['entries_sent']}, truncated={mp['truncated']}")
                    if args.dump_prof and mp['entries']:
                        logger.info("┌─ Per-function (name, total_us, count, avg_us, max_us, last_us)")
                        for name,tot,cnt,avg,mx,last in mp['entries']:
                            logger.info(f"│ {name:32s} {tot:10d}  {cnt:6d}  {avg:8.1f}  {mx:8d}  {last:8d}")
                        if mp['truncated']:
                            logger.info("│ … truncated …")
                        logger.info("└─")

                logger.debug(f"[DEBUG] Run {run}: After epoch loop, about to show summary")

                if total_time>0:
                    logger.info(f"Total: {total_sent} smp in {total_time:.1f}s (rate={total_sent/total_time:.1f} smp/s)")

                if args.epochs > 1:
                    logger.info("="*70)
                    if epoch_failed:
                        logger.warning("⚠️  MULTI-EPOCH RUN COMPLETED WITH ERRORS")
                    else:
                        logger.success(f"✅ ALL {args.epochs} EPOCHS COMPLETED SUCCESSFULLY")
                    logger.info("="*70)

                logger.debug(f"[DEBUG] Run {run}: About to check accuracy display condition")
                logger.debug(f"[DEBUG] Run {run}: epoch_train_acc={epoch_train_acc}, epoch_test_acc={epoch_test_acc}")
                if any(a is not None for a in epoch_train_acc) or any(a is not None for a in epoch_test_acc):
                    logger.info("Per-epoch accuracy:")
                    for i in range(1, args.epochs+1):
                        ta = epoch_train_acc[i-1] if i-1 < len(epoch_train_acc) else None
                        va = epoch_test_acc[i-1] if i-1 < len(epoch_test_acc) else None
                        ta_str = (f"{ta:.4f}" if ta is not None else "n/a")
                        va_str = (f"{va:.4f}" if va is not None else "n/a")
                        status = "✅" if (ta is not None or va is not None) else "❌"
                        logger.info(f"  {status} Epoch {i}: train={ta_str}  test={va_str}")
                    if best_epoch>0:
                        logger.info(f"🏆 Best: epoch {best_epoch} acc={best_acc:.4f}")

                logger.debug(f"[DEBUG] Run {run}: About to store results to all_runs_results (current len={len(all_runs_results)})")
                run_duration = time.time() - run_start_time
                avg_throughput = total_sent/total_time if total_time > 0 else 0
                avg_latency_ms = (1000.0 / avg_throughput) if avg_throughput > 0 else 0
                train_samples_sum = sum(int(ep.get('train', {}).get('samples_sent', 0) or 0) for ep in epochs_data)
                train_time_sum = sum(float(ep.get('train', {}).get('duration_sec', 0.0) or 0.0) for ep in epochs_data)
                test_samples_sum = sum(int(ep.get('test', {}).get('samples_sent', 0) or 0) for ep in epochs_data)
                test_time_sum = sum(float(ep.get('test', {}).get('duration_sec', 0.0) or 0.0) for ep in epochs_data)
                train_throughput_sps = (train_samples_sum / train_time_sum) if train_time_sum > 0 else 0.0
                test_throughput_sps = (test_samples_sum / test_time_sum) if test_time_sum > 0 else 0.0
                train_latency_ms = (1000.0 / train_throughput_sps) if train_throughput_sps > 0 else 0.0
                test_latency_ms = (1000.0 / test_throughput_sps) if test_throughput_sps > 0 else 0.0
                train_energy_j = sum(float(ep.get('train', {}).get('energy_j') or 0) for ep in epochs_data)
                test_energy_j = sum(float(ep.get('test', {}).get('energy_j') or 0) for ep in epochs_data)
                train_energy_uj = ((train_energy_j * 1e6) / float(train_samples_sum)) if train_energy_j > 0 and train_samples_sum > 0 else None
                test_energy_uj = ((test_energy_j * 1e6) / float(test_samples_sum)) if test_energy_j > 0 and test_samples_sum > 0 else None
                train_powers = [ep.get('train', {}).get('power_mean_w') for ep in epochs_data if ep.get('train', {}).get('power_mean_w') is not None]
                test_powers = [ep.get('test', {}).get('power_mean_w') for ep in epochs_data if ep.get('test', {}).get('power_mean_w') is not None]
                train_currents = [ep.get('train', {}).get('current_mean_a') for ep in epochs_data if ep.get('train', {}).get('current_mean_a') is not None]
                test_currents = [ep.get('test', {}).get('current_mean_a') for ep in epochs_data if ep.get('test', {}).get('current_mean_a') is not None]
                train_voltages = [ep.get('train', {}).get('voltage_mean_v') for ep in epochs_data if ep.get('train', {}).get('voltage_mean_v') is not None]
                test_voltages = [ep.get('test', {}).get('voltage_mean_v') for ep in epochs_data if ep.get('test', {}).get('voltage_mean_v') is not None]
                run_results = {
                    'run_number': run,
                    'summary': {
                        'best_accuracy': best_acc,
                        'best_epoch': best_epoch,
                        'final_train_accuracy': epoch_train_acc[-1] if epoch_train_acc and epoch_train_acc[-1] is not None else None,
                        'final_test_accuracy': epoch_test_acc[-1] if epoch_test_acc and epoch_test_acc[-1] is not None else None,
                        'total_duration_sec': run_duration,
                        'total_samples_sent': total_sent,
                        'total_bytes_sent': total_bytes,
                        'avg_throughput_sps': avg_throughput,
                        'avg_latency_ms': avg_latency_ms,
                        'train_throughput_sps': train_throughput_sps,
                        'test_throughput_sps': test_throughput_sps,
                        'train_latency_ms': train_latency_ms,
                        'test_latency_ms': test_latency_ms,
                        'failed': epoch_failed,
                        'train_energy_j': train_energy_j if train_energy_j > 0 else None,
                        'test_energy_j': test_energy_j if test_energy_j > 0 else None,
                        'train_energy_uj_per_sample': train_energy_uj,
                        'test_energy_uj_per_sample': test_energy_uj,
                        'train_power_mean_w': float(statistics.mean(train_powers)) if train_powers else None,
                        'test_power_mean_w': float(statistics.mean(test_powers)) if test_powers else None,
                        'train_current_mean_a': float(statistics.mean(train_currents)) if train_currents else None,
                        'test_current_mean_a': float(statistics.mean(test_currents)) if test_currents else None,
                        'train_voltage_mean_v': float(statistics.mean(train_voltages)) if train_voltages else None,
                        'test_voltage_mean_v': float(statistics.mean(test_voltages)) if test_voltages else None,
                    },
                    'epochs': epochs_data,
                    'memory_profile': memprof_data,
                }
                all_runs_results.append(run_results)
                master_results['runs'].append(run_results)
                logger.debug(f"[DEBUG] Run {run}: Results stored! all_runs_results now has {len(all_runs_results)} entries")

                if args.runs > 1:
                    logger.info(f"✅ Run {run}/{args.runs} complete - Best Acc: {best_acc:.4f}")

                logger.debug(f"[DEBUG] Run {run}: End of run loop iteration")

            for epoch in range(1, args.epochs+1):
                logger.debug(f"[DEBUG] Run {run}, Epoch {epoch}: Starting epoch")
                epoch_data = {
                    'epoch': epoch,
                    'train': {},
                    'test': {},
                    'snapshot': None,
                    'error': None,
                }
                
                try:
                    logger.info(f"Epoch {epoch}/{args.epochs} TRAIN")
                    if js_sampler is not None and js_sampler._device is not None:
                        js_sampler.start_phase()
                    t0=time.time()
                    use_raw_samples = args.input_real and (using_csv or using_memory_dataset)
                    if use_raw_samples:
                        tr_sent,tr_bytes,_ = stream_phase_raw(
                            ser, Xtr, Ytr, "train", args.window, args.train_samples, args.throttle_ms,
                            args.status_every, quiet=True, csv_bitpack=False,
                            use_data_checksum=not args.unsafe_no_data_checksum, batch_records=args.batch_records,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    elif using_csv:
                        tr_sent,tr_bytes,_ = stream_phase_raw(
                            ser, Xtr, Ytr, "train", args.window, args.train_samples, args.throttle_ms,
                            args.status_every, quiet=True, csv_bitpack=True,
                            use_data_checksum=not args.unsafe_no_data_checksum, batch_records=args.batch_records,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    else:
                        tr_sent,tr_bytes,_=stream_phase(
                            ser, train_paths, "train", args.window, args.train_samples, args.throttle_ms,
                            args.status_every, quiet=True, multiclass=False,
                            use_data_checksum=not args.unsafe_no_data_checksum,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    t1=time.time()
                    train_time = t1-t0
                    total_sent += tr_sent; total_bytes += tr_bytes; total_time += train_time
                    sps = tr_sent/train_time if train_time>0 else 0.0

                    # Query training metrics snapshot for training accuracy
                    seq, snap = request_snapshot(ser, 1)
                    if snap and snap['trained']>0:
                        tr_acc = snap['train_correct']/float(snap['trained'])
                        epoch_train_acc.append(tr_acc)
                        logger.info(f"  train: {tr_sent} smp in {train_time:.1f}s ({sps:.1f} smp/s) | acc={tr_acc:.4f} ema={snap['ema_train_acc']:.3f}")
                        
                        # Store training data
                        epoch_data['train'] = {
                            'samples_sent': tr_sent,
                            'bytes_sent': tr_bytes,
                            'duration_sec': train_time,
                            'throughput_sps': sps,
                            'accuracy': tr_acc,
                            'ema_accuracy': snap['ema_train_acc'],
                            'ema_score': snap['ema_train_score'],
                        }
                    else:
                        epoch_train_acc.append(None)
                        logger.info(f"  train: {tr_sent} smp in {train_time:.1f}s ({sps:.1f} smp/s)")
                        
                        epoch_data['train'] = {
                            'samples_sent': tr_sent,
                            'bytes_sent': tr_bytes,
                            'duration_sec': train_time,
                            'throughput_sps': sps,
                            'accuracy': None,
                        }
                    if js_sampler is not None:
                        train_js = js_sampler.stop_phase()
                        epoch_data['train']['power_mean_w'] = train_js.get('power_mean_w')
                        epoch_data['train']['current_mean_a'] = train_js.get('current_mean_a')
                        epoch_data['train']['voltage_mean_v'] = train_js.get('voltage_mean_v')
                        epoch_data['train']['energy_j'] = train_js.get('energy_j')
                        if train_js.get('energy_j') is not None:
                            logger.info(f"  train (Joulescope): {train_js['energy_j']:.4f} J, {train_js.get('power_mean_w') or 0:.4f} W mean")
                except Exception as e:
                    if js_sampler is not None:
                        js_sampler.stop_phase()
                    logger.error(f"❌ Error during epoch {epoch} training: {e}")
                    epoch_train_acc.append(None)
                    epoch_failed = True
                    epoch_data['error'] = f"Training error: {str(e)}"
                    if epoch < args.epochs:
                        logger.warning(f"⚠️  Continuing to next epoch...")
                        epochs_data.append(epoch_data)
                        continue
                    else:
                        epochs_data.append(epoch_data)
                        break

                # NOTE: Memprof moved to AFTER testing to avoid timing artifacts

                if args.snapshot:
                    seq, snap = request_snapshot(ser, 1)
                    if snap:
                        logger.info(f"📸 Snapshot: trained={snap['trained']} train_correct={snap['train_correct']} "
                              f"ema_acc={snap['ema_train_acc']:.3f} ema_score={snap['ema_train_score']:.3f} "
                              f"last_score={snap['last_score']} last_pred={snap['last_pred']}")
                        epoch_data['snapshot'] = snap

                try:
                    transition_start = time.time()
                    logger.debug(f"⏱️ Starting TRAIN→TEST transition...")
                    logger.info(f"Epoch {epoch}/{args.epochs} TEST")
                    if js_sampler is not None and js_sampler._device is not None:
                        js_sampler.start_phase()
                    u0=time.time()
                    transition_time = (u0 - transition_start) * 1000
                    logger.debug(f"⏱️ Transition overhead: {transition_time:.0f}ms")
                    eval_cap = args.eval_samples if args.eval_samples is not None else args.test_samples
                    if use_raw_samples:
                        te_sent,te_bytes,te_stat = stream_phase_raw(
                            ser, Xte, Yte, "test", args.window, eval_cap, args.throttle_ms, 0.0,
                            quiet=True, csv_bitpack=False,
                            use_data_checksum=not args.unsafe_no_data_checksum, batch_records=args.batch_records,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    elif using_csv:
                        te_sent,te_bytes,te_stat = stream_phase_raw(
                            ser, Xte, Yte, "test", args.window, eval_cap, args.throttle_ms, 0.0,
                            quiet=True, csv_bitpack=True,
                            use_data_checksum=not args.unsafe_no_data_checksum, batch_records=args.batch_records,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    else:
                        te_sent,te_bytes,te_stat=stream_phase(
                            ser, test_paths, "test", args.window, eval_cap, args.throttle_ms, 0.0,
                            quiet=True, multiclass=False, use_data_checksum=not args.unsafe_no_data_checksum,
                            preproc_id=effective_preproc_id, preproc_param0=effective_preproc_param0,
                            tm_runtime_cfg=tm_runtime_cfg)
                    u1=time.time()
                    test_time = u1-u0
                    total_sent += te_sent; total_bytes += te_bytes; total_time += test_time
                    if te_stat:
                        _,tested,acc=te_stat
                        epoch_test_acc.append(acc)
                        if acc>best_acc: best_acc=acc; best_epoch=epoch
                        logger.info(f"  test:  {te_sent} smp in {test_time:.1f}s | tested={tested} acc={acc:.4f}")
                        
                        epoch_data['test'] = {
                            'samples_sent': te_sent,
                            'bytes_sent': te_bytes,
                            'duration_sec': test_time,
                            'tested_count': tested,
                            'accuracy': acc,
                        }
                    else:
                        epoch_test_acc.append(None)
                        logger.info(f"  test:  {te_sent} smp in {test_time:.1f}s")
                        
                        epoch_data['test'] = {
                            'samples_sent': te_sent,
                            'bytes_sent': te_bytes,
                            'duration_sec': test_time,
                            'tested_count': None,
                            'accuracy': None,
                        }
                    if js_sampler is not None:
                        test_js = js_sampler.stop_phase()
                        epoch_data['test']['power_mean_w'] = test_js.get('power_mean_w')
                        epoch_data['test']['current_mean_a'] = test_js.get('current_mean_a')
                        epoch_data['test']['voltage_mean_v'] = test_js.get('voltage_mean_v')
                        epoch_data['test']['energy_j'] = test_js.get('energy_j')
                        if test_js.get('energy_j') is not None:
                            logger.info(f"  test (Joulescope): {test_js['energy_j']:.4f} J, {test_js.get('power_mean_w') or 0:.4f} W mean")
                except Exception as e:
                    if js_sampler is not None:
                        js_sampler.stop_phase()
                    logger.error(f"❌ Error during epoch {epoch} testing: {e}")
                    epoch_test_acc.append(None)
                    epoch_failed = True
                    if not epoch_data['error']:
                        epoch_data['error'] = f"Testing error: {str(e)}"
                    else:
                        epoch_data['error'] += f"; Testing error: {str(e)}"
                    if epoch < args.epochs:
                        logger.warning(f"⚠️  Continuing to next epoch...")
                        epochs_data.append(epoch_data)
                        continue
                    else:
                        epochs_data.append(epoch_data)
                        break
                
                epochs_data.append(epoch_data)
                
                # Save intermediate results after each epoch if requested
                if args.save_epoch_results:
                    epoch_json_file = results_dir / f"epoch_{epoch}_{timestamp}.json"
                    epoch_txt_file = results_dir / f"epoch_{epoch}_{timestamp}.txt"
                    try:
                        # Save JSON
                        with open(epoch_json_file, 'w', encoding='utf-8') as f:
                            json.dump(epoch_data, f, indent=2)
                        
                        # Save TXT using ResultsFormatter
                        with open(epoch_txt_file, 'w', encoding='utf-8') as f:
                            f.write(ResultsFormatter.format_epoch_report(epoch_data, epoch, args.epochs))
                        
                        logger.info(f"💾 Epoch {epoch} results saved: {epoch_json_file.name}, {epoch_txt_file.name}")
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to save epoch {epoch} results: {e}")
                
                # If this was the last epoch of the run, finalize run results now
                if epoch == args.epochs:
                    _finalize_current_run()
        
        # === END OF EPOCH LOOP - Back to run loop level ===
        # Ensure per-run finalization always executes for each run
        def _finalize_current_run():
            # Request memprof AFTER testing to avoid timing artifacts
            memprof_data = None
            seq = 1
            seq, mp = request_memprof(ser, seq)
            if mp:
                memprof_data = mp
                logger.info("🧮 MEMORY & PROFILING:")
                logger.info(f"  Samples:  trained={mp['trained']}, tested={mp['tested']}, acc={mp['acc']:.4f}")
                logger.info(f"  Heap:     free={mp['free_heap']}, min={mp['min_heap']}, maxAlloc={mp['max_alloc']}")
                logger.info(f"  Utils:    cur={mp['util_current']}, peak={mp['util_peak']}, allocs={mp['util_alloc_count']}, frees={mp['util_free_count']}, alive={mp['util_active']}")
                logger.info(f"  Profiler: entries={mp['entries_total']}, sent={mp['entries_sent']}, truncated={mp['truncated']}")
                if args.dump_prof and mp['entries']:
                    logger.info("┌─ Per-function (name, total_us, count, avg_us, max_us, last_us)")
                    for name,tot,cnt,avg,mx,last in mp['entries']:
                        logger.info(f"│ {name:32s} {tot:10d}  {cnt:6d}  {avg:8.1f}  {mx:8d}  {last:8d}")
                    if mp['truncated']:
                        logger.info("│ … truncated …")
                    logger.info("└─")
            
            logger.debug(f"[DEBUG] Run {run}: After epoch loop, about to show summary")
            
            if total_time>0:
                logger.info(f"Total: {total_sent} smp in {total_time:.1f}s (rate={total_sent/total_time:.1f} smp/s)")
            
            # Summary with robust epoch status
            if args.epochs > 1:
                logger.info("="*70)
                if epoch_failed:
                    logger.warning("⚠️  MULTI-EPOCH RUN COMPLETED WITH ERRORS")
                else:
                    logger.success(f"✅ ALL {args.epochs} EPOCHS COMPLETED SUCCESSFULLY")
                logger.info("="*70)
            
            # Display per-epoch accuracy
            logger.debug(f"[DEBUG] Run {run}: About to check accuracy display condition")
            logger.debug(f"[DEBUG] Run {run}: epoch_train_acc={epoch_train_acc}, epoch_test_acc={epoch_test_acc}")
            if any(a is not None for a in epoch_train_acc) or any(a is not None for a in epoch_test_acc):
                logger.info("Per-epoch accuracy:")
                for i in range(1, args.epochs+1):
                    ta = epoch_train_acc[i-1] if i-1 < len(epoch_train_acc) else None
                    va = epoch_test_acc[i-1] if i-1 < len(epoch_test_acc) else None
                    ta_str = (f"{ta:.4f}" if ta is not None else "n/a")
                    va_str = (f"{va:.4f}" if va is not None else "n/a")
                    status = "✅" if (ta is not None or va is not None) else "❌"
                    logger.info(f"  {status} Epoch {i}: train={ta_str}  test={va_str}")
                if best_epoch>0:
                    logger.info(f"🏆 Best: epoch {best_epoch} acc={best_acc:.4f}")
            
            # Store comprehensive results from this run (ALWAYS, even if failed)
            logger.debug(f"[DEBUG] Run {run}: About to store results to all_runs_results (current len={len(all_runs_results)})")
            run_duration = time.time() - run_start_time
            run_results = {
                'run_number': run,
                'summary': {
                    'best_accuracy': best_acc,
                    'best_epoch': best_epoch,
                    'final_train_accuracy': epoch_train_acc[-1] if epoch_train_acc and epoch_train_acc[-1] is not None else None,
                    'final_test_accuracy': epoch_test_acc[-1] if epoch_test_acc and epoch_test_acc[-1] is not None else None,
                    'total_duration_sec': run_duration,
                    'total_samples_sent': total_sent,
                    'total_bytes_sent': total_bytes,
                    'avg_throughput_sps': total_sent/total_time if total_time > 0 else 0,
                    'failed': epoch_failed,
                },
                'epochs': epochs_data,
                'memory_profile': memprof_data,
            }
            all_runs_results.append(run_results)
            master_results['runs'].append(run_results)
            logger.debug(f"[DEBUG] Run {run}: Results stored! all_runs_results now has {len(all_runs_results)} entries")
            
            if args.runs > 1:
                logger.info(f"✅ Run {run}/{args.runs} complete - Best Acc: {best_acc:.4f}")
            
            logger.debug(f"[DEBUG] Run {run}: End of run loop iteration")

        
        
        # === END OF RUNS LOOP ===
        if js_sampler is not None:
            js_sampler.close_device()

        # Compute and display statistics across all runs
        if args.runs > 1:
            logger.info("="*70)
            logger.info("📊 STATISTICAL ANALYSIS ACROSS ALL RUNS")
            logger.info("="*70)
            
            # Extract metrics
            successful_runs = [r for r in all_runs_results if not r['summary']['failed']]
            best_accs = [r['summary']['best_accuracy'] for r in successful_runs if r['summary']['best_accuracy'] >= 0]
            final_test_accs = [r['summary']['final_test_accuracy'] for r in successful_runs if r['summary']['final_test_accuracy'] is not None]
            final_train_accs = [r['summary']['final_train_accuracy'] for r in successful_runs if r['summary']['final_train_accuracy'] is not None]
            throughputs = [r['summary']['avg_throughput_sps'] for r in successful_runs if r['summary']['avg_throughput_sps'] > 0]
            # Calculate latencies from throughputs (or use stored values if available)
            latencies = []
            for r in successful_runs:
                if r['summary'].get('avg_latency_ms', 0) > 0:
                    latencies.append(r['summary']['avg_latency_ms'])
                elif r['summary']['avg_throughput_sps'] > 0:
                    latencies.append(1000.0 / r['summary']['avg_throughput_sps'])
            
            # Extract precision, recall, F1 from comprehensive analysis (if available)
            precisions = []
            recalls = []
            f1_scores = []
            for r in successful_runs:
                if master_results.get('comprehensive_analysis') and \
                   master_results['comprehensive_analysis'].get('performance_benchmarks'):
                    pb = master_results['comprehensive_analysis']['performance_benchmarks']
                    if pb.get('precision') is not None:
                        precisions.append(pb['precision'])
                    if pb.get('recall') is not None:
                        recalls.append(pb['recall'])
                    if pb.get('f1_score') is not None:
                        f1_scores.append(pb['f1_score'])
            
            if best_accs:
                logger.info("Test Accuracy (Best per run):")
                logger.info(f"  Mean:   {statistics.mean(best_accs):.4f}")
                logger.info(f"  Std:    {statistics.stdev(best_accs) if len(best_accs) > 1 else 0.0:.4f}")
                logger.info(f"  Min:    {min(best_accs):.4f}")
                logger.info(f"  Max:    {max(best_accs):.4f}")
                logger.info(f"  Median: {statistics.median(best_accs):.4f}")
                
                # 95% confidence interval (approximate)
                if len(best_accs) > 1:
                    mean = statistics.mean(best_accs)
                    std = statistics.stdev(best_accs)
                    n = len(best_accs)
                    ci_95 = 1.96 * std / (n ** 0.5)
                    logger.info(f"  95% CI: {mean:.4f} ± {ci_95:.4f}")
            
            if final_test_accs:
                logger.info("\nTest Accuracy (Final epoch per run):")
                logger.info(f"  Mean:   {statistics.mean(final_test_accs):.4f}")
                logger.info(f"  Std:    {statistics.stdev(final_test_accs) if len(final_test_accs) > 1 else 0.0:.4f}")
            
            if final_train_accs:
                logger.info("\nTrain Accuracy (Final epoch per run):")
                logger.info(f"  Mean:   {statistics.mean(final_train_accs):.4f}")
                logger.info(f"  Std:    {statistics.stdev(final_train_accs) if len(final_train_accs) > 1 else 0.0:.4f}")
            
            if throughputs:
                logger.info("\nThroughput (samples/sec):")
                logger.info(f"  Mean:   {statistics.mean(throughputs):.1f}")
                logger.info(f"  Std:    {statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0:.1f}")
                logger.info(f"  Min:    {min(throughputs):.1f}")
                logger.info(f"  Max:    {max(throughputs):.1f}")
                logger.info(f"  Median: {statistics.median(throughputs):.1f}")
            
            if latencies:
                logger.info("\nLatency (ms per sample):")
                logger.info(f"  Mean:   {statistics.mean(latencies):.2f}")
                logger.info(f"  Std:    {statistics.stdev(latencies) if len(latencies) > 1 else 0.0:.2f}")
                logger.info(f"  Min:    {min(latencies):.2f}")
                logger.info(f"  Max:    {max(latencies):.2f}")
                logger.info(f"  Median: {statistics.median(latencies):.2f}")
            
            if precisions:
                logger.info("\nPrecision:")
                logger.info(f"  Mean:   {statistics.mean(precisions):.4f}")
                logger.info(f"  Std:    {statistics.stdev(precisions) if len(precisions) > 1 else 0.0:.4f}")
                logger.info(f"  Min:    {min(precisions):.4f}")
                logger.info(f"  Max:    {max(precisions):.4f}")
                logger.info(f"  Median: {statistics.median(precisions):.4f}")
            
            if recalls:
                logger.info("\nRecall:")
                logger.info(f"  Mean:   {statistics.mean(recalls):.4f}")
                logger.info(f"  Std:    {statistics.stdev(recalls) if len(recalls) > 1 else 0.0:.4f}")
                logger.info(f"  Min:    {min(recalls):.4f}")
                logger.info(f"  Max:    {max(recalls):.4f}")
                logger.info(f"  Median: {statistics.median(recalls):.4f}")
            
            if f1_scores:
                logger.info("\nF1 Score:")
                logger.info(f"  Mean:   {statistics.mean(f1_scores):.4f}")
                logger.info(f"  Std:    {statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0:.4f}")
                logger.info(f"  Min:    {min(f1_scores):.4f}")
                logger.info(f"  Max:    {max(f1_scores):.4f}")
                logger.info(f"  Median: {statistics.median(f1_scores):.4f}")
            
            logger.info(f"\nRuns completed: {len(successful_runs)}/{args.runs}")
            logger.info("="*70)
            
            # Build statistics dictionary
            stats_dict = {
                'total_runs': args.runs,
                'successful_runs': len(successful_runs),
                'failed_runs': args.runs - len(successful_runs),
                'test_accuracy_best': {
                    'values': best_accs,
                    'mean': statistics.mean(best_accs) if best_accs else None,
                    'std': statistics.stdev(best_accs) if len(best_accs) > 1 else 0.0,
                    'min': min(best_accs) if best_accs else None,
                    'max': max(best_accs) if best_accs else None,
                    'median': statistics.median(best_accs) if best_accs else None,
                    'ci_95': None,
                },
                'test_accuracy_final': {
                    'values': final_test_accs,
                    'mean': statistics.mean(final_test_accs) if final_test_accs else None,
                    'std': statistics.stdev(final_test_accs) if len(final_test_accs) > 1 else 0.0,
                    'min': min(final_test_accs) if final_test_accs else None,
                    'max': max(final_test_accs) if final_test_accs else None,
                    'median': statistics.median(final_test_accs) if len(final_test_accs) > 1 else None,
                },
                'train_accuracy_final': {
                    'values': final_train_accs,
                    'mean': statistics.mean(final_train_accs) if final_train_accs else None,
                    'std': statistics.stdev(final_train_accs) if len(final_train_accs) > 1 else 0.0,
                    'min': min(final_train_accs) if final_train_accs else None,
                    'max': max(final_train_accs) if final_train_accs else None,
                    'median': statistics.median(final_train_accs) if len(final_train_accs) > 1 else None,
                },
                'throughput': {
                    'values': throughputs,
                    'mean': statistics.mean(throughputs) if throughputs else None,
                    'std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
                    'min': min(throughputs) if throughputs else None,
                    'max': max(throughputs) if throughputs else None,
                    'median': statistics.median(throughputs) if len(throughputs) > 1 else None,
                },
                'latency': {
                    'values': latencies,
                    'mean': statistics.mean(latencies) if latencies else None,
                    'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                    'min': min(latencies) if latencies else None,
                    'max': max(latencies) if latencies else None,
                    'median': statistics.median(latencies) if len(latencies) > 1 else None,
                },
                'precision': {
                    'values': precisions,
                    'mean': statistics.mean(precisions) if precisions else None,
                    'std': statistics.stdev(precisions) if len(precisions) > 1 else 0.0,
                    'min': min(precisions) if precisions else None,
                    'max': max(precisions) if precisions else None,
                    'median': statistics.median(precisions) if precisions else None,
                },
                'recall': {
                    'values': recalls,
                    'mean': statistics.mean(recalls) if recalls else None,
                    'std': statistics.stdev(recalls) if len(recalls) > 1 else 0.0,
                    'min': min(recalls) if recalls else None,
                    'max': max(recalls) if recalls else None,
                    'median': statistics.median(recalls) if recalls else None,
                },
                'f1_score': {
                    'values': f1_scores,
                    'mean': statistics.mean(f1_scores) if f1_scores else None,
                    'std': statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0,
                    'min': min(f1_scores) if f1_scores else None,
                    'max': max(f1_scores) if f1_scores else None,
                    'median': statistics.median(f1_scores) if f1_scores else None,
                },
            }
            
            # Compute 95% CI for test accuracy
            if len(best_accs) > 1:
                mean = statistics.mean(best_accs)
                std = statistics.stdev(best_accs)
                n = len(best_accs)
                ci_95 = 1.96 * std / (n ** 0.5)
                stats_dict['test_accuracy_best']['ci_95'] = ci_95
            
            master_results['statistics'] = stats_dict
            
            # Save detailed statistics if requested
            if args.save_run_stats:
                stats_json_file = results_dir / f"statistics_{config_name}_{timestamp}.json"
                stats_txt_file = results_dir / f"statistics_{config_name}_{timestamp}.txt"
                
                # Save JSON
                with open(stats_json_file, 'w', encoding='utf-8') as f:
                    json.dump(stats_dict, f, indent=2)
                
                # Save TXT using ResultsFormatter
                stats_txt_content = ResultsFormatter.format_statistics_report(
                    stats_dict, 
                    master_results['configuration'], 
                    timestamp
                )
                # Add per-run details
                per_run_lines = ["Per-Run Results:", "-" * 70]
                for r in all_runs_results:
                    status = "✅" if not r['summary']['failed'] else "❌"
                    per_run_lines.append(f"{status} Run {r['run_number']}: Best Acc={r['summary']['best_accuracy']:.4f}, Final Test={r['summary']['final_test_accuracy']}, Time={r['summary']['total_duration_sec']:.1f}s")
                
                # Insert per-run details into stats report
                with open(stats_txt_file, 'w', encoding='utf-8') as f:
                    # Write first part (header)
                    f.write(f"Statistical Analysis - {args.runs} Runs\n")
                    f.write("="*70 + "\n")
                    f.write(f"Configuration: window={args.window}, throttle={args.throttle_ms}ms, epochs={args.epochs}\n")
                    f.write(f"Timestamp: {timestamp}\n\n")
                    # Write per-run details
                    f.write("\n".join(per_run_lines))
                    f.write("\n\n")
                    # Write statistics (skip header since we already wrote it)
                    stats_lines = stats_txt_content.split('\n')
                    # Find where summary statistics start
                    for i, line in enumerate(stats_lines):
                        if "Summary Statistics:" in line:
                            f.write("\n".join(stats_lines[i:]))
                            break
                
                logger.info(f"📊 Statistics saved: {stats_json_file.name}, {stats_txt_file.name}")
        
        # Request final comprehensive report from ESP32 (only for last run or single run)
        logger.info("="*70)
        logger.info("📊 Requesting comprehensive analysis from ESP32...")
        logger.info("="*70)
        
        # Storage for structured analysis data
        analysis_data = {
            'memory_efficiency': None,
            'performance_benchmarks': None,
            'protocol_statistics': None,
            'log_output': ""  # For bounds monitoring JSON and other LOG frames
        }
        
        try:
            seq = 1
            logger.debug("Sending OPC_SHOW_FINAL command...")
            seq = send_cmd(ser, seq, OPC_SHOW_FINAL)
            time.sleep(0.1)
            
            # Read structured analysis frames; only drain LOG frames when requested.
            timeout_s = 30.0 if args.final_logs else 8.0
            timeout = time.time() + timeout_s
            logger.debug("Starting to read frames from ESP32...")
            frames_read = 0
            got_structured = set()
            
            while time.time() < timeout:
                try:
                    fr = read_frame(ser, timeout_s=0.5)
                except Exception as e:
                    logger.debug(f"Exception reading frame: {e}")
                    continue
                    
                if fr:
                    logger.debug(f"Frame received: type=0x{fr.type:02X}, len={len(fr.payload)}, checksum_ok={fr.checksum_ok}")
                
                # Decode structured binary analysis frames
                if fr and fr.checksum_ok and fr.type == FRAME_TYPE_MEM_EFFICIENCY:
                    try:
                        analysis_data['memory_efficiency'] = decode_mem_efficiency(fr.payload)
                        got_structured.add('mem')
                        logger.success("✅ Received Memory Efficiency Analysis")
                    except Exception as e:
                        logger.error(f"Error decoding MEM_EFFICIENCY: {e}")
                elif fr and fr.checksum_ok and fr.type == FRAME_TYPE_PERF_BENCH:
                    try:
                        analysis_data['performance_benchmarks'] = decode_perf_bench(fr.payload)
                        got_structured.add('perf')
                        logger.success("✅ Received Performance Benchmarks")
                    except Exception as e:
                        logger.error(f"Error decoding PERF_BENCH: {e}")
                elif fr and fr.checksum_ok and fr.type == FRAME_TYPE_PROTOCOL_STATS:
                    try:
                        analysis_data['protocol_statistics'] = decode_protocol_stats(fr.payload)
                        got_structured.add('proto')
                        logger.success("✅ Received Protocol Statistics")
                    except Exception as e:
                        logger.error(f"Error decoding PROTOCOL_STATS: {e}")
                elif fr and fr.checksum_ok and fr.type == FRAME_TYPE_LOG:
                    frames_read += 1
                    if args.final_logs:
                        try:
                            line = fr.payload.decode('utf-8', errors='replace').rstrip()
                            logger.info(line)
                            analysis_data['log_output'] += line + "\n"
                        except Exception as e:
                            logger.debug(f"Error decoding LOG frame: {e}")
                elif fr and fr.type == FRAME_TYPE_READY:
                    # Device sent READY - analysis is complete
                    logger.debug("Received READY frame - analysis complete")
                    break
                elif not fr:
                    # If only structured data is requested, finish as soon as all frames are collected.
                    if not args.final_logs and len(got_structured) == 3:
                        break
                if not args.final_logs and len(got_structured) == 3:
                    logger.debug("All structured analysis frames collected; skipping verbose LOG drain")
                    break
            
            if frames_read > 0:
                logger.success(f"✅ Read {frames_read} LOG frames from ESP32")
            else:
                logger.debug(f"No LOG frames received (structured data only)")
            
            # Store ALL analysis data in master results
            master_results['comprehensive_analysis'] = analysis_data
            
            if analysis_data['memory_efficiency'] or analysis_data['performance_benchmarks'] or analysis_data['protocol_statistics']:
                logger.success(f"✅ Received structured analysis data from ESP32")
        except Exception as e:
            logger.error(f"⚠️  Error reading final report: {e}")
            master_results['comprehensive_analysis'] = {'error': str(e)}
        
        # Software reset: Always return device to READY state for next run (no hardware reboot)
        logger.info("🔄 Performing software reset to READY state...")
        seq = 1
        seq = send_cmd(ser, seq, OPC_RESET_STATE)
        time.sleep(0.2)
        
        # Wait for READY confirmation
        ready_ok, dev_trained, dev_tested = wait_ready(ser, timeout_s=3.0)
        if ready_ok:
            logger.success(f"✅ Device reset to READY (cumulative: trained={dev_trained}, tested={dev_tested})")
        else:
            logger.warning("⚠️  Did not receive READY confirmation after reset")
        
        # === SAVE MASTER RESULTS (JSON + TXT) ===
        logger.info("="*70)
        logger.info("💾 Saving experiment results...")
        logger.info("="*70)
        
        result_json_file = results_dir / f"experiment_{config_name}_{timestamp}.json"
        result_txt_file = results_dir / f"experiment_{config_name}_{timestamp}.txt"
        
        # Save complete data as JSON
        with open(result_json_file, 'w', encoding='utf-8') as f:
            json.dump(master_results, f, indent=2)
        logger.success(f"✅ JSON data saved to: {result_json_file}")
        
        # Generate human-readable text report using ResultsFormatter
        with open(result_txt_file, 'w', encoding='utf-8') as f:
            f.write(ResultsFormatter.format_full_report(master_results))
        
        logger.success(f"✅ Text report saved to: {result_txt_file}")
        logger.info("="*70)
        logger.success("✨ All results saved successfully!")
        logger.info("="*70)

if __name__=="__main__":
    main()
