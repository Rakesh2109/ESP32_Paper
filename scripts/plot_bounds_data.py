#!/usr/bin/env python3
"""
Generate graphs from experimental bounds checking data
Run after collecting data from ESP32
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import re
import sys
from pathlib import Path

def parse_csv_from_serial(log_file):
    """Extract CSV data from serial output"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract main metrics
    metrics = {}
    csv_match = re.search(r'=== BOUNDS_DATA_CSV_START ===\n(.*?)\n=== BOUNDS_DATA_CSV_END ===', content, re.DOTALL)
    if csv_match:
        lines = csv_match.group(1).strip().split('\n')
        for line in lines[1:]:  # Skip header
            if ',' in line:
                key, value = line.split(',', 1)
                try:
                    metrics[key] = float(value)
                except:
                    metrics[key] = value
    
    # Extract heap history
    heap_history = []
    heap_match = re.search(r'=== HEAP_HISTORY_CSV ===\n(.*?)\n\n', content, re.DOTALL)
    if heap_match:
        lines = heap_match.group(1).strip().split('\n')
        for line in lines[1:]:  # Skip header
            if ',' in line:
                _, value = line.split(',')
                heap_history.append(int(value))
    
    # Extract stack history
    stack_history = []
    stack_match = re.search(r'=== STACK_HISTORY_CSV ===\n(.*?)\n', content, re.DOTALL)
    if stack_match:
        lines = stack_match.group(1).strip().split('\n')
        for line in lines[1:]:  # Skip header
            if ',' in line:
                _, value = line.split(',')
                stack_history.append(int(value))
    
    return metrics, heap_history, stack_history

def plot_memory_usage(metrics, heap_history, output_dir):
    """Plot heap memory usage over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('ESP32 Memory Bounds Analysis - Experimental Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Heap usage over time
    if heap_history:
        samples = range(len(heap_history))
        heap_kb = [h / 1024 for h in heap_history]
        
        ax1.plot(samples, heap_kb, 'b-', linewidth=1.5, label='Free Heap')
        ax1.axhline(y=50, color='r', linestyle='--', label='Critical Threshold (50KB)', linewidth=2)
        ax1.fill_between(samples, 0, 50, alpha=0.2, color='red', label='Danger Zone')
        ax1.fill_between(samples, 50, max(heap_kb), alpha=0.1, color='green')
        
        ax1.set_xlabel('Sample Number', fontsize=12)
        ax1.set_ylabel('Free Heap (KB)', fontsize=12)
        ax1.set_title('Heap Memory Usage Over Time', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"Min: {min(heap_kb):.1f} KB\nMax: {max(heap_kb):.1f} KB\nAvg: {np.mean(heap_kb):.1f} KB"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Memory allocation summary
    if metrics:
        categories = ['Heap Min', 'Heap Avg', 'Heap Max']
        values = [
            metrics.get('heap_min_bytes', 0) / 1024,
            metrics.get('heap_avg_bytes', 0) / 1024,
            metrics.get('heap_max_bytes', 0) / 1024
        ]
        colors = ['#ff4444', '#44ff44', '#4444ff']
        
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.axhline(y=320, color='k', linestyle='-', linewidth=2, label='Total Heap (320KB)')
        ax2.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Critical (50KB)')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f} KB',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('Memory (KB)', fontsize=12)
        ax2.set_title('Heap Memory Statistics', fontsize=14)
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_ylim(0, 350)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure1_memory_usage.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure1_memory_usage.pdf', bbox_inches='tight')
    print(f"✓ Saved: figure1_memory_usage.png/pdf")
    plt.close()

def plot_stack_usage(stack_history, output_dir):
    """Plot stack usage over time"""
    if not stack_history:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    samples = range(len(stack_history))
    stack_kb = [s / 1024 for s in stack_history]
    stack_percent = [s / 8192 * 100 for s in stack_history]
    
    ax.plot(samples, stack_percent, 'g-', linewidth=2, label='Stack Usage %')
    ax.axhline(y=75, color='orange', linestyle='--', label='Warning (75%)', linewidth=2)
    ax.axhline(y=90, color='red', linestyle='--', label='Critical (90%)', linewidth=2)
    ax.fill_between(samples, 90, 100, alpha=0.2, color='red', label='Overflow Risk')
    
    ax.set_xlabel('Sample Number', fontsize=12)
    ax.set_ylabel('Stack Usage (%)', fontsize=12)
    ax.set_title('Stack Usage Analysis - Experimental Validation', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add statistics
    stats_text = f"Peak: {max(stack_percent):.1f}%\nAvg: {np.mean(stack_percent):.1f}%\nSafety Margin: {100-max(stack_percent):.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure2_stack_usage.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure2_stack_usage.pdf', bbox_inches='tight')
    print(f"✓ Saved: figure2_stack_usage.png/pdf")
    plt.close()

def plot_protocol_validation(metrics, output_dir):
    """Plot protocol frame validation statistics"""
    if not metrics:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Protocol Bounds Validation - Experimental Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Frame validation pie chart
    valid = metrics.get('frames_valid', 0)
    rejected_size = metrics.get('frames_rejected_size', 0)
    rejected_checksum = metrics.get('frames_rejected_checksum', 0)
    rejected_format = metrics.get('frames_rejected_format', 0)
    
    labels = ['Valid Frames', 'Size Violation', 'Checksum Fail', 'Format Error']
    sizes = [valid, rejected_size, rejected_checksum, rejected_format]
    colors = ['#4CAF50', '#FF5722', '#FF9800', '#9C27B0']
    explode = (0.05, 0.1, 0.1, 0.1)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Frame Validation Distribution', fontsize=14)
    
    # Plot 2: Bar chart of rejections
    rejection_types = ['Size\nViolation', 'Checksum\nFail', 'Format\nError']
    rejection_counts = [rejected_size, rejected_checksum, rejected_format]
    colors_bar = ['#FF5722', '#FF9800', '#9C27B0']
    
    bars = ax2.bar(rejection_types, rejection_counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, rejection_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Number of Frames', fontsize=12)
    ax2.set_title('Bounds Violations Detected', fontsize=14)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add total rejections text
    total_rejected = sum(rejection_counts)
    total_frames = valid + total_rejected
    rejection_rate = (total_rejected / total_frames * 100) if total_frames > 0 else 0
    
    stats_text = f"Total Frames: {total_frames:,}\nRejected: {total_rejected:,}\nRejection Rate: {rejection_rate:.2f}%"
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure3_protocol_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure3_protocol_validation.pdf', bbox_inches='tight')
    print(f"✓ Saved: figure3_protocol_validation.png/pdf")
    plt.close()

def plot_bounds_summary(metrics, output_dir):
    """Create comprehensive bounds analysis summary"""
    if not metrics:
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    fig.suptitle('Comprehensive Bounds Analysis - Experimental Validation', fontsize=18, fontweight='bold')
    
    # 1. Memory Safety Status
    ax1 = fig.add_subplot(gs[0, 0])
    heap_min_kb = metrics.get('heap_min_bytes', 0) / 1024
    heap_utilization = (1 - heap_min_kb / 320) * 100
    
    categories = ['Used', 'Free']
    sizes = [heap_utilization, 100 - heap_utilization]
    colors = ['#FF6B6B' if heap_utilization > 80 else '#4ECDC4', '#95E1D3']
    
    wedges, texts, autotexts = ax1.pie(sizes, labels=categories, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f'Peak Heap Utilization: {heap_utilization:.1f}%', fontsize=12, fontweight='bold')
    
    # 2. Bounds Violations
    ax2 = fig.add_subplot(gs[0, 1])
    violations = metrics.get('bounds_violations', 0)
    checks = metrics.get('bounds_checks_passed', 0)
    
    categories = ['Passed', 'Violations']
    values = [checks, violations]
    colors = ['#2ECC71', '#E74C3C']
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_title('Array Bounds Checks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 3. Stack Safety
    ax3 = fig.add_subplot(gs[1, 0])
    stack_usage = metrics.get('stack_usage_percent', 0)
    stack_free = 100 - stack_usage
    
    categories = ['Used', 'Free']
    values = [stack_usage, stack_free]
    colors = ['#FFA07A' if stack_usage > 75 else '#98D8C8', '#F7DC6F']
    
    bars = ax3.barh(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%',
                ha='left', va='center', fontsize=11, fontweight='bold', color='black')
    
    ax3.axvline(x=75, color='orange', linestyle='--', linewidth=2, label='Warning')
    ax3.axvline(x=90, color='red', linestyle='--', linewidth=2, label='Critical')
    ax3.set_xlabel('Percentage (%)', fontsize=11)
    ax3.set_title(f'Peak Stack Usage: {stack_usage:.1f}%', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, axis='x', alpha=0.3)
    
    # 4. Memory Allocations
    ax4 = fig.add_subplot(gs[1, 1])
    malloc_calls = metrics.get('malloc_calls', 0)
    malloc_failures = metrics.get('malloc_failures', 0)
    free_calls = metrics.get('free_calls', 0)
    
    categories = ['Malloc\nCalls', 'Malloc\nFailures', 'Free\nCalls']
    values = [malloc_calls, malloc_failures, free_calls]
    colors = ['#3498DB', '#E74C3C', '#2ECC71']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Memory Allocation Operations', fontsize=12, fontweight='bold')
    ax4.grid(True, axis='y', alpha=0.3)
    
    # 5. Safety Score (custom metric)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate safety scores
    heap_score = 100 - heap_utilization  # More free = better
    stack_score = 100 - stack_usage  # More free = better
    bounds_score = (checks / (checks + violations) * 100) if (checks + violations) > 0 else 100
    protocol_score = (metrics.get('frames_valid', 0) / metrics.get('frames_received', 1) * 100)
    
    scores = [heap_score, stack_score, bounds_score, protocol_score]
    labels = ['Heap\nSafety', 'Stack\nSafety', 'Bounds\nIntegrity', 'Protocol\nValidation']
    colors = ['#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
    
    bars = ax5.bar(labels, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax5.axhline(y=80, color='orange', linestyle='--', linewidth=2, label='Good Threshold', alpha=0.7)
    ax5.axhline(y=95, color='green', linestyle='--', linewidth=2, label='Excellent Threshold', alpha=0.7)
    ax5.set_ylabel('Safety Score (%)', fontsize=12)
    ax5.set_title('Overall Safety Analysis (Higher = Better)', fontsize=14, fontweight='bold')
    ax5.set_ylim(0, 105)
    ax5.legend(loc='lower right', fontsize=10)
    ax5.grid(True, axis='y', alpha=0.3)
    
    overall_score = np.mean(scores)
    score_text = f"Overall Safety Score: {overall_score:.1f}%"
    score_color = '#2ECC71' if overall_score > 95 else '#F39C12' if overall_score > 80 else '#E74C3C'
    ax5.text(0.5, 0.95, score_text, transform=ax5.transAxes, 
            fontsize=14, verticalalignment='top', horizontalalignment='center',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=score_color, alpha=0.3, edgecolor='black', linewidth=2))
    
    plt.savefig(f'{output_dir}/figure4_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/figure4_comprehensive_summary.pdf', bbox_inches='tight')
    print(f"✓ Saved: figure4_comprehensive_summary.png/pdf")
    plt.close()

def generate_latex_table(metrics, output_dir):
    """Generate LaTeX table for paper"""
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Experimental Bounds Analysis Results}
\label{tab:bounds_results}
\begin{tabular}{|l|r|r|}
\hline
\textbf{Metric} & \textbf{Value} & \textbf{Status} \\
\hline
\hline
\multicolumn{3}{|c|}{\textbf{Memory Bounds}} \\
\hline
"""
    
    heap_min = metrics.get('heap_min_bytes', 0)
    heap_avg = metrics.get('heap_avg_bytes', 0)
    heap_max = metrics.get('heap_max_bytes', 0)
    critical_events = metrics.get('heap_critical_events', 0)
    
    latex_content += f"Min Free Heap & {heap_min/1024:.1f} KB & {'⚠️ Low' if heap_min < 51200 else '✓ OK'} \\\n"
    latex_content += f"Avg Free Heap & {heap_avg/1024:.1f} KB & ✓ OK \\\n"
    latex_content += f"Max Free Heap & {heap_max/1024:.1f} KB & ✓ OK \\\n"
    latex_content += f"Critical Events & {int(critical_events)} & {'⚠️ Warning' if critical_events > 0 else '✓ None'} \\\n"
    latex_content += r"""\hline
\multicolumn{3}{|c|}{\textbf{Array Bounds}} \\
\hline
"""
    
    array_accesses = metrics.get('array_accesses', 0)
    bounds_violations = metrics.get('bounds_violations', 0)
    
    latex_content += f"Total Accesses & {int(array_accesses):,} & ✓ OK \\\n"
    latex_content += f"Violations Detected & {int(bounds_violations)} & {'⚠️ Warning' if bounds_violations > 0 else '✓ None'} \\\n"
    latex_content += f"Success Rate & {(array_accesses-bounds_violations)/array_accesses*100:.2f}\\% & ✓ OK \\\n"
    
    latex_content += r"""\hline
\multicolumn{3}{|c|}{\textbf{Stack Usage}} \\
\hline
"""
    
    stack_usage = metrics.get('stack_usage_percent', 0)
    stack_warnings = metrics.get('stack_overflow_warnings', 0)
    
    latex_content += f"Peak Usage & {stack_usage:.1f}\\% & {'⚠️ High' if stack_usage > 75 else '✓ OK'} \\\n"
    latex_content += f"Safety Margin & {100-stack_usage:.1f}\\% & ✓ OK \\\n"
    latex_content += f"Overflow Warnings & {int(stack_warnings)} & {'⚠️ Warning' if stack_warnings > 0 else '✓ None'} \\\n"
    
    latex_content += r"""\hline
\multicolumn{3}{|c|}{\textbf{Protocol Validation}} \\
\hline
"""
    
    frames_total = metrics.get('frames_received', 0)
    frames_valid = metrics.get('frames_valid', 0)
    frames_rejected = frames_total - frames_valid
    
    latex_content += f"Frames Received & {int(frames_total):,} & ✓ OK \\\n"
    latex_content += f"Frames Rejected & {int(frames_rejected)} & ✓ OK \\\n"
    latex_content += f"Rejection Rate & {frames_rejected/frames_total*100:.2f}\\% & ✓ OK \\\n"
    
    latex_content += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(f'{output_dir}/bounds_table.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"✓ Saved: bounds_table.tex (for LaTeX paper)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_bounds_data.py <serial_log_file>")
        print("Example: python plot_bounds_data.py bounds_experiment.log")
        sys.exit(1)
    
    log_file = sys.argv[1]
    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = "bounds_graphs"
    Path(output_dir).mkdir(exist_ok=True)
    
    print("📊 Parsing experimental data...")
    metrics, heap_history, stack_history = parse_csv_from_serial(log_file)
    
    if not metrics:
        print("❌ No bounds data found in log file!")
        print("Make sure you ran the experiment and collected OPC_SHOW_FINAL output")
        sys.exit(1)
    
    print(f"✓ Found {len(metrics)} metrics, {len(heap_history)} heap samples, {len(stack_history)} stack samples")
    
    print("\n📈 Generating graphs...")
    plot_memory_usage(metrics, heap_history, output_dir)
    plot_stack_usage(stack_history, output_dir)
    plot_protocol_validation(metrics, output_dir)
    plot_bounds_summary(metrics, output_dir)
    generate_latex_table(metrics, output_dir)
    
    print(f"\n✅ All graphs generated in '{output_dir}/' directory")
    print("\n📄 Files for your paper:")
    print("   - figure1_memory_usage.png/pdf")
    print("   - figure2_stack_usage.png/pdf")
    print("   - figure3_protocol_validation.png/pdf")
    print("   - figure4_comprehensive_summary.png/pdf")
    print("   - bounds_table.tex (LaTeX table)")
    print("\n💡 You can now use these in your academic paper!")

if __name__ == "__main__":
    main()

 

