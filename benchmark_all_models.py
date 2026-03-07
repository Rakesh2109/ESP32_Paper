#!/usr/bin/env python3
"""
Comprehensive benchmark script for all ML models on ESP32-S3
Tests: Hoeffding, HAT, EFDT, SGT models
Measures: Throughput (smp/s), Accuracy, Train/Test time, Memory
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import serial

# Model configurations
MODELS = {
    'HOEFFDING': {
        'model_enum': 'MODEL_HOEFFDING',
        'name': 'Hoeffding Tree',
        'expected_size': '13.8 KB',
        'description': 'Fast, minimal drift detection'
    },
    'HAT': {
        'model_enum': 'MODEL_HAT',
        'name': 'Hoeffding Adaptive Tree (HAT)',
        'expected_size': '16.7 KB',
        'description': 'Drift detection with alternate trees'
    },
    'EFDT': {
        'model_enum': 'MODEL_EFDT',
        'name': 'Extremely Fast DT (EFDT)',
        'expected_size': '13.8 KB',
        'description': 'Optimized for speed'
    },
    'SGT': {
        'model_enum': 'MODEL_SGT',
        'name': 'Soft Gauss Tree (SGT)',
        'expected_size': '5 KB',
        'description': 'Smallest model'
    },
}

SERIAL_PORT = '/dev/cu.usbmodem1101'
DATASET = 'IoT_clean'
WINDOW_SIZE = 64
BAUD = 921600

class BenchmarkRunner:
    def __init__(self):
        self.results = {}
        self.project_root = Path(__file__).parent
        
    def modify_model(self, model_key):
        """Change MODEL_TYPE in main.cpp"""
        main_cpp_path = self.project_root / 'src' / 'core' / 'main.cpp'
        with open(main_cpp_path, 'r') as f:
            content = f.read()
        
        # Find and replace #define MODEL_TYPE
        import re
        old = re.search(r'#define MODEL_TYPE MODEL_\w+', content)
        if not old:
            print(f"❌ Could not find MODEL_TYPE definition")
            return False
            
        model_enum = MODELS[model_key]['model_enum']
        new = f'#define MODEL_TYPE {model_enum}'
        content = content.replace(old.group(0), new)
        
        with open(main_cpp_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Changed model to {model_key}")
        return True
    
    def build_firmware(self):
        """Build firmware"""
        print("\n🔨 Building firmware...")
        result = subprocess.run(
            ['pio', 'run', '--environment', 'esp32_s3_n16r8'],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"❌ Build failed!")
            print(result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            return False
        
        print(f"✅ Build successful")
        return True
    
    def upload_firmware(self):
        """Upload firmware to device"""
        print("📤 Uploading firmware...")
        result = subprocess.run(
            ['pio', 'run', '--environment', 'esp32_s3_n16r8', '--target', 'upload'],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"❌ Upload failed!")
            return False
        
        print(f"✅ Upload successful")
        time.sleep(2)  # Wait for device reset
        return True
    
    def run_test(self, model_key):
        """Run test with test_serial.py"""
        print(f"\n🧪 Testing {MODELS[model_key]['name']}...")
        
        cmd = [
            'python3', 'test_serial.py',
            '-p', SERIAL_PORT,
            '--dataset', DATASET,
            '--input-real',
            '--window', str(WINDOW_SIZE)
        ]
        
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=180
        )
        
        if result.returncode != 0:
            print(f"❌ Test failed!")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return None
        
        # Parse results from output
        metrics = self._parse_output(result.stdout)
        print(f"✅ Test completed")
        
        return metrics
    
    def _parse_output(self, output):
        """Extract metrics from test output"""
        metrics = {
            'train_samples': None,
            'train_time': None,
            'train_throughput': None,
            'train_accuracy': None,
            'train_ema': None,
            'test_samples': None,
            'test_time': None,
            'test_accuracy': None,
            'model_size': None,
            'heap_used': None,
        }
        
        import re
        
        # Parse train metrics
        train_match = re.search(r'train:\s+(\d+)\s+smp\s+in\s+([\d.]+)s\s+\(([\d.]+)\s+smp/s\).*acc=([\d.]+).*ema=([\d.]+)', output)
        if train_match:
            metrics['train_samples'] = int(train_match.group(1))
            metrics['train_time'] = float(train_match.group(2))
            metrics['train_throughput'] = float(train_match.group(3))
            metrics['train_accuracy'] = float(train_match.group(4))
            metrics['train_ema'] = float(train_match.group(5))
        
        # Parse test metrics
        test_match = re.search(r'test:\s+(\d+)\s+smp\s+in\s+([\d.]+)s.*acc=([\d.]+)', output)
        if test_match:
            metrics['test_samples'] = int(test_match.group(1))
            metrics['test_time'] = float(test_match.group(2))
            metrics['test_accuracy'] = float(test_match.group(3))
        
        # Parse memory
        memory_match = re.search(r'Model Weights:\s+([\d.]+)\s+bytes\s+\(([\d.]+)\s+KB\)', output)
        if memory_match:
            metrics['model_size'] = float(memory_match.group(2))
        
        heap_match = re.search(r'Free Heap:\s+(\d+)\s+bytes', output)
        if heap_match:
            metrics['heap_used'] = int(heap_match.group(1))
        
        return metrics
    
    def benchmark_all(self):
        """Run all model benchmarks"""
        print("=" * 80)
        print("🚀 ESP32-S3 ML Model Benchmark Suite")
        print("=" * 80)
        print(f"Port: {SERIAL_PORT} @ {BAUD} baud")
        print(f"Dataset: {DATASET}")
        print(f"Window: {WINDOW_SIZE}")
        print("=" * 80)
        
        for model_key in MODELS.keys():
            print(f"\n\n{'='*80}")
            print(f"⚙️  Testing: {MODELS[model_key]['name']}")
            print(f"{'='*80}")
            
            # Modify source
            if not self.modify_model(model_key):
                print(f"⏭️  Skipping {model_key}")
                continue
            
            # Build
            if not self.build_firmware():
                print(f"⏭️  Skipping {model_key}")
                continue
            
            # Upload
            if not self.upload_firmware():
                print(f"⏭️  Skipping {model_key}")
                continue
            
            # Test
            metrics = self.run_test(model_key)
            if metrics:
                self.results[model_key] = metrics
            
            # Small delay between tests
            time.sleep(2)
        
        # Save and display results
        self._save_results()
        self._display_results()
    
    def _save_results(self):
        """Save results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.project_root / 'results' / f'benchmark_{timestamp}.json'
        results_file.parent.mkdir(exist_ok=True)
        
        # Prepare data
        data = {
            'timestamp': timestamp,
            'device': 'ESP32-S3 N16R8',
            'dataset': DATASET,
            'window_size': WINDOW_SIZE,
            'models': {}
        }
        
        for model_key, metrics in self.results.items():
            data['models'][model_key] = {
                'name': MODELS[model_key]['name'],
                'metrics': metrics
            }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
    
    def _display_results(self):
        """Display benchmark results as table"""
        if not self.results:
            print("❌ No results to display")
            return
        
        print("\n\n" + "=" * 120)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 120)
        
        # Header
        print(f"\n{'Model':<20} {'Size':>10} {'Train Acc':>12} {'Train(s)':>10} {'Train(sps)':>12} {'Test Acc':>12} {'Test(s)':>10} {'Test(sps)':>12}")
        print("-" * 120)
        
        # Results
        for model_key in MODELS.keys():
            if model_key not in self.results:
                print(f"{MODELS[model_key]['name']:<20} {'SKIPPED':>10}")
                continue
            
            m = self.results[model_key]
            model_name = MODELS[model_key]['name']
            
            size = f"{m.get('model_size', 0):.1f}KB" if m.get('model_size') else "N/A"
            train_acc = f"{m.get('train_accuracy', 0)*100:.2f}%" if m.get('train_accuracy') else "N/A"
            train_time = f"{m.get('train_time', 0):.1f}s" if m.get('train_time') else "N/A"
            train_sps = f"{m.get('train_throughput', 0):.0f}" if m.get('train_throughput') else "N/A"
            test_acc = f"{m.get('test_accuracy', 0)*100:.2f}%" if m.get('test_accuracy') else "N/A"
            test_time = f"{m.get('test_time', 0):.1f}s" if m.get('test_time') else "N/A"
            test_sps = f"{m.get('test_samples', 0)/m.get('test_time', 1):.0f}" if m.get('test_time') else "N/A"
            
            print(f"{model_name:<20} {size:>10} {train_acc:>12} {train_time:>10} {train_sps:>12} {test_acc:>12} {test_time:>10} {test_sps:>12}")
        
        # Detailed breakdown
        print("\n" + "=" * 120)
        print("📈 DETAILED RESULTS")
        print("=" * 120)
        
        for model_key in MODELS.keys():
            if model_key not in self.results:
                continue
            
            m = self.results[model_key]
            print(f"\n🔹 {MODELS[model_key]['name']}")
            print(f"   Description: {MODELS[model_key]['description']}")
            print(f"   Expected Size: {MODELS[model_key]['expected_size']}")
            print(f"   \n   Training:")
            print(f"     Samples: {m.get('train_samples', 0):,}")
            print(f"     Time: {m.get('train_time', 0):.1f}s")
            print(f"     Throughput: {m.get('train_throughput', 0):.1f} smp/s")
            print(f"     Accuracy (cumulative): {m.get('train_accuracy', 0)*100:.2f}%")
            print(f"     Accuracy (EMA): {m.get('train_ema', 0)*100:.2f}%")
            print(f"   \n   Testing:")
            print(f"     Samples: {m.get('test_samples', 0):,}")
            print(f"     Time: {m.get('test_time', 0):.1f}s")
            test_sps = m.get('test_samples', 0) / m.get('test_time', 1)
            print(f"     Throughput: {test_sps:.1f} smp/s")
            print(f"     Accuracy: {m.get('test_accuracy', 0)*100:.2f}%")
            print(f"   \n   Hardware:")
            print(f"     Model Size: {m.get('model_size', 0):.1f} KB")
            print(f"     Free Heap: {m.get('heap_used', 0):,} bytes")

if __name__ == '__main__':
    runner = BenchmarkRunner()
    try:
        runner.benchmark_all()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        runner._display_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
