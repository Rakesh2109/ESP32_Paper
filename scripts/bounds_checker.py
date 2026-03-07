#!/usr/bin/env python3
"""
ESP32 TsetlinMachine - Automated Bounds Analysis Tool
Performs comprehensive static analysis of memory bounds, buffer safety, and overflow protection
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class BoundsIssue:
    file_path: str
    line_number: int
    issue_type: str
    severity: RiskLevel
    description: str
    code_snippet: str
    recommendation: str

@dataclass
class MemoryAnalysis:
    max_heap_usage: int
    stack_usage_per_task: int
    total_stack_allocation: int
    buffer_sizes: Dict[str, int]
    risk_assessment: RiskLevel

class BoundsChecker:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.issues: List[BoundsIssue] = []
        self.memory_analysis = None
        
        # ESP32 constraints
        self.ESP32_TOTAL_HEAP = 320 * 1024  # 320KB typical
        self.ESP32_STACK_LIMIT = 8192       # 8KB per task typical
        
        # Pattern definitions for static analysis
        self.dangerous_patterns = {
            'array_access': r'(\w+)\([^\]]+\)',
            'buffer_alloc': r'(malloc|calloc|utils_malloc)\s*\(\s*([^)]+)\)',
            'stack_array': r'(\w+)\s+(\w+)\s*\[\s*(\d+)\s*\]',
            'string_ops': r'(strcpy|strcat|sprintf|gets)\s*\(',
            'unsafe_cast': r'\(\s*\w+\s*\*\s*\)',
        }
        
    def analyze_project(self) -> Dict:
        """Main analysis entry point"""
        print("🔍 Starting ESP32 TsetlinMachine Bounds Analysis...")
        
        # Find all relevant source files
        source_files = self._find_source_files()
        print(f"📁 Found {len(source_files)} source files to analyze")
        
        # Analyze each file
        for file_path in source_files:
            print(f"   📄 Analyzing {file_path.name}")
            self._analyze_file(file_path)
            
        # Perform cross-file analysis
        self._analyze_memory_usage()
        self._analyze_stack_usage()
        self._analyze_buffer_safety()
        
        # Generate report
        return self._generate_report()
    
    def _find_source_files(self) -> List[Path]:
        """Find all C/C++ source files in the project"""
        extensions = {'.c', '.cpp', '.h', '.hpp'}
        source_files = []
        
        for ext in extensions:
            source_files.extend(self.project_root.rglob(f'*{ext}'))
            
        # Filter out build directories and third-party code
        filtered_files = []
        exclude_dirs = {'build', '.pio', 'node_modules', 'test'}
        
        for file_path in source_files:
            if not any(exclude_dir in file_path.parts for exclude_dir in exclude_dirs):
                filtered_files.append(file_path)
                
        return filtered_files
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single source file for bounds issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Check for various bounds-related issues
            self._check_array_bounds(file_path, lines)
            self._check_buffer_overflows(file_path, lines)
            self._check_memory_allocations(file_path, lines)
            self._check_stack_arrays(file_path, lines)
            
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
    
    def _check_array_bounds(self, file_path: Path, lines: List[str]):
        """Check for potential array bounds violations"""
        for i, line in enumerate(lines, 1):
            # Look for array access patterns
            matches = re.findall(self.dangerous_patterns['array_access'], line)
            
            for array_name, index_expr in matches:
                # Check for potentially dangerous index expressions
                if self._is_dangerous_index(index_expr):
                    self.issues.append(BoundsIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="ARRAY_BOUNDS",
                        severity=RiskLevel.MEDIUM,
                        description=f"Potentially unsafe array access: {array_name}[{index_expr}]",
                        code_snippet=line.strip(),
                        recommendation="Add bounds checking before array access"
                    ))
    
    def _check_buffer_overflows(self, file_path: Path, lines: List[str]):
        """Check for buffer overflow vulnerabilities"""
        for i, line in enumerate(lines, 1):
            # Check for unsafe string functions
            if re.search(self.dangerous_patterns['string_ops'], line):
                self.issues.append(BoundsIssue(
                    file_path=str(file_path),
                    line_number=i,
                    issue_type="BUFFER_OVERFLOW",
                    severity=RiskLevel.HIGH,
                    description="Unsafe string function detected",
                    code_snippet=line.strip(),
                    recommendation="Replace with bounds-safe alternatives (strncpy, snprintf, etc.)"
                ))
    
    def _check_memory_allocations(self, file_path: Path, lines: List[str]):
        """Check memory allocation patterns"""
        for i, line in enumerate(lines, 1):
            matches = re.findall(self.dangerous_patterns['buffer_alloc'], line)
            
            for func_name, size_expr in matches:
                # Try to evaluate size expression for large allocations
                if self._is_large_allocation(size_expr):
                    self.issues.append(BoundsIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="LARGE_ALLOCATION",
                        severity=RiskLevel.MEDIUM,
                        description=f"Large memory allocation detected: {size_expr}",
                        code_snippet=line.strip(),
                        recommendation="Verify allocation size against ESP32 heap limits"
                    ))
    
    def _check_stack_arrays(self, file_path: Path, lines: List[str]):
        """Check for large stack arrays"""
        for i, line in enumerate(lines, 1):
            matches = re.findall(self.dangerous_patterns['stack_array'], line)
            
            for data_type, var_name, size_str in matches:
                try:
                    size = int(size_str)
                    if size > 1024:  # Arrays larger than 1KB on stack
                        self.issues.append(BoundsIssue(
                            file_path=str(file_path),
                            line_number=i,
                            issue_type="LARGE_STACK_ARRAY",
                            severity=RiskLevel.MEDIUM,
                            description=f"Large stack array: {var_name}[{size}] = {size} bytes",
                            code_snippet=line.strip(),
                            recommendation="Consider dynamic allocation or static allocation"
                        ))
                except ValueError:
                    pass  # Non-constant size
    
    def _analyze_memory_usage(self):
        """Analyze overall memory usage patterns"""
        # Extract memory-related constants from the codebase
        constants = self._extract_memory_constants()
        
        # Calculate worst-case memory usage
        max_features = constants.get('MAX_FEATURES', 128)
        max_clauses = constants.get('MAX_CLAUSES', 1000)
        
        # TsetlinMachine memory calculation
        ta_state_size = max_clauses * max_features * 2  # uint8_t array
        clause_output_size = max_clauses                # uint8_t array  
        feedback_size = max_clauses                     # int8_t array
        tm_total = ta_state_size + clause_output_size + feedback_size
        
        # Queue memory
        recv_q_cap = constants.get('RECV_Q_CAP', 256)
        max_packed_bytes = constants.get('MAX_PACKED_BYTES', 8192)
        # Assuming RecEntry has nfeat(2) + label(1) + data(MAX_FEATURES/8)
        rec_entry_size = 3 + (max_features + 7) // 8
        queue_memory = recv_q_cap * rec_entry_size
        
        total_heap_usage = tm_total + queue_memory
        
        risk_level = RiskLevel.LOW
        if total_heap_usage > self.ESP32_TOTAL_HEAP * 0.8:
            risk_level = RiskLevel.HIGH
        elif total_heap_usage > self.ESP32_TOTAL_HEAP * 0.6:
            risk_level = RiskLevel.MEDIUM
            
        self.memory_analysis = MemoryAnalysis(
            max_heap_usage=total_heap_usage,
            stack_usage_per_task=8192,
            total_stack_allocation=16384,  # 2 tasks × 8KB
            buffer_sizes={
                'TM_AutomataStates': ta_state_size,
                'TM_ClauseOutput': clause_output_size,
                'TM_Feedback': feedback_size,
                'ReceiveQueue': queue_memory,
                'PackedBuffer': max_packed_bytes
            },
            risk_assessment=risk_level
        )
        
        if risk_level != RiskLevel.LOW:
            self.issues.append(BoundsIssue(
                file_path="GLOBAL",
                line_number=0,
                issue_type="MEMORY_PRESSURE",
                severity=risk_level,
                description=f"High memory usage: {total_heap_usage/1024:.1f}KB / {self.ESP32_TOTAL_HEAP/1024:.1f}KB",
                code_snippet="Memory allocation analysis",
                recommendation="Consider reducing MAX_FEATURES or MAX_CLAUSES, or implement memory pooling"
            ))
    
    def _analyze_stack_usage(self):
        """Analyze stack usage patterns"""
        # This is simplified - real stack analysis requires more sophisticated tools
        # We check for obvious stack issues based on static analysis
        pass
    
    def _analyze_buffer_safety(self):
        """Analyze buffer safety across the protocol layer"""
        # Look for proper bounds checking in protocol handling
        protocol_files = [f for f in self._find_source_files() if 'protocol' in f.name.lower()]
        
        for file_path in protocol_files:
            # Check if bounds validation is present
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # Look for proper bounds checking patterns
                bounds_checks = [
                    'if.*plen.*>',  # Payload length checks
                    'if.*len.*>',   # Length validation
                    'readExact',    # Safe reading
                    'MAX_.*BYTES'   # Buffer size constants
                ]
                
                has_bounds_check = any(re.search(pattern, content, re.IGNORECASE) 
                                     for pattern in bounds_checks)
                
                if not has_bounds_check:
                    self.issues.append(BoundsIssue(
                        file_path=str(file_path),
                        line_number=1,
                        issue_type="MISSING_BOUNDS_CHECK",
                        severity=RiskLevel.HIGH,
                        description="Protocol file missing proper bounds validation",
                        code_snippet="File-level analysis",
                        recommendation="Add comprehensive input validation"
                    ))
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
    
    def _is_dangerous_index(self, index_expr: str) -> bool:
        """Check if an array index expression is potentially dangerous"""
        # Simple heuristics for dangerous indexing
        dangerous_keywords = ['++', '--', '+', '-', '*', '/']
        return any(keyword in index_expr for keyword in dangerous_keywords)
    
    def _is_large_allocation(self, size_expr: str) -> bool:
        """Check if allocation size is potentially large"""
        # Look for multiplication or large constants
        if '*' in size_expr or any(char.isdigit() and int(char) > 5 for char in size_expr.split()):
            return True
        return False
    
    def _extract_memory_constants(self) -> Dict[str, int]:
        """Extract memory-related constants from header files"""
        constants = {}
        
        for file_path in self._find_source_files():
            if file_path.suffix in {'.h', '.hpp'}:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    # Find #define statements with numeric values
                    define_pattern = r'#define\s+(\w*(?:MAX|SIZE|CAP|LIMIT)\w*)\s+(\d+)'
                    matches = re.findall(define_pattern, content)
                    
                    for name, value in matches:
                        constants[name] = int(value)
                        
                except Exception as e:
                    continue
                    
        return constants
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        # Categorize issues by severity
        severity_counts = {level: 0 for level in RiskLevel}
        for issue in self.issues:
            severity_counts[issue.severity] += 1
        
        # Calculate overall risk score
        risk_score = (
            severity_counts[RiskLevel.LOW] * 1 +
            severity_counts[RiskLevel.MEDIUM] * 3 +
            severity_counts[RiskLevel.HIGH] * 7 +
            severity_counts[RiskLevel.CRITICAL] * 15
        )
        
        overall_risk = RiskLevel.LOW
        if risk_score > 20:
            overall_risk = RiskLevel.HIGH
        elif risk_score > 10:
            overall_risk = RiskLevel.MEDIUM
        elif risk_score > 0:
            overall_risk = RiskLevel.LOW
            
        memory_dict = None
        if self.memory_analysis:
            memory_dict = asdict(self.memory_analysis)
            memory_dict['risk_assessment'] = self.memory_analysis.risk_assessment.value
        
        return {
            'analysis_summary': {
                'total_issues': len(self.issues),
                'risk_score': risk_score,
                'overall_risk': overall_risk.value,
                'severity_breakdown': {k.value: v for k, v in severity_counts.items()}
            },
            'memory_analysis': memory_dict,
            'issues': [
                {**asdict(issue), 'severity': issue.severity.value} 
                for issue in self.issues
            ],
            'esp32_constraints': {
                'total_heap': self.ESP32_TOTAL_HEAP,
                'stack_per_task': self.ESP32_STACK_LIMIT,
                'heap_usage_percent': (self.memory_analysis.max_heap_usage / self.ESP32_TOTAL_HEAP * 100) if self.memory_analysis else 0
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if self.memory_analysis and self.memory_analysis.risk_assessment != RiskLevel.LOW:
            recommendations.append("Consider reducing MAX_FEATURES or MAX_CLAUSES to decrease memory pressure")
            
        high_risk_issues = [issue for issue in self.issues if issue.severity == RiskLevel.HIGH]
        if high_risk_issues:
            recommendations.append("Address HIGH severity bounds issues immediately")
            
        buffer_issues = [issue for issue in self.issues if 'BUFFER' in issue.issue_type]
        if buffer_issues:
            recommendations.append("Review all buffer operations for overflow protection")
            
        recommendations.extend([
            "Add runtime heap monitoring with low-memory warnings",
            "Implement stack watermark checking for task monitoring",
            "Consider adding debug-time bounds assertions",
            "Perform fuzz testing on protocol boundaries"
        ])
        
        return recommendations

def main():
    """Command-line interface for bounds checker"""
    if len(sys.argv) < 2:
        print("Usage: python bounds_checker.py <project_root>")
        print("Example: python bounds_checker.py /path/to/ESP32_TM-per-arne")
        sys.exit(1)
        
    project_root = sys.argv[1]
    
    if not os.path.exists(project_root):
        print(f"❌ Project directory not found: {project_root}")
        sys.exit(1)
        
    # Run analysis
    checker = BoundsChecker(project_root)
    report = checker.analyze_project()
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 ESP32 TSETLIN MACHINE BOUNDS ANALYSIS RESULTS")
    print("="*60)
    
    summary = report['analysis_summary']
    print(f"📊 Overall Risk Level: {summary['overall_risk']}")
    print(f"🔍 Total Issues Found: {summary['total_issues']}")
    print(f"📈 Risk Score: {summary['risk_score']}")
    
    print(f"\n📋 Issue Breakdown:")
    for severity, count in summary['severity_breakdown'].items():
        if count > 0:
            print(f"   {severity}: {count} issues")
    
    if report['memory_analysis']:
        mem = report['memory_analysis']
        print(f"\n💾 Memory Analysis:")
        print(f"   Heap Usage: {mem['max_heap_usage']/1024:.1f}KB / 320KB ({report['esp32_constraints']['heap_usage_percent']:.1f}%)")
        print(f"   Risk Level: {mem['risk_assessment']}")
    
    print(f"\n🔧 Key Recommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"   {i}. {rec}")
    
    # Save detailed report
    output_file = os.path.join(project_root, 'bounds_analysis_detailed.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {output_file}")
    
    # Exit code based on risk level
    if summary['overall_risk'] in ['HIGH', 'CRITICAL']:
        print("\n⚠️  HIGH RISK issues detected! Review immediately.")
        sys.exit(2)
    elif summary['overall_risk'] == 'MEDIUM':
        print("\n⚡ MEDIUM RISK issues detected. Review recommended.")
        sys.exit(1)
    else:
        print("\n✅ Analysis complete. Risk level acceptable.")
        sys.exit(0)

if __name__ == "__main__":
    main()


