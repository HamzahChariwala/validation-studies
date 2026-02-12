#!/usr/bin/env python3
"""
Compute Roofline Profiler - Multi-GPU Version

Production-ready GPU compute benchmarking tool that extracts roofline model parameters
(peak FLOPS and memory bandwidth) for use in performance simulators.

FEATURES:
- Profiles matrix multiplications across size/precision/batch ranges
- Supports square and non-square matrices (production workloads)
- Multi-GPU support with torchrun (profiles all GPUs in parallel)
- Generates PyTorch profiler traces with GPU kernel timing
- Extracts actual GPU kernel execution times (not CPU dispatch times)
- Computes arithmetic intensity for each operation
- Fits roofline models per (precision, batch_size) + overall per precision
- GPU clock locking (default: ON, requires sudo)
- Comprehensive output (JSON, CSV, human-readable summary)
- Automatic trace cleanup after extraction
- Graceful error handling for unsupported dtypes

USAGE:
    # Multi-GPU (recommended - profiles all GPUs)
    torchrun --nproc_per_node=4 compute/matmul/minimal/profile_minimal.py
    
    # Multi-GPU without clock locking
    torchrun --nproc_per_node=4 compute/matmul/minimal/profile_minimal.py --no-lock-clocks
    
    # Single GPU (backward compatible)
    python3 compute/matmul/minimal/profile_minimal.py --no-lock-clocks
    
    # Custom configuration
    torchrun --nproc_per_node=4 compute/matmul/minimal/profile_minimal.py \
        --output-dir ./my_results \
        --repeat-count 10

DEFAULTS:
    - Clock locking: ENABLED (use --no-lock-clocks to disable)
    - Matrix sizes: 14 sizes (6 square + 8 non-square)
    - Precisions: fp32, fp16, bf16 (fp8 if H100+)
    - Batch sizes: [1, 4, 8, 16]
    - Repetitions: 5 per config
    - Output: ./compute_roofline_results_TIMESTAMP/

REQUIREMENTS:
    - PyTorch with CUDA support
    - numpy, scipy, scikit-learn
    - sudo access (optional, for clock locking)

OUTPUT FILES (per GPU + aggregated):
    - RESULTS_SUMMARY.txt      (comprehensive human-readable report)
    - results.json             (structured data with roofline models)
    - roofline_parameters.csv  (fitted params per precision × batch size)
    - measurements.csv         (raw kernel timing data)
    - gpu_info.txt             (GPU hardware information)
"""

import torch
import torch.distributed as dist
import time
import os
import sys
import csv
import subprocess
import argparse
import json
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add repo root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../../..'))
sys.path.insert(0, repo_root)

# Import configuration and utilities (from local directory)
from compute_config import (
    generate_benchmark_configs,
    get_config_summary,
    dtype_str_to_torch,
    compute_matmul_metrics,
    WARMUP_ITERATIONS,
    REPEAT_COUNT,
)

# Import roofline fitting (multi-method)
from compute.matmul.analysis.fit_roofline import fit_roofline_all_methods

# Import distributed and profiler utilities (reuse from networking)
sys.path.insert(0, os.path.join(repo_root, 'networking'))
from networking.utils.profiling_utils import configure_profiler, ProfilerContext
from networking.utils.distributed_utils import (
    init_distributed,
    cleanup_distributed,
)


# ============================================================================
# DISTRIBUTED UTILITIES
# ============================================================================

def print_once(msg: str, rank: Optional[int] = None):
    """Print message only from rank 0."""
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(msg)


def barrier_all():
    """Synchronization barrier across all ranks."""
    if dist.is_initialized():
        dist.barrier()


# ============================================================================
# GPU INFORMATION AND CLOCK LOCKING
# ============================================================================

def get_gpu_info(gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get comprehensive GPU information.
    
    Args:
        gpu_id: GPU ID to query (default: current device)
    
    Returns:
        Dictionary with GPU metadata
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    if gpu_id is None:
        gpu_id = torch.cuda.current_device()
    
    props = torch.cuda.get_device_properties(gpu_id)
    
    info = {
        'gpu_id': gpu_id,
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'total_memory_gb': props.total_memory / (1024**3),
        'multi_processor_count': props.multi_processor_count,
    }
    
    # Query nvidia-smi for additional info
    try:
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=driver_version,memory.total,clocks.max.graphics,clocks.max.sm,clocks.max.memory',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        values = result.stdout.strip().split(',')
        if len(values) >= 5:
            info['driver_version'] = values[0].strip()
            info['memory_total_mb'] = values[1].strip()
            info['clock_max_graphics_mhz'] = values[2].strip()
            info['clock_max_sm_mhz'] = values[3].strip()
            info['clock_max_memory_mhz'] = values[4].strip()
    except:
        pass
    
    # CUDA/PyTorch versions
    info['cuda_version'] = torch.version.cuda
    info['pytorch_version'] = torch.__version__
    info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    
    return info


def lock_gpu_clocks(gpu_id: int, rank: int = 0):
    """Lock GPU clocks to maximum frequency (requires sudo)."""
    print_once(f"\nLocking GPU {gpu_id} clocks to maximum frequency...", rank)
    print_once("  (This requires sudo access)", rank)
    
    try:
        # Enable persistence mode
        subprocess.run(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-pm', '1'],
            check=False, capture_output=True
        )
        
        # Get max clock speed
        result = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_id), '--query-gpu=clocks.max.graphics', 
             '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True
        )
        max_clock = result.stdout.strip()
        
        # Lock to max clock
        subprocess.run(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc', max_clock],
            check=False, capture_output=True
        )
        
        print_once(f"  ✓ GPU {gpu_id} locked to {max_clock} MHz", rank)
    except Exception as e:
        print_once(f"  ⚠ Warning: Could not lock GPU {gpu_id} clocks: {e}", rank)
    
    print_once("GPU clocks locked\n", rank)


def unlock_gpu_clocks(gpu_id: int, rank: int = 0):
    """Unlock GPU clocks (reset to default)."""
    print_once(f"\nUnlocking GPU {gpu_id} clocks...", rank)
    
    try:
        subprocess.run(
            ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'],
            check=False, capture_output=True
        )
        print_once(f"GPU {gpu_id} clocks unlocked\n", rank)
    except:
        pass


# ============================================================================
# TENSOR CREATION AND MATMUL EXECUTION
# ============================================================================

def create_tensors(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random NON-ZERO tensors for matmul.
    Uses torch.rand() + offset to ensure non-zero values and prevent sparsity optimizations.
    
    Args:
        m: Number of rows in A
        n: Number of columns in B
        k: Inner dimension (columns of A, rows of B)
        dtype: PyTorch data type
        batch_size: Batch size (1 for non-batched 2D tensors)
        device: Device string ('cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of (A, B) tensors ready for matmul
    """
    # Check if dtype is an FP8 type (these need special handling)
    is_fp8 = False
    if hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        is_fp8 = True
    if hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        is_fp8 = True
    
    if is_fp8:
        # For FP8 types, create as float32 first, then convert
        if batch_size == 1:
            A = (torch.rand(m, k, dtype=torch.float32, device=device) + 0.1).to(dtype)
            B = (torch.rand(k, n, dtype=torch.float32, device=device) + 0.1).to(dtype)
        else:
            A = (torch.rand(batch_size, m, k, dtype=torch.float32, device=device) + 0.1).to(dtype)
            B = (torch.rand(batch_size, k, n, dtype=torch.float32, device=device) + 0.1).to(dtype)
    else:
        # For standard floating point types (fp32, fp16, bf16), use rand + offset
        if batch_size == 1:
            A = torch.rand(m, k, dtype=dtype, device=device) + 0.1
            B = torch.rand(k, n, dtype=dtype, device=device) + 0.1
        else:
            A = torch.rand(batch_size, m, k, dtype=dtype, device=device) + 0.1
            B = torch.rand(batch_size, k, n, dtype=dtype, device=device) + 0.1
    
    return A, B


def run_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Execute matmul."""
    return torch.matmul(A, B)


# ============================================================================
# TRACE EXTRACTION - Get actual GPU kernel times
# ============================================================================

def parse_cpu_trace_for_rf_ids(cpu_trace_path: str) -> Dict[str, int]:
    """
    Parse CPU trace to extract rf_ids for configs.
    
    Returns:
        dict: config_name -> rf_id
    """
    print(f"  Parsing CPU trace...")
    
    try:
        with open(cpu_trace_path, 'r') as f:
            cpu_trace = json.load(f)
    except Exception as e:
        print(f"  ⚠ Failed to load CPU trace: {e}")
        return {}
    
    nodes = cpu_trace.get('nodes', [])
    config_rf_ids = {}
    
    # Find all matmul_cfg nodes
    for node in nodes:
        name = node.get('name', '')
        
        if 'matmul_cfg' in name:
            # Extract rf_id from attrs
            attrs = node.get('attrs', [])
            rf_id = None
            for attr in attrs:
                if attr.get('name') == 'rf_id':
                    rf_id = attr.get('value')
                    break
            
            if rf_id is not None:
                config_rf_ids[name] = rf_id
    
    print(f"    Found {len(config_rf_ids)} configs with rf_ids")
    return config_rf_ids


def parse_gpu_traces_for_kernels(gpu_trace_dir: str, config_rf_ids: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    """
    Parse GPU traces to find GEMM kernel events and match them to configs.
    
    Returns:
        dict: config_name -> {'duration_us': float, 'timestamp': float}
    """
    print(f"  Parsing GPU traces...")
    
    if not os.path.exists(gpu_trace_dir):
        print(f"  ⚠ GPU trace directory not found: {gpu_trace_dir}")
        return {}
    
    gpu_trace_files = sorted([f for f in os.listdir(gpu_trace_dir) if f.endswith('.json')])
    print(f"    Found {len(gpu_trace_files)} GPU trace files")
    
    # Build rf_id -> config_name mapping
    rf_id_to_config = {rf_id: config_name for config_name, rf_id in config_rf_ids.items()}
    
    # Track time ranges for each rf_id
    rf_id_time_ranges = {}  # rf_id -> [(start_ts, end_ts), ...]
    
    # Track kernel durations
    config_kernel_times = {}
    
    # Phase 1: Find record_function events and their time ranges
    print(f"    Phase 1: Finding record_function time ranges...")
    for i, filename in enumerate(gpu_trace_files):
        if i % 100 == 0 and i > 0:
            print(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
        filepath = os.path.join(gpu_trace_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                gpu_trace = json.load(f)
            
            events = gpu_trace.get('traceEvents', [])
            
            for event in events:
                args = event.get('args', {})
                rf_id = args.get('Record function id')
                
                if rf_id in rf_id_to_config:
                    start_ts = event.get('ts')
                    duration = event.get('dur', 0)
                    if start_ts is not None:
                        end_ts = start_ts + duration
                        
                        if rf_id not in rf_id_time_ranges:
                            rf_id_time_ranges[rf_id] = []
                        rf_id_time_ranges[rf_id].append((start_ts, end_ts))
        
        except Exception:
            continue
    
    print(f"    Found time ranges for {len(rf_id_time_ranges)} rf_ids")
    
    # Phase 2: Find GEMM kernel events within those time ranges
    print(f"    Phase 2: Finding GEMM kernel events...")
    
    # GEMM kernel name patterns (different for different precisions and GPUs)
    gemm_patterns = ['gemm', 'Gemm', 'GEMM', 'matmul', 'Matmul', 'MATMUL']
    
    for i, filename in enumerate(gpu_trace_files):
        if i % 100 == 0 and i > 0:
            print(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
        filepath = os.path.join(gpu_trace_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                gpu_trace = json.load(f)
            
            events = gpu_trace.get('traceEvents', [])
            
            for event in events:
                name = event.get('name', '')
                cat = event.get('cat', '')
                
                # Look for kernel events (category: 'kernel')
                if cat != 'kernel':
                    continue
                
                # Check if it's a GEMM kernel
                is_gemm = any(pattern in name for pattern in gemm_patterns)
                if not is_gemm:
                    continue
                
                kernel_ts = event.get('ts')
                kernel_dur = event.get('dur')
                
                if kernel_ts is None or kernel_dur is None:
                    continue
                
                # Check if this kernel falls within any rf_id's time range
                for rf_id, time_ranges in rf_id_time_ranges.items():
                    for start_ts, end_ts in time_ranges:
                        if start_ts <= kernel_ts <= end_ts:
                            config_name = rf_id_to_config[rf_id]
                            
                            # Store kernel time for this config
                            if config_name not in config_kernel_times:
                                config_kernel_times[config_name] = []
                            
                            config_kernel_times[config_name].append({
                                'duration_us': kernel_dur,  # Already in microseconds
                                'timestamp': kernel_ts,
                                'kernel_name': name
                            })
                            break
        
        except Exception:
            continue
    
    print(f"    Found kernel times for {len(config_kernel_times)} configs")
    
    # Average multiple kernels per config (if any)
    config_avg_times = {}
    for config_name, kernels in config_kernel_times.items():
        if kernels:
            avg_duration = np.mean([k['duration_us'] for k in kernels])
            config_avg_times[config_name] = {
                'duration_us': avg_duration,
                'timestamp': kernels[0]['timestamp'],
                'num_kernels': len(kernels),
                'kernel_name': kernels[0]['kernel_name']
            }
    
    return config_avg_times


def extract_kernel_times_from_traces(rank: int, output_dir: str) -> Dict[str, Dict[str, float]]:
    """
    Extract kernel times from traces for a specific rank.
    
    ProfilerContext creates:
    - CPU trace: output_dir/rank_X_CPU_trace.json
    - GPU traces: output_dir/rank_X/*.pt.trace.json
    
    Args:
        rank: GPU rank
        output_dir: Base output directory
    
    Returns:
        dict: config_name -> {'duration_us': float, ...}
    """
    print_once("\n" + "=" * 80, rank)
    print_once("EXTRACTING KERNEL TIMES FROM GPU TRACES", rank)
    print_once("=" * 80, rank)
    
    # CPU trace is at output_dir/rank_X_CPU_trace.json
    cpu_trace_path = os.path.join(output_dir, f'rank_{rank}_CPU_trace.json')
    
    # GPU traces are in output_dir/rank_X/
    gpu_trace_dir = os.path.join(output_dir, f'rank_{rank}')
    
    if not os.path.exists(cpu_trace_path):
        print_once(f"  ⚠ CPU trace not found: {cpu_trace_path}", rank)
        print_once(f"  Checking directory contents: {output_dir}", rank)
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print_once(f"  Files found: {files[:10]}", rank)
        return {}
    
    # Parse CPU trace to get rf_ids
    config_rf_ids = parse_cpu_trace_for_rf_ids(cpu_trace_path)
    
    # Parse GPU traces to get kernel times
    kernel_times = parse_gpu_traces_for_kernels(gpu_trace_dir, config_rf_ids)
    
    print_once("\n" + "=" * 80, rank)
    print_once("KERNEL EXTRACTION COMPLETE", rank)
    print_once("=" * 80, rank)
    print_once(f"Extracted kernel times for {len(kernel_times)} configs", rank)
    
    return kernel_times


# ============================================================================
# PROFILING
# ============================================================================

def run_profiling(rank: int, output_dir: str, repeat_count: int) -> List[Dict[str, Any]]:
    """
    Run profiling experiments with PyTorch profiler to capture kernel times.
    
    Args:
        rank: GPU rank (for multi-GPU)
        output_dir: Output directory for traces
        repeat_count: Number of repetitions per config
    
    Returns:
        measurements list
    """
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    
    print_once(f"\nGenerating benchmark configurations...", rank)
    configs = generate_benchmark_configs(verbose=(rank==0))
    print_once(f"✓ Generated {len(configs)} configurations", rank)
    
    # Warmup
    if WARMUP_ITERATIONS > 0:
        print_once(f"\nWarmup ({WARMUP_ITERATIONS} iterations)...", rank)
        for config in configs[:min(len(configs), 20)]:  # Warmup on first 20 configs
            try:
                dtype = dtype_str_to_torch(config['dtype'])
                for _ in range(WARMUP_ITERATIONS):
                    A, B = create_tensors(
                        config['m'], config['n'], config['k'],
                        dtype, config['batch_size'], device
                    )
                    with torch.no_grad():
                        _ = run_matmul(A, B)
            except torch.cuda.OutOfMemoryError as e:
                if rank == 0:
                    print(f"  Warning: Warmup OOM for {config['dtype']} config {config['config_id']}: {config['m']}x{config['n']}x{config['k']} batch={config['batch_size']}")
                    print(f"           Skipping this configuration (GPU memory insufficient)")
                continue
            except Exception as e:
                if rank == 0:
                    print(f"  Warning: Warmup failed for {config['dtype']} config {config['config_id']}: {e}")
                continue
        
        torch.cuda.synchronize()
        barrier_all()
        print_once(f"✓ Warmup complete", rank)
    
    # Profiling with PyTorch profiler
    print_once(f"\nRunning profiling ({repeat_count} repetitions per config)...", rank)
    
    # Configure profiler schedule
    total_steps = len(configs) * repeat_count
    schedule = configure_profiler(
        wait_steps=0,
        warmup_steps=0,
        active_steps=1,
        repeat_count=total_steps
    )
    
    # Profile with CPU and GPU tracing using ProfilerContext (like networking profiler)
    # ProfilerContext will create rank_X_CPU_trace.json and rank_X/ subdirectory
    with ProfilerContext(
        rank=rank,
        output_dir=output_dir,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        
        for i, config in enumerate(configs):
            if i % 10 == 0 and i > 0 and rank == 0:
                print(f"  Progress: {i}/{len(configs)} configs...")
            
            try:
                dtype = dtype_str_to_torch(config['dtype'])
                
                for rep in range(repeat_count):
                    # Create profiler annotation name
                    name = (f"matmul_cfg{config['config_id']:04d}_"
                            f"{config['m']}x{config['n']}x{config['k']}_"
                            f"{config['dtype']}_"
                            f"b{config['batch_size']}_"
                            f"r{rep}")
                    
                    # Synchronize before measurement
                    torch.cuda.synchronize()
                    
                    # Execute with profiler annotation
                    with torch.profiler.record_function(name):
                        A, B = create_tensors(
                            config['m'], config['n'], config['k'],
                            dtype, config['batch_size'], device
                        )
                        with torch.no_grad():
                            _ = run_matmul(A, B)
                    
                    torch.cuda.synchronize()
                    
                    # Step profiler after each measurement
                    prof.step()
            
            except torch.cuda.OutOfMemoryError as e:
                if rank == 0:
                    print(f"  ⚠ OOM: Config {config['config_id']} ({config['m']}x{config['n']}x{config['k']}, "
                          f"{config['dtype']}, batch={config['batch_size']}) - skipping")
                # Try to free memory
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                if rank == 0:
                    print(f"  ⚠ Error: Config {config['config_id']} failed: {type(e).__name__}: {e}")
                continue
    
    print_once(f"✓ Profiling complete (traces generated)", rank)
    
    # Wait for all ranks to finish profiling
    barrier_all()
    
    # Extract kernel times from traces
    kernel_times = extract_kernel_times_from_traces(rank, output_dir)
    
    # Build measurements from kernel times
    measurements = []
    for config in configs:
        for rep in range(repeat_count):
            config_name = (f"matmul_cfg{config['config_id']:04d}_"
                          f"{config['m']}x{config['n']}x{config['k']}_"
                          f"{config['dtype']}_"
                          f"b{config['batch_size']}_"
                          f"r{rep}")
            
            kernel_data = kernel_times.get(config_name)
            
            if kernel_data:
                duration_us = kernel_data['duration_us']
                
                # Calculate throughput (FLOPS / second)
                throughput = config['flops'] / (duration_us * 1e-6)  # Convert us to seconds
                
                measurements.append({
                    'rank': rank,
                    'gpu_id': rank,
                    'config_id': config['config_id'],
                    'm': config['m'],
                    'n': config['n'],
                    'k': config['k'],
                    'dtype': config['dtype'],
                    'batch_size': config['batch_size'],
                    'repetition': rep,
                    'duration_us': duration_us,
                    'flops': config['flops'],
                    'bytes': config['bytes'],
                    'arithmetic_intensity': config['arithmetic_intensity'],
                    'throughput': throughput,
                    'kernel_name': kernel_data.get('kernel_name', 'unknown')
                })
    
    print_once(f"✓ Extracted {len(measurements)} timing measurements", rank)
    
    return measurements


# ============================================================================
# ROOFLINE FITTING
# ============================================================================

def fit_roofline_models(measurements: List[Dict[str, Any]], rank: int = 0) -> Dict[str, Any]:
    """
    Fit roofline models using multiple methods (L2, Huber, IRLS) per (precision, batch_size) and overall per precision.
    
    Args:
        measurements: List of measurement dictionaries
        rank: GPU rank (for printing)
    
    Returns:
        Dictionary with roofline parameters from all 3 methods:
        - per_precision_batch: Models split by (precision, batch_size), each with 3 methods
        - per_precision_overall: Models aggregated by precision only, each with 3 methods
    """
    print_once("\n" + "=" * 80, rank)
    print_once("FITTING ROOFLINE MODELS (3 METHODS: L2, Huber, IRLS)", rank)
    print_once("=" * 80, rank)
    
    # Group measurements by (precision, batch_size)
    by_precision_batch = {}
    by_precision_overall = {}
    
    for m in measurements:
        dtype = m['dtype']
        batch_size = m['batch_size']
        key = (dtype, batch_size)
        
        if key not in by_precision_batch:
            by_precision_batch[key] = []
        by_precision_batch[key].append(m)
        
        if dtype not in by_precision_overall:
            by_precision_overall[dtype] = []
        by_precision_overall[dtype].append(m)
    
    roofline_models = {
        'per_precision_batch': {},
        'per_precision_overall': {}
    }
    
    # Fit models per (precision, batch_size) with ALL methods
    print_once("\n  Fitting per (precision, batch_size)...", rank)
    for (dtype, batch_size) in sorted(by_precision_batch.keys()):
        key_str = f"{dtype}_b{batch_size}"
        print_once(f"\n    {dtype} (batch={batch_size})...", rank)
        
        data = by_precision_batch[(dtype, batch_size)]
        
        # Extract arrays for fitting
        I = np.array([m['arithmetic_intensity'] for m in data])
        T = np.array([m['throughput'] for m in data])
        
        print_once(f"      Samples: {len(I)}, I range: {I.min():.1f}-{I.max():.1f} ops/byte", rank)
        
        # Fit with all 3 methods
        try:
            all_results = fit_roofline_all_methods(I, T, k_fixed=100.0)
            
            # Store results for each method
            roofline_models['per_precision_batch'][key_str] = {
                'dtype': dtype,
                'batch_size': batch_size,
                'num_samples': len(I),
                'methods': {}
            }
            
            # Print comparison
            print_once(f"      {'Method':<8} {'Success':<10} {'BW (GB/s)':<12} {'Peak (TF)':<12} {'R²':<10}", rank)
            print_once(f"      {'-'*60}", rank)
            
            for method_name, result in all_results.items():
                if result['success']:
                    bw_gbps = result['B'] / 1e9
                    peak_tflops = result['P'] / 1e12
                    r2 = result['metrics']['r_squared']
                    
                    print_once(f"      {method_name:<8} {'✓':<10} {bw_gbps:>10.2f}  {peak_tflops:>10.2f}  {r2:>8.6f}", rank)
                    
                    roofline_models['per_precision_batch'][key_str]['methods'][method_name] = {
                        'bandwidth_bytes_per_sec': result['B'],
                        'bandwidth_gb_per_sec': bw_gbps,
                        'peak_flops': result['P'],
                        'peak_tflops': peak_tflops,
                        'k_sharpness': result['k'],
                        'r_squared': r2,
                        'success': True
                    }
                else:
                    error_short = result.get('message', 'unknown')[:30]
                    print_once(f"      {method_name:<8} {'✗':<10} FAILED - {error_short}", rank)
                    
                    roofline_models['per_precision_batch'][key_str]['methods'][method_name] = {
                        'success': False,
                        'error': result.get('message', 'unknown')
                    }
        
        except Exception as e:
            print_once(f"      ⚠ Error: {type(e).__name__}: {e}", rank)
            roofline_models['per_precision_batch'][key_str] = {
                'dtype': dtype,
                'batch_size': batch_size,
                'num_samples': len(I),
                'methods': {},
                'error': str(e)
            }
    
    # Fit overall models per precision with ALL methods
    print_once("\n  Fitting overall per precision...", rank)
    for dtype in sorted(by_precision_overall.keys()):
        print_once(f"\n    {dtype} (all batches)...", rank)
        
        data = by_precision_overall[dtype]
        
        # Extract arrays for fitting
        I = np.array([m['arithmetic_intensity'] for m in data])
        T = np.array([m['throughput'] for m in data])
        
        print_once(f"      Samples: {len(I)}, I range: {I.min():.1f}-{I.max():.1f} ops/byte", rank)
        
        try:
            all_results = fit_roofline_all_methods(I, T, k_fixed=100.0)
            
            roofline_models['per_precision_overall'][dtype] = {
                'num_samples': len(I),
                'methods': {}
            }
            
            print_once(f"      {'Method':<8} {'Success':<10} {'BW (GB/s)':<12} {'Peak (TF)':<12} {'R²':<10}", rank)
            print_once(f"      {'-'*60}", rank)
            
            for method_name, result in all_results.items():
                if result['success']:
                    bw_gbps = result['B'] / 1e9
                    peak_tflops = result['P'] / 1e12
                    r2 = result['metrics']['r_squared']
                    
                    print_once(f"      {method_name:<8} {'✓':<10} {bw_gbps:>10.2f}  {peak_tflops:>10.2f}  {r2:>8.6f}", rank)
                    
                    roofline_models['per_precision_overall'][dtype]['methods'][method_name] = {
                        'bandwidth_bytes_per_sec': result['B'],
                        'bandwidth_gb_per_sec': bw_gbps,
                        'peak_flops': result['P'],
                        'peak_tflops': peak_tflops,
                        'k_sharpness': result['k'],
                        'r_squared': r2,
                        'success': True
                    }
                else:
                    error_short = result.get('message', 'unknown')[:30]
                    print_once(f"      {method_name:<8} {'✗':<10} FAILED - {error_short}", rank)
                    
                    roofline_models['per_precision_overall'][dtype]['methods'][method_name] = {
                        'success': False,
                        'error': result.get('message', 'unknown')
                    }
        
        except Exception as e:
            print_once(f"      ⚠ Error: {type(e).__name__}: {e}", rank)
            roofline_models['per_precision_overall'][dtype] = {
                'num_samples': len(I),
                'methods': {},
                'error': str(e)
            }
    
    print_once("\n" + "=" * 80, rank)
    print_once("ROOFLINE FITTING COMPLETE", rank)
    print_once("=" * 80, rank)
    
    return roofline_models


# ============================================================================
# OUTPUT AND REPORTING
# ============================================================================

def save_results(measurements: List[Dict[str, Any]], roofline_models: Dict[str, Any],
                 gpu_info: Dict[str, Any], output_dir: str, run_config: Dict[str, Any]):
    """Save results to files in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'=' * 80}")
    print(f"SAVING RESULTS")
    print(f"{'=' * 80}")
    
    # 1. Save raw measurements CSV
    csv_path = os.path.join(output_dir, 'measurements.csv')
    with open(csv_path, 'w', newline='') as f:
        if measurements:
            writer = csv.DictWriter(f, fieldnames=measurements[0].keys())
            writer.writeheader()
            writer.writerows(measurements)
    print(f"✓ Saved measurements CSV: {csv_path}")
    
    # 2. Save roofline parameters CSV (overall models per precision, aggregated across GPUs) - MULTI-METHOD
    roofline_path = os.path.join(output_dir, 'roofline_parameters.csv')
    with open(roofline_path, 'w', newline='') as f:
        fieldnames = ['precision', 'method', 'bandwidth_gb_per_sec', 'peak_tflops', 'r_squared', 'num_samples', 'success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write overall models per precision (all 3 methods) - aggregated across all GPUs
        overall_models = roofline_models.get('overall', roofline_models)
        for dtype in sorted(overall_models['per_precision_overall'].keys()):
            model_group = overall_models['per_precision_overall'][dtype]
            for method_name in ['l2', 'huber', 'irls']:
                if method_name in model_group.get('methods', {}):
                    method_result = model_group['methods'][method_name]
                    if method_result.get('success'):
                        writer.writerow({
                            'precision': dtype,
                            'method': method_name,
                            'bandwidth_gb_per_sec': method_result['bandwidth_gb_per_sec'],
                            'peak_tflops': method_result['peak_tflops'],
                            'r_squared': method_result['r_squared'],
                            'num_samples': model_group['num_samples'],
                            'success': True
                        })
                    else:
                        writer.writerow({
                            'precision': dtype,
                            'method': method_name,
                            'bandwidth_gb_per_sec': 'N/A',
                            'peak_tflops': 'N/A',
                            'r_squared': 'N/A',
                            'num_samples': model_group['num_samples'],
                            'success': False
                        })
    print(f"✓ Saved aggregated roofline parameters CSV: {roofline_path}")
    
    # 2b. Save detailed roofline parameters CSV (per precision and batch size, aggregated across GPUs) - MULTI-METHOD
    roofline_detailed_path = os.path.join(output_dir, 'roofline_parameters_detailed.csv')
    with open(roofline_detailed_path, 'w', newline='') as f:
        fieldnames = ['precision', 'batch_size', 'method', 'bandwidth_gb_per_sec', 'peak_tflops', 'r_squared', 'num_samples', 'success']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write per (precision, batch_size) models (all 3 methods) - aggregated across all GPUs
        overall_models = roofline_models.get('overall', roofline_models)
        for key in sorted(overall_models['per_precision_batch'].keys()):
            model_group = overall_models['per_precision_batch'][key]
            dtype = model_group.get('dtype', key.split('_')[0])
            batch_size = model_group.get('batch_size', 'N/A')
            
            for method_name in ['l2', 'huber', 'irls']:
                if method_name in model_group.get('methods', {}):
                    method_result = model_group['methods'][method_name]
                    if method_result.get('success'):
                        writer.writerow({
                            'precision': dtype,
                            'batch_size': batch_size,
                            'method': method_name,
                            'bandwidth_gb_per_sec': method_result['bandwidth_gb_per_sec'],
                            'peak_tflops': method_result['peak_tflops'],
                            'r_squared': method_result['r_squared'],
                            'num_samples': model_group['num_samples'],
                            'success': True
                        })
                    else:
                        writer.writerow({
                            'precision': dtype,
                            'batch_size': batch_size,
                            'method': method_name,
                            'bandwidth_gb_per_sec': 'N/A',
                            'peak_tflops': 'N/A',
                            'r_squared': 'N/A',
                            'num_samples': model_group['num_samples'],
                            'success': False
                        })
    print(f"✓ Saved aggregated detailed roofline parameters CSV: {roofline_detailed_path}")
    
    # 2c. Save per-GPU roofline parameters CSV files
    if 'per_gpu' in roofline_models:
        for gpu_id in sorted(roofline_models['per_gpu'].keys()):
            gpu_models = roofline_models['per_gpu'][gpu_id]
            
            # Per-GPU overall roofline parameters
            gpu_roofline_path = os.path.join(output_dir, f'roofline_parameters_gpu{gpu_id}.csv')
            with open(gpu_roofline_path, 'w', newline='') as f:
                fieldnames = ['gpu_id', 'precision', 'method', 'bandwidth_gb_per_sec', 'peak_tflops', 'r_squared', 'num_samples', 'success']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for dtype in sorted(gpu_models['per_precision_overall'].keys()):
                    model_group = gpu_models['per_precision_overall'][dtype]
                    for method_name in ['l2', 'huber', 'irls']:
                        if method_name in model_group.get('methods', {}):
                            method_result = model_group['methods'][method_name]
                            if method_result.get('success'):
                                writer.writerow({
                                    'gpu_id': gpu_id,
                                    'precision': dtype,
                                    'method': method_name,
                                    'bandwidth_gb_per_sec': method_result['bandwidth_gb_per_sec'],
                                    'peak_tflops': method_result['peak_tflops'],
                                    'r_squared': method_result['r_squared'],
                                    'num_samples': model_group['num_samples'],
                                    'success': True
                                })
                            else:
                                writer.writerow({
                                    'gpu_id': gpu_id,
                                    'precision': dtype,
                                    'method': method_name,
                                    'bandwidth_gb_per_sec': 'N/A',
                                    'peak_tflops': 'N/A',
                                    'r_squared': 'N/A',
                                    'num_samples': model_group['num_samples'],
                                    'success': False
                                })
            print(f"✓ Saved GPU {gpu_id} roofline parameters: {gpu_roofline_path}")
            
            # Per-GPU detailed roofline parameters
            gpu_detailed_path = os.path.join(output_dir, f'roofline_parameters_gpu{gpu_id}_detailed.csv')
            with open(gpu_detailed_path, 'w', newline='') as f:
                fieldnames = ['gpu_id', 'precision', 'batch_size', 'method', 'bandwidth_gb_per_sec', 'peak_tflops', 'r_squared', 'num_samples', 'success']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for key in sorted(gpu_models['per_precision_batch'].keys()):
                    model_group = gpu_models['per_precision_batch'][key]
                    dtype = model_group.get('dtype', key.split('_')[0])
                    batch_size = model_group.get('batch_size', 'N/A')
                    
                    for method_name in ['l2', 'huber', 'irls']:
                        if method_name in model_group.get('methods', {}):
                            method_result = model_group['methods'][method_name]
                            if method_result.get('success'):
                                writer.writerow({
                                    'gpu_id': gpu_id,
                                    'precision': dtype,
                                    'batch_size': batch_size,
                                    'method': method_name,
                                    'bandwidth_gb_per_sec': method_result['bandwidth_gb_per_sec'],
                                    'peak_tflops': method_result['peak_tflops'],
                                    'r_squared': method_result['r_squared'],
                                    'num_samples': model_group['num_samples'],
                                    'success': True
                                })
                            else:
                                writer.writerow({
                                    'gpu_id': gpu_id,
                                    'precision': dtype,
                                    'batch_size': batch_size,
                                    'method': method_name,
                                    'bandwidth_gb_per_sec': 'N/A',
                                    'peak_tflops': 'N/A',
                                    'r_squared': 'N/A',
                                    'num_samples': model_group['num_samples'],
                                    'success': False
                                })
            print(f"✓ Saved GPU {gpu_id} detailed roofline parameters: {gpu_detailed_path}")
    
    # 3. Save GPU info
    gpu_info_path = os.path.join(output_dir, 'gpu_info.txt')
    with open(gpu_info_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        for key, value in gpu_info.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Saved GPU info: {gpu_info_path}")
    
    # 4. Save JSON with all data
    json_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'gpu_info': gpu_info,
            'run_config': run_config,
        },
        'roofline_models': roofline_models,
        'raw_measurements': measurements,
    }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON data: {json_path}")
    
    # 5. Save comprehensive summary report
    summary_path = os.path.join(output_dir, 'RESULTS_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPUTE ROOFLINE PROFILER - COMPREHENSIVE RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("GPU INFORMATION\n")
        f.write("-" * 80 + "\n")
        for key, value in gpu_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("RUN CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        for key, value in run_config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("OVERALL ROOFLINE MODELS (AGGREGATED ACROSS ALL GPUs) - ALL 3 METHODS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{'Precision':<12} {'Method':<8} {'Bandwidth':<16} {'Peak FLOPS':<16} {'R²':<10} {'Status':<8}\n")
        f.write("-" * 80 + "\n")
        
        overall_models = roofline_models.get('overall', roofline_models)
        for dtype in sorted(overall_models['per_precision_overall'].keys()):
            model_group = overall_models['per_precision_overall'][dtype]
            for method_name in ['l2', 'huber', 'irls']:
                if method_name in model_group.get('methods', {}):
                    method_result = model_group['methods'][method_name]
                    if method_result.get('success'):
                        f.write(f"{dtype:<12} {method_name:<8} {method_result['bandwidth_gb_per_sec']:>12.2f} GB/s  "
                               f"{method_result['peak_tflops']:>12.2f} TFLOPS  "
                               f"{method_result['r_squared']:>8.6f}  {'OK':<8}\n")
                    else:
                        f.write(f"{dtype:<12} {method_name:<8} {'N/A':<16} {'N/A':<16} {'N/A':<10} {'FAILED':<8}\n")
            f.write("\n")  # Spacing between dtypes
        
        # Add per-GPU roofline models summary
        if 'per_gpu' in roofline_models:
            f.write("-" * 80 + "\n")
            f.write("PER-GPU ROOFLINE MODELS (OVERALL BY PRECISION)\n")
            f.write("-" * 80 + "\n\n")
            
            for gpu_id in sorted(roofline_models['per_gpu'].keys()):
                f.write(f"GPU {gpu_id}:\n")
                f.write(f"{'  Precision':<12} {'Method':<8} {'Bandwidth':<16} {'Peak FLOPS':<16} {'R²':<10} {'Status':<8}\n")
                f.write("  " + "-" * 78 + "\n")
                
                gpu_models = roofline_models['per_gpu'][gpu_id]
                for dtype in sorted(gpu_models['per_precision_overall'].keys()):
                    model_group = gpu_models['per_precision_overall'][dtype]
                    for method_name in ['l2', 'huber', 'irls']:
                        if method_name in model_group.get('methods', {}):
                            method_result = model_group['methods'][method_name]
                            if method_result.get('success'):
                                f.write(f"  {dtype:<10} {method_name:<8} {method_result['bandwidth_gb_per_sec']:>12.2f} GB/s  "
                                       f"{method_result['peak_tflops']:>12.2f} TFLOPS  "
                                       f"{method_result['r_squared']:>8.6f}  {'OK':<8}\n")
                            else:
                                f.write(f"  {dtype:<10} {method_name:<8} {'N/A':<16} {'N/A':<16} {'N/A':<10} {'FAILED':<8}\n")
                    f.write("\n")
                f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("DETAILED ROOFLINE MODELS (AGGREGATED, BY PRECISION AND BATCH SIZE) - ALL 3 METHODS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write(f"{'Precision':<10} {'Batch':<8} {'Method':<8} {'Bandwidth':<14} {'Peak FLOPS':<14} {'R²':<10} {'Status':<8}\n")
        f.write("-" * 80 + "\n")
        
        for key in sorted(overall_models['per_precision_batch'].keys()):
            model_group = overall_models['per_precision_batch'][key]
            dtype = model_group.get('dtype', key.split('_')[0])
            batch = model_group.get('batch_size', 'N/A')
            
            for method_name in ['l2', 'huber', 'irls']:
                if method_name in model_group.get('methods', {}):
                    method_result = model_group['methods'][method_name]
                    if method_result.get('success'):
                        f.write(f"{dtype:<10} {str(batch):<8} {method_name:<8} {method_result['bandwidth_gb_per_sec']:>10.2f} GB/s  "
                               f"{method_result['peak_tflops']:>10.2f} TFLOPS  "
                               f"{method_result['r_squared']:>8.6f}  {'OK':<8}\n")
                    else:
                        f.write(f"{dtype:<10} {str(batch):<8} {method_name:<8} {'N/A':<14} {'N/A':<14} {'N/A':<10} {'FAILED':<8}\n")
            f.write("\n")  # Spacing between batch sizes
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SAMPLE MEASUREMENTS (first 10)\n")
        f.write("-" * 80 + "\n\n")
        
        for m in measurements[:10]:
            f.write(f"Config {m['config_id']:04d}: {m['m']}×{m['n']}×{m['k']} "
                   f"({m['dtype']}) - {m['duration_us']:.2f} μs, "
                   f"I={m['arithmetic_intensity']:.1f} ops/byte\n")
        
        if len(measurements) > 10:
            f.write(f"\n... and {len(measurements) - 10} more measurements\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved comprehensive summary: {summary_path}")
    
    print(f"\n{'=' * 80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 80}")


def cleanup_traces(output_dir: str):
    """
    Clean up trace files after extraction.
    """
    print(f"\n{'=' * 80}")
    print("CLEANUP: REMOVING TRACE FILES")
    print(f"{'=' * 80}")
    
    cleaned_files = []
    
    # Remove CPU trace files
    cpu_traces = glob.glob(os.path.join(output_dir, '*CPU_trace.json'))
    for f in cpu_traces:
        try:
            os.remove(f)
            cleaned_files.append(os.path.basename(f))
        except Exception as e:
            print(f"⚠ Warning: Could not remove {f}: {e}")
    
    # Remove GPU trace JSON files (keep only our generated outputs)
    trace_patterns = ['*.pt.trace.json']
    for pattern in trace_patterns:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                os.remove(f)
                cleaned_files.append(os.path.basename(f))
            except Exception as e:
                print(f"⚠ Warning: Could not remove {f}: {e}")
    
    if cleaned_files:
        print(f"✓ Cleaned up {len(cleaned_files)} trace file(s)")
    else:
        print("✓ No trace files to clean up")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function with multi-GPU support."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Compute Roofline Profiler - Multi-GPU Version',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--no-lock-clocks', action='store_true',
                        help='Disable GPU clock locking (default: enabled, requires sudo)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./compute_roofline_results_YYYYMMDD_HHMMSS)')
    parser.add_argument('--repeat-count', type=int, default=REPEAT_COUNT,
                        help=f'Number of repetitions per config (default: {REPEAT_COUNT})')
    args = parser.parse_args()
    
    lock_clocks = not args.no_lock_clocks
    
    # Initialize distributed (if using torchrun) or single GPU
    if 'RANK' in os.environ:
        # Multi-GPU with torchrun
        rank, world_size = init_distributed(backend='nccl')
        gpu_id = rank
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        gpu_id = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
    
    print_once("\n" + "=" * 80, rank)
    print_once("COMPUTE ROOFLINE PROFILER - MULTI-GPU", rank)
    print_once("=" * 80, rank)
    print_once(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", rank)
    print_once(f"GPUs: {world_size}", rank)
    print_once(f"Lock clocks: {'YES' if lock_clocks else 'NO'}", rank)
    print_once(f"Repeat count: {args.repeat_count}", rank)
    
    # Get GPU information for this rank
    gpu_info = get_gpu_info(gpu_id)
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"GPU {gpu_id} INFORMATION")
        print("=" * 80)
        for key, value in gpu_info.items():
            print(f"  {key}: {value}")
    
    # Store run configuration
    run_config = {
        'lock_clocks': lock_clocks,
        'repeat_count': args.repeat_count,
        'warmup_iterations': WARMUP_ITERATIONS,
        'world_size': world_size,
    }
    
    # Lock GPU clocks if enabled
    if lock_clocks and torch.cuda.is_available():
        lock_gpu_clocks(gpu_id, rank)
    
    barrier_all()
    
    # Setup output directory (rank 0 creates it)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'./compute_roofline_results_{timestamp}'
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    barrier_all()
    
    # Run profiling (each rank profiles its own GPU)
    print_once("\n" + "=" * 80, rank)
    print_once("RUNNING PROFILING", rank)
    print_once("=" * 80, rank)
    measurements = run_profiling(rank, output_dir, args.repeat_count)
    
    # Gather measurements from all ranks to rank 0
    all_measurements = None
    if world_size > 1:
        # Use simple file-based gathering
        rank_measurements_file = os.path.join(output_dir, f'measurements_rank_{rank}.json')
        with open(rank_measurements_file, 'w') as f:
            json.dump(measurements, f)
        
        barrier_all()
        
        if rank == 0:
            all_measurements = []
            for r in range(world_size):
                rank_file = os.path.join(output_dir, f'measurements_rank_{r}.json')
                if os.path.exists(rank_file):
                    with open(rank_file, 'r') as f:
                        all_measurements.extend(json.load(f))
                    # Cleanup temp file
                    os.remove(rank_file)
    else:
        if rank == 0:
            all_measurements = measurements
    
    # All ranks save their GPU info for gathering
    gpu_info_file = os.path.join(output_dir, f'gpu_info_rank_{rank}.json')
    with open(gpu_info_file, 'w') as f:
        json.dump(gpu_info, f)
    
    barrier_all()
    
    # Only rank 0 fits models and saves results
    if rank == 0:
        if not all_measurements:
            print("\n⚠ ERROR: No measurements extracted from traces!")
            return
        
        print(f"\nTotal measurements from all GPUs: {len(all_measurements)}")
        
        # Fit roofline models on aggregated data
        roofline_models_overall = fit_roofline_models(all_measurements, rank=0)
        
        # Fit per-GPU roofline models
        print("\n" + "=" * 80)
        print("FITTING PER-GPU ROOFLINE MODELS")
        print("=" * 80)
        
        roofline_models_per_gpu = {}
        for r in range(world_size):
            gpu_measurements = [m for m in all_measurements if m['rank'] == r]
            if gpu_measurements:
                print(f"\nGPU {r}: {len(gpu_measurements)} measurements")
                roofline_models_per_gpu[r] = fit_roofline_models(gpu_measurements, rank=0)
        
        # Combine into single structure
        roofline_models = {
            'overall': roofline_models_overall,
            'per_gpu': roofline_models_per_gpu
        }
        
        # Gather GPU info from all ranks
        all_gpu_info = {'gpus': {}}
        for r in range(world_size):
            gpu_info_file = os.path.join(output_dir, f'gpu_info_rank_{r}.json')
            if os.path.exists(gpu_info_file):
                with open(gpu_info_file, 'r') as f:
                    all_gpu_info['gpus'][r] = json.load(f)
                os.remove(gpu_info_file)
        
        # Save results
        save_results(all_measurements, roofline_models, all_gpu_info, output_dir, run_config)
        
        # Cleanup traces
        cleanup_traces(output_dir)
    
    barrier_all()
    
    # Unlock GPU clocks if we locked them
    if lock_clocks and torch.cuda.is_available():
        unlock_gpu_clocks(gpu_id, rank)
    
    barrier_all()
    
    # Cleanup distributed
    if world_size > 1:
        cleanup_distributed()
    
    print_once("\n" + "=" * 80, rank)
    print_once("PROFILING COMPLETE", rank)
    print_once("=" * 80, rank)
    if rank == 0:
        print(f"\nResults saved to: {output_dir}")
        print("\nKey files:")
        print(f"  - RESULTS_SUMMARY.txt                      (human-readable report with per-GPU results)")
        print(f"  - results.json                             (complete structured data)")
        print(f"  - roofline_parameters.csv                  (aggregated roofline params per precision)")
        print(f"  - roofline_parameters_detailed.csv         (aggregated roofline params by precision & batch)")
        print(f"  - roofline_parameters_gpu*.csv             (per-GPU roofline params)")
        print(f"  - roofline_parameters_gpu*_detailed.csv    (per-GPU detailed roofline params)")
        print(f"  - measurements.csv                         (raw timing data with GPU IDs)")


if __name__ == "__main__":
    main()

