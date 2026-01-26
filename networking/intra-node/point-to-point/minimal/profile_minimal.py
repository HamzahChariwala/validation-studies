#!/usr/bin/env python3
"""
Minimal Point-to-Point GPU Communication Profiler

Production-ready GPU communication benchmarking tool with triple regression analysis
that extracts actual GPU kernel execution times (not CPU dispatch times).

FEATURES:
- Generates PyTorch profiler traces
- Extracts ncclDevKernel_SendRecv kernel durations from GPU traces
- Maps kernels to configs using rf_ids
- Triple regression analysis (Linear, Huber, IRLS) on actual kernel times
- GPU clock locking (default: ON, requires sudo)
- Strict serialization (default: ON, maximally isolated measurements)
- 10 message sizes (1 KB to 1 GB), 5 repetitions per config
- Comprehensive output (JSON, CSV, human-readable summary)
- Automatic trace cleanup after extraction

USAGE:
    # Default (recommended - maximum reliability)
    torchrun --nproc_per_node=4 profile_minimal.py
    
    # Fast mode (no sudo, less isolated)
    torchrun --nproc_per_node=4 profile_minimal.py --no-lock-clocks --no-strict-serial
    
    # Custom output directory
    torchrun --nproc_per_node=4 profile_minimal.py --output-dir /data/results
    
    # 8 GPUs
    torchrun --nproc_per_node=8 profile_minimal.py

DEFAULTS:
    - Clock locking: ENABLED (use --no-lock-clocks to disable)
    - Strict serial: ENABLED (use --no-strict-serial to disable)
    - Message sizes: 10 (evenly spaced from 1 KB to 1 GB)
    - Repetitions: 5 per config
    - Output: ./p2p_results_TIMESTAMP/

REQUIREMENTS:
    - PyTorch with CUDA support
    - numpy, scipy, scikit-learn
    - Run setup.sh at repo root to install dependencies

OUTPUT FILES:
    - RESULTS_SUMMARY.txt      (comprehensive human-readable report)
    - results.json             (structured data with all 3 regressions)
    - regression_results.csv   (all metrics per GPU pair)
    - measurements.csv         (raw kernel timing data)
    - topology.txt             (GPU topology information)
"""

import torch
import torch.distributed as dist
import time
import os
import sys
import csv
import subprocess
import re
import argparse
import json
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.linear_model import HuberRegressor
from scipy import stats

# Add repo root to path to import networking.utils
# This file is at: networking/intra-node/point-to-point/minimal/profile_minimal.py
# We need to add: /path/to/repo (4 levels up from minimal/)
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
sys.path.insert(0, repo_root)

from networking.utils import (
    init_distributed,
    cleanup_distributed,
    barrier_all,
    print_once,
    ProfilerContext,
    configure_profiler,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Message sizes (10 evenly-spaced for efficient profiling)
MIN_SIZE = 1024  # 1 KB
MAX_SIZE = 1073741824  # 1 GB
NUM_SIZES = 10
MESSAGE_SIZES = np.linspace(MIN_SIZE, MAX_SIZE, NUM_SIZES, dtype=np.int64).tolist()

# Profiling parameters
WARMUP_ITERATIONS = 3
REPEAT_COUNT = 5  # 5 repetitions per config


# ============================================================================
# TOPOLOGY DETECTION
# ============================================================================

def get_pcie_info_per_gpu(num_gpus: int) -> List[Dict[str, Any]]:
    """
    Get PCIe information for each GPU from nvidia-smi.
    Reads actual values from 'GPU Link Info' section.
    
    Returns:
        List of dicts with PCIe info per GPU
    """
    pcie_info = []
    
    for gpu_id in range(num_gpus):
        info = {
            'gpu_id': gpu_id,
            'pcie_gen_max': 'Unknown',
            'pcie_gen_current': 'Unknown',
            'link_width_max': 'Unknown',
            'link_width_current': 'Unknown',
        }
        
        try:
            # Query nvidia-smi for PCIe information
            result = subprocess.run(
                ['nvidia-smi', '-i', str(gpu_id), '-q'],
                capture_output=True,
                text=True,
                check=True
            )
            
            lines = result.stdout.split('\n')
            in_gpu_link_info = False
            in_pcie_generation = False
            in_link_width = False
            
            for line in lines:
                stripped = line.strip()
                
                # Track sections
                if 'GPU Link Info' in line:
                    in_gpu_link_info = True
                    continue
                
                if in_gpu_link_info:
                    # Exit GPU Link Info section if we hit another major section
                    if line and not line.startswith(' ') and ':' not in stripped:
                        in_gpu_link_info = False
                        continue
                    
                    # PCIe Generation subsection
                    if 'PCIe Generation' in stripped:
                        in_pcie_generation = True
                        in_link_width = False
Hamzahitu                        continue
                    
                    # Link Width subsection
                    if 'Link Width' in stripped and 'PCIe' not in stripped:
                        in_link_width = True
                        in_pcie_generation = False
                        continue
                    
                    # Parse PCIe Generation values
                    if in_pcie_generation:
                        if 'Max' in stripped and ':' in stripped:
                            match = re.search(r':\s*(\d+)', stripped)
                            if match:
                                info['pcie_gen_max'] = f"Gen{match.group(1)}"
                        elif 'Current' in stripped and 'Device' not in stripped and 'Host' not in stripped and ':' in stripped:
                            match = re.search(r':\s*(\d+)', stripped)
                            if match:
                                info['pcie_gen_current'] = f"Gen{match.group(1)}"
                    
                    # Parse Link Width values
                    if in_link_width:
                        if 'Max' in stripped and ':' in stripped:
                            match = re.search(r':\s*(\d+x)', stripped)
                            if match:
                                info['link_width_max'] = match.group(1)
                        elif 'Current' in stripped and ':' in stripped:
                            match = re.search(r':\s*(\d+x)', stripped)
                            if match:
                                info['link_width_current'] = match.group(1)
        
        except Exception as e:
            pass  # Keep default 'Unknown' values
        
        pcie_info.append(info)
    
    return pcie_info


def get_topology_info() -> Dict[str, Any]:
    """
    Detect GPU topology and interconnect information.
    
    Returns:
        Dictionary with topology information
    """
    print("=" * 80)
    print("DETECTING GPU TOPOLOGY")
    print("=" * 80)
    
    topology = {
        'num_gpus': 0,
        'interconnect_type': 'Unknown',
        'nvlink_lanes': 0,
        'pcie_info': [],
        'topo_matrix': '',
    }
    
    try:
        # Get number of GPUs
        topology['num_gpus'] = torch.cuda.device_count()
        print(f"✓ Detected {topology['num_gpus']} GPU(s)")
        
        # Get PCIe information per GPU
        pcie_info = get_pcie_info_per_gpu(topology['num_gpus'])
        topology['pcie_info'] = pcie_info
        
        print(f"\n{'GPU':<6} {'PCIe Gen (Max)':<16} {'PCIe Gen (Cur)':<16} {'Width (Max)':<13} {'Width (Cur)':<13}")
        print("-" * 80)
        for info in pcie_info:
            print(f"GPU {info['gpu_id']:<3} {info['pcie_gen_max']:<16} {info['pcie_gen_current']:<16} "
                  f"{info['link_width_max']:<13} {info['link_width_current']:<13}")
        
        # Get topology matrix
        result = subprocess.run(
            ['nvidia-smi', 'topo', '-m'],
            capture_output=True,
            text=True,
            check=True
        )
        topology['topo_matrix'] = result.stdout
        print(f"\nTopology Matrix:")
        print(topology['topo_matrix'])
        
        # Parse interconnect type from topology matrix
        # Extract only the actual matrix rows (lines starting with GPU)
        matrix_lines = [line for line in topology['topo_matrix'].split('\n') 
                       if line.strip().startswith('GPU') and '\t' in line]
        matrix_data = '\n'.join(matrix_lines)
        
        # Look for NVLink connections (NV followed by digit) or PCIe connections
        nvlink_matches = re.findall(r'\bNV(\d+)\b', matrix_data)
        if nvlink_matches:
            # Found actual NVLink connections (NV2, NV4, NV6, NV12, etc.)
            max_nvlink = max(int(x) for x in nvlink_matches)
            topology['interconnect_type'] = f'NVLink'
            topology['nvlink_lanes'] = max_nvlink
            print(f"\n✓ Interconnect: NVLink (up to {max_nvlink} lanes)")
        elif 'PIX' in matrix_data:
            topology['interconnect_type'] = 'PCIe (single bridge)'
            print(f"\n✓ Interconnect: PCIe (single bridge)")
        elif 'PXB' in matrix_data:
            topology['interconnect_type'] = 'PCIe (multiple bridges)'
            print(f"\n✓ Interconnect: PCIe (multiple bridges)")
        elif 'PHB' in matrix_data:
            topology['interconnect_type'] = 'PCIe (host bridge)'
            print(f"\n✓ Interconnect: PCIe (host bridge)")
        elif 'NODE' in matrix_data:
            topology['interconnect_type'] = 'PCIe (NUMA node traversal)'
            print(f"\n✓ Interconnect: PCIe (NUMA node traversal)")
        elif 'SYS' in matrix_data:
            topology['interconnect_type'] = 'PCIe (cross-NUMA)'
            print(f"\n✓ Interconnect: PCIe (cross-NUMA)")
        else:
            topology['interconnect_type'] = 'PCIe'
            print(f"\n✓ Interconnect: PCIe")
        
        # Try to get NVLink status details
        try:
            result = subprocess.run(
                ['nvidia-smi', 'nvlink', '--status'],
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                print(f"\nNVLink Status:")
                # Parse and summarize
                lines = result.stdout.split('\n')[:20]  # First 20 lines
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
    except Exception as e:
        print(f"⚠ Warning: Could not fully detect topology: {e}")
    
    print("=" * 80)
    return topology


# ============================================================================
# GPU CLOCK LOCKING
# ============================================================================

def lock_gpu_clocks(world_size: int):
    """Lock GPU clocks to maximum frequency (requires sudo)."""
    print_once("\nLocking GPU clocks to maximum frequency...")
    print_once("  (This requires sudo access)")
    
    for gpu_id in range(world_size):
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
            
            print_once(f"  ✓ GPU {gpu_id} locked to {max_clock} MHz")
            
        except Exception as e:
            print_once(f"  ⚠ Warning: Could not lock GPU {gpu_id} clocks: {e}")
    
    print_once("GPU clocks locked\n")


def unlock_gpu_clocks(world_size: int):
    """Unlock GPU clocks (reset to default)."""
    print_once("\nUnlocking GPU clocks...")
    
    for gpu_id in range(world_size):
        try:
            subprocess.run(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'],
                check=False, capture_output=True
            )
        except:
            pass
    
    print_once("GPU clocks unlocked\n")


# ============================================================================
# CONFIGURATION GENERATION
# ============================================================================

def generate_configs(world_size: int, message_sizes: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all point-to-point configurations.
    
    Args:
        world_size: Number of GPUs
        message_sizes: List of message sizes in bytes
    
    Returns:
        List of config dictionaries
    """
    configs = []
    config_id = 0
    
    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                continue
            
            for size in message_sizes:
                configs.append({
                    'config_id': config_id,
                    'src': src,
                    'dst': dst,
                    'size': size,
                })
                config_id += 1
    
    return configs


# ============================================================================
# COMMUNICATION PRIMITIVES
# ============================================================================

def create_buffers(message_sizes: List[int], device: str) -> Dict[int, torch.Tensor]:
    """
    Create communication buffers.
    
    Args:
        message_sizes: List of message sizes in bytes
        device: Device to allocate on
    
    Returns:
        Dictionary mapping size -> tensor
    """
    buffers = {}
    dtype = torch.float32
    bytes_per_element = 4  # float32
    
    # Allocate largest buffer
    max_size = max(message_sizes)
    max_elements = max_size // bytes_per_element
    max_buffer = torch.randn(max_elements, dtype=dtype, device=device)
    
    # Use slices for smaller sizes
    for size in message_sizes:
        num_elements = max(1, size // bytes_per_element)
        buffers[size] = max_buffer[:num_elements].contiguous()
    
    return buffers


def run_p2p(rank: int, src: int, dst: int, buffer: torch.Tensor):
    """Execute a single point-to-point communication."""
    if rank == src:
        dist.send(buffer, dst=dst)
    elif rank == dst:
        dist.recv(buffer, src=src)


# ============================================================================
# TRACE EXTRACTION - Get actual GPU kernel times
# ============================================================================

def parse_cpu_trace_for_rf_ids(cpu_trace_path: str, rank: int) -> Dict[str, int]:
    """
    Parse CPU trace to extract rf_ids for configs.
    
    Returns:
        dict: config_name -> rf_id
    """
    print_once(f"  Parsing CPU trace for rank {rank}...")
    
    try:
        with open(cpu_trace_path, 'r') as f:
            cpu_trace = json.load(f)
    except Exception as e:
        print_once(f"  ⚠ Failed to load CPU trace: {e}")
        return {}
    
    nodes = cpu_trace.get('nodes', [])
    config_rf_ids = {}
    
    # Find all p2p_cfg nodes
    for node in nodes:
        name = node.get('name', '')
        
        if 'p2p_cfg' in name:
            # Extract rf_id from attrs
            attrs = node.get('attrs', [])
            rf_id = None
            for attr in attrs:
                if attr.get('name') == 'rf_id':
                    rf_id = attr.get('value')
                    break
            
            if rf_id is not None:
                # Store config name with rf_id
                config_rf_ids[name] = rf_id
    
    print_once(f"    Found {len(config_rf_ids)} configs with rf_ids")
    return config_rf_ids


def parse_gpu_traces_for_kernels(gpu_trace_dir: str, config_rf_ids: Dict[str, int], rank: int) -> Dict[str, Dict[str, float]]:
    """
    Parse GPU traces to find ncclDevKernel_SendRecv events and match them to configs.
    
    Returns:
        dict: config_name -> {'duration_us': float, 'timestamp': float}
    """
    print_once(f"  Parsing GPU traces for rank {rank}...")
    
    if not os.path.exists(gpu_trace_dir):
        print_once(f"  ⚠ GPU trace directory not found: {gpu_trace_dir}")
        return {}
    
    gpu_trace_files = sorted([f for f in os.listdir(gpu_trace_dir) if f.endswith('.json')])
    print_once(f"    Found {len(gpu_trace_files)} GPU trace files")
    
    # Build rf_id -> config_name mapping
    rf_id_to_config = {rf_id: config_name for config_name, rf_id in config_rf_ids.items()}
    
    # Track time ranges for each rf_id
    rf_id_time_ranges = {}  # rf_id -> (start_ts, end_ts)
    
    # Track kernel durations
    config_kernel_times = {}
    
    # Phase 1: Find record_function events and their time ranges
    print_once(f"    Phase 1: Finding record_function time ranges...")
    for i, filename in enumerate(gpu_trace_files):
        if i % 100 == 0 and i > 0:
            print_once(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
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
                        
                        # Store the time range for this rf_id
                        if rf_id not in rf_id_time_ranges:
                            rf_id_time_ranges[rf_id] = []
                        rf_id_time_ranges[rf_id].append((start_ts, end_ts))
        
        except Exception as e:
            continue
    
    print_once(f"    Found time ranges for {len(rf_id_time_ranges)} rf_ids")
    
    # Phase 2: Find ncclDevKernel_SendRecv events within those time ranges
    print_once(f"    Phase 2: Finding ncclDevKernel_SendRecv events...")
    
    for i, filename in enumerate(gpu_trace_files):
        if i % 100 == 0 and i > 0:
            print_once(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
        filepath = os.path.join(gpu_trace_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                gpu_trace = json.load(f)
            
            events = gpu_trace.get('traceEvents', [])
            
            for event in events:
                name = event.get('name', '')
                
                if 'ncclDevKernel_SendRecv' in name or 'ncclKernel_SendRecv' in name:
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
                                    'timestamp': kernel_ts
                                })
                                break
        
        except Exception as e:
            continue
    
    print_once(f"    Found kernel times for {len(config_kernel_times)} configs")
    
    # Average multiple kernels per config (if any)
    config_avg_times = {}
    for config_name, kernels in config_kernel_times.items():
        if kernels:
            avg_duration = np.mean([k['duration_us'] for k in kernels])
            config_avg_times[config_name] = {
                'duration_us': avg_duration,
                'timestamp': kernels[0]['timestamp'],  # Use first timestamp
                'num_kernels': len(kernels)
            }
    
    return config_avg_times


def extract_kernel_times_from_traces(output_dir: str, world_size: int) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Extract kernel times from all ranks' traces.
    
    Returns:
        dict: rank -> (config_name -> {'duration_us': float, ...})
    """
    print_once("\n" + "=" * 80)
    print_once("EXTRACTING KERNEL TIMES FROM GPU TRACES")
    print_once("=" * 80)
    
    all_kernel_times = {}
    
    for rank in range(world_size):
        print_once(f"\nProcessing rank {rank}...")
        
        cpu_trace_path = os.path.join(output_dir, f'rank_{rank}_CPU_trace.json')
        gpu_trace_dir = os.path.join(output_dir, f'rank_{rank}')
        
        if not os.path.exists(cpu_trace_path):
            print_once(f"  ⚠ CPU trace not found for rank {rank}")
            continue
        
        if not os.path.exists(gpu_trace_dir):
            print_once(f"  ⚠ GPU trace directory not found for rank {rank}")
            continue
        
        # Parse CPU trace to get rf_ids
        config_rf_ids = parse_cpu_trace_for_rf_ids(cpu_trace_path, rank)
        
        # Parse GPU traces to get kernel times
        kernel_times = parse_gpu_traces_for_kernels(gpu_trace_dir, config_rf_ids, rank)
        
        all_kernel_times[rank] = kernel_times
    
    print_once("\n" + "=" * 80)
    print_once("KERNEL EXTRACTION COMPLETE")
    print_once("=" * 80)
    
    for rank in range(world_size):
        num_configs = len(all_kernel_times.get(rank, {}))
        print_once(f"Rank {rank}: {num_configs} configs with kernel times")
    
    return all_kernel_times


# ============================================================================
# PROFILING
# ============================================================================

def run_profiling(rank: int, world_size: int, output_dir: str, strict_serial: bool = True) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Dict[str, float]]]]:
    """
    Run profiling experiments with PyTorch profiler to capture kernel times.
    
    Returns:
        (measurements list, kernel_times dict)
    """
    print_once(f"\nGenerating configurations...")
    configs = generate_configs(world_size, MESSAGE_SIZES)
    print_once(f"✓ Generated {len(configs)} configurations")
    print_once(f"  {len(MESSAGE_SIZES)} message sizes × {world_size * (world_size - 1)} GPU pairs")
    
    print_once(f"\nAllocating buffers...")
    buffers = create_buffers(MESSAGE_SIZES, f'cuda:{rank}')
    print_once(f"✓ Buffers allocated")
    
    # Warmup
    if WARMUP_ITERATIONS > 0:
        print_once(f"\nWarmup ({WARMUP_ITERATIONS} iterations)...", rank)
        for config in configs[:min(len(configs), 100)]:  # Warmup on first 100 configs
            for _ in range(WARMUP_ITERATIONS):
                run_p2p(rank, config['src'], config['dst'], buffers[config['size']])
        
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize()
        print_once(f"✓ Warmup complete")
    
    # Profiling with PyTorch profiler
    print_once(f"\nRunning profiling ({REPEAT_COUNT} repetitions)...")
    if strict_serial:
        print_once(f"  Mode: STRICT SERIAL (barrier + sync after every measurement)")
    else:
        print_once(f"  Mode: RELAXED (sync only between configs)")
    
    # Configure profiler schedule
    total_steps = len(configs) * REPEAT_COUNT
    schedule = configure_profiler(
        wait_steps=0,
        warmup_steps=0,
        active_steps=1,
        repeat_count=total_steps
    )
    
    # Profile with CPU and GPU tracing
    with ProfilerContext(
        rank=rank,
        output_dir=output_dir,
        schedule=schedule,
        record_shapes=False,  # Don't need shapes for comms
        profile_memory=False,  # Don't need memory for comms
        with_stack=False,  # Faster without stack traces
    ) as prof:
        
        for i, config in enumerate(configs):
            if rank == 0 and i % 50 == 0 and i > 0:
                print(f"  Progress: {i}/{len(configs)} configs...")
            
            for rep in range(REPEAT_COUNT):
                # Create profiler annotation name
                name = (f"p2p_cfg{config['config_id']:04d}_"
                        f"s{config['src']}d{config['dst']}_"
                        f"sz{config['size']}_"
                        f"r{rep}")
                
                # Synchronize before measurement
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize()
                
                # Execute with profiler annotation
                with torch.profiler.record_function(name):
                    run_p2p(rank, config['src'], config['dst'], buffers[config['size']])
                
                torch.cuda.synchronize()
                
                # Step profiler after each config
                prof.step()
                
                # Strict serialization: barrier after every measurement
                if strict_serial:
                    torch.cuda.synchronize()
                    if dist.is_initialized():
                        dist.barrier()
    
    print_once(f"✓ Profiling complete (traces generated)")
    
    # Synchronize before extracting
    if dist.is_initialized():
        dist.barrier()
    
    # Extract kernel times from traces (rank 0 only)
    kernel_times = {}
    if rank == 0:
        kernel_times = extract_kernel_times_from_traces(output_dir, world_size)
    
    # Build measurements from kernel times
    measurements = []
    if rank == 0:
        for config in configs:
            src = config['src']
            # Get kernel times from source GPU
            src_kernel_times = kernel_times.get(src, {})
            
            for rep in range(REPEAT_COUNT):
                config_name = (f"p2p_cfg{config['config_id']:04d}_"
                              f"s{config['src']}d{config['dst']}_"
                              f"sz{config['size']}_"
                              f"r{rep}")
                
                kernel_data = src_kernel_times.get(config_name)
                
                if kernel_data:
                    measurements.append({
                        'config_id': config['config_id'],
                        'src': config['src'],
                        'dst': config['dst'],
                        'size': config['size'],
                        'repetition': rep,
                        'duration_ns': kernel_data['duration_us'] * 1000,  # Convert to ns
                        'duration_us': kernel_data['duration_us'],
                    })
    
    print_once(f"✓ Extracted {len(measurements)} kernel timing measurements")
    
    return measurements, kernel_times


# ============================================================================
# ANALYSIS - THREE REGRESSION METHODS
# ============================================================================

def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform standard linear regression (Ordinary Least Squares).
    
    Returns:
        (slope, intercept, r_squared)
    """
    result = stats.linregress(x, y)
    slope = result.slope
    intercept = result.intercept
    r_squared = result.rvalue ** 2
    
    # Ensure intercept is positive
    if intercept <= 0:
        intercept = 0.001
    
    return slope, intercept, r_squared


def huber_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Perform Huber regression (robust to outliers).
    
    Returns:
        (slope, intercept, r_squared)
    """
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    X = x.reshape(-1, 1)
    huber.fit(X, y)
    
    slope = huber.coef_[0]
    intercept = huber.intercept_
    
    # Ensure intercept is positive
    if intercept <= 0:
        intercept = 0.001
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared


def irls_regression(x: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-6) -> Tuple[float, float, float]:
    """
    Perform Iteratively Reweighted Least Squares (IRLS) regression.
    
    Returns:
        (slope, intercept, r_squared)
    """
    # Initialize with OLS
    X_design = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    
    for iteration in range(max_iter):
        # Calculate residuals
        y_pred = X_design @ beta
        residuals = y - y_pred
        
        # Calculate weights using Huber's function
        k = 1.345 * np.median(np.abs(residuals))  # Robust scale estimate
        weights = np.ones_like(residuals)
        outlier_mask = np.abs(residuals) > k
        weights[outlier_mask] = k / np.abs(residuals[outlier_mask])
        
        # Weighted least squares
        W = np.diag(weights)
        try:
            beta_new = np.linalg.lstsq(X_design.T @ W @ X_design, 
                                        X_design.T @ W @ y, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        
        # Check convergence
        if np.allclose(beta, beta_new, atol=tol):
            break
        
        beta = beta_new
    
    slope, intercept = beta
    
    # Ensure intercept is positive
    if intercept <= 0:
        intercept = 0.001
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared


def analyze_results(measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze kernel timing results using three regression methods.
    
    Returns:
        Analysis results dictionary with results from all three methods
    """
    print("\n" + "=" * 80)
    print("ANALYZING RESULTS (THREE REGRESSION METHODS ON KERNEL TIMES)")
    print("=" * 80)
    
    # Convert to arrays
    data = {}
    for m in measurements:
        key = (m['src'], m['dst'])
        if key not in data:
            data[key] = {'sizes': [], 'durations_us': []}
        data[key]['sizes'].append(m['size'])
        data[key]['durations_us'].append(m['duration_us'])
    
    # Perform all three regressions for each GPU pair
    results = []
    for (src, dst), pair_data in data.items():
        sizes = np.array(pair_data['sizes'])
        durations_us = np.array(pair_data['durations_us'])
        
        if len(sizes) < 5:
            print(f"⚠ Warning: GPU {src}→{dst} has insufficient data")
            continue
        
        # 1. Vanilla Linear Regression
        slope_lin, intercept_lin, r2_lin = linear_regression(sizes, durations_us)
        bw_lin = (1.0 / slope_lin) * 1e-3 if slope_lin > 0 else 0
        lat_lin = intercept_lin
        
        # 2. Huber Regression
        slope_hub, intercept_hub, r2_hub = huber_regression(sizes, durations_us)
        bw_hub = (1.0 / slope_hub) * 1e-3 if slope_hub > 0 else 0
        lat_hub = intercept_hub
        
        # 3. IRLS Regression
        slope_irls, intercept_irls, r2_irls = irls_regression(sizes, durations_us)
        bw_irls = (1.0 / slope_irls) * 1e-3 if slope_irls > 0 else 0
        lat_irls = intercept_irls
        
        results.append({
            'src': src,
            'dst': dst,
            # Linear regression results
            'latency_us_linear': lat_lin,
            'bandwidth_gbps_linear': bw_lin,
            'r_squared_linear': r2_lin,
            # Huber regression results
            'latency_us_huber': lat_hub,
            'bandwidth_gbps_huber': bw_hub,
            'r_squared_huber': r2_hub,
            # IRLS regression results
            'latency_us_irls': lat_irls,
            'bandwidth_gbps_irls': bw_irls,
            'r_squared_irls': r2_irls,
            # Metadata
            'num_samples': len(sizes),
        })
    
    return {
        'per_pair': results,
        'num_pairs': len(results),
    }


# ============================================================================
# OUTPUT AND REPORTING
# ============================================================================

def print_summary(analysis: Dict[str, Any], topology: Dict[str, Any]):
    """Print analysis summary with results from all three regression methods."""
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Based on GPU Kernel Times)")
    print("=" * 80)
    
    print(f"\nTopology:")
    print(f"  GPUs: {topology['num_gpus']}")
    print(f"  Interconnect: {topology['interconnect_type']}")
    if topology['nvlink_lanes'] > 0:
        print(f"  NVLink Lanes: {topology['nvlink_lanes']}")
    
    # Show PCIe info summary
    if topology.get('pcie_info'):
        pcie_info = topology['pcie_info']
        gen_max = set(info['pcie_gen_max'] for info in pcie_info if info['pcie_gen_max'] != 'Unknown')
        gen_cur = set(info['pcie_gen_current'] for info in pcie_info if info['pcie_gen_current'] != 'Unknown')
        width_max = set(info['link_width_max'] for info in pcie_info if info['link_width_max'] != 'Unknown')
        width_cur = set(info['link_width_current'] for info in pcie_info if info['link_width_current'] != 'Unknown')
        
        if len(gen_max) == 1 and len(width_max) == 1:
            gen = list(gen_max)[0]
            width = list(width_max)[0]
            if len(gen_cur) == 1 and list(gen_cur)[0] != gen:
                print(f"  PCIe: {gen} {width} (max), currently running at {list(gen_cur)[0]}")
            else:
                print(f"  PCIe: {gen} {width}")
        elif gen_max or width_max:
            print(f"  PCIe: Mixed configurations (see below for per-GPU details)")
        else:
            print(f"  PCIe: Information not available")
    
    # Show full topology matrix
    if topology.get('topo_matrix'):
        print(f"\nTopology Matrix (nvidia-smi topo -m):")
        print(topology['topo_matrix'])
    
    print(f"\nRegression Methods: Linear (OLS), Huber (robust), IRLS (iterative)")
    print(f"GPU Pairs Analyzed: {analysis['num_pairs']}")
    
    results = analysis['per_pair']
    
    # Print detailed table for each method
    for method, method_name in [('linear', 'LINEAR (OLS)'), ('huber', 'HUBER'), ('irls', 'IRLS')]:
        print(f"\n{'=' * 80}")
        print(f"REGRESSION METHOD: {method_name}")
        print(f"{'=' * 80}")
        print(f"\n{'Pair':<12} {'Latency (μs)':<15} {'Bandwidth (GB/s)':<18} {'R²':<10} {'Samples':<10}")
        print("-" * 80)
        
        for r in sorted(results, key=lambda x: (x['src'], x['dst'])):
            pair_str = f"GPU {r['src']}→{r['dst']}"
            lat_key = f'latency_us_{method}'
            bw_key = f'bandwidth_gbps_{method}'
            r2_key = f'r_squared_{method}'
            
            print(f"{pair_str:<12} {r[lat_key]:>12.2f}    {r[bw_key]:>15.3f}    "
                  f"{r[r2_key]:>8.4f}  {r['num_samples']:>8d}")
        
        # Summary statistics for this method
        latencies = [r[f'latency_us_{method}'] for r in results]
        bandwidths = [r[f'bandwidth_gbps_{method}'] for r in results]
        r_squareds = [r[f'r_squared_{method}'] for r in results]
        
        print("\n" + "-" * 80)
        print(f"Summary Statistics ({method_name}):")
        print(f"  Latency:    min={min(latencies):.2f} μs, max={max(latencies):.2f} μs, "
              f"mean={np.mean(latencies):.2f} μs, std={np.std(latencies):.2f} μs")
        print(f"  Bandwidth:  min={min(bandwidths):.3f} GB/s, max={max(bandwidths):.3f} GB/s, "
              f"mean={np.mean(bandwidths):.3f} GB/s, std={np.std(bandwidths):.3f} GB/s")
        print(f"  R² (fit quality): min={min(r_squareds):.4f}, max={max(r_squareds):.4f}, "
              f"mean={np.mean(r_squareds):.4f}")
    
    # Comparison table
    print(f"\n{'=' * 80}")
    print(f"COMPARISON: AVERAGE METRICS ACROSS ALL GPU PAIRS")
    print(f"{'=' * 80}")
    print(f"\n{'Method':<15} {'Avg Latency (μs)':<20} {'Avg Bandwidth (GB/s)':<22} {'Avg R²':<10}")
    print("-" * 80)
    
    for method, method_name in [('linear', 'Linear (OLS)'), ('huber', 'Huber'), ('irls', 'IRLS')]:
        lat_avg = np.mean([r[f'latency_us_{method}'] for r in results])
        bw_avg = np.mean([r[f'bandwidth_gbps_{method}'] for r in results])
        r2_avg = np.mean([r[f'r_squared_{method}'] for r in results])
        
        print(f"{method_name:<15} {lat_avg:>17.2f}    {bw_avg:>19.3f}    {r2_avg:>8.4f}")
    
    print("=" * 80)


def save_results(measurements: List[Dict[str, Any]], analysis: Dict[str, Any], 
                 topology: Dict[str, Any], output_dir: str, run_config: Dict[str, Any]):
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
    
    # 2. Save regression results CSV
    results_path = os.path.join(output_dir, 'regression_results.csv')
    with open(results_path, 'w', newline='') as f:
        if analysis['per_pair']:
            writer = csv.DictWriter(f, fieldnames=analysis['per_pair'][0].keys())
            writer.writeheader()
            writer.writerows(analysis['per_pair'])
    print(f"✓ Saved regression results CSV: {results_path}")
    
    # 3. Save topology info
    topo_path = os.path.join(output_dir, 'topology.txt')
    with open(topo_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU TOPOLOGY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Number of GPUs: {topology['num_gpus']}\n")
        f.write(f"Interconnect Type: {topology['interconnect_type']}\n")
        f.write(f"NVLink Lanes: {topology['nvlink_lanes']}\n\n")
        
        # PCIe information per GPU
        f.write("PCIe Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'GPU':<6} {'Gen (Max)':<12} {'Gen (Cur)':<12} {'Width (Max)':<13} {'Width (Cur)':<13}\n")
        f.write("-" * 80 + "\n")
        for info in topology.get('pcie_info', []):
            f.write(f"GPU {info['gpu_id']:<3} {info['pcie_gen_max']:<12} {info['pcie_gen_current']:<12} "
                   f"{info['link_width_max']:<13} {info['link_width_current']:<13}\n")
        f.write("\n")
        
        f.write("Topology Matrix:\n")
        f.write(topology['topo_matrix'])
    print(f"✓ Saved topology: {topo_path}")
    
    # 4. Save JSON with all extracted data
    json_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_gpus': topology['num_gpus'],
            'interconnect_type': topology['interconnect_type'],
            'nvlink_lanes': topology['nvlink_lanes'],
            'run_config': run_config,
        },
        'topology': {
            'num_gpus': topology['num_gpus'],
            'interconnect_type': topology['interconnect_type'],
            'nvlink_lanes': topology['nvlink_lanes'],
            'pcie_info': topology.get('pcie_info', []),
            'topo_matrix': topology['topo_matrix'],
        },
        'regression_results': {
            'num_pairs': analysis['num_pairs'],
            'per_pair': analysis['per_pair'],
        },
        'summary_statistics': {}
    }
    
    # Add summary statistics for each method
    results = analysis['per_pair']
    for method in ['linear', 'huber', 'irls']:
        latencies = [r[f'latency_us_{method}'] for r in results]
        bandwidths = [r[f'bandwidth_gbps_{method}'] for r in results]
        r_squareds = [r[f'r_squared_{method}'] for r in results]
        
        json_data['summary_statistics'][method] = {
            'latency_us': {
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies)),
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
            },
            'bandwidth_gbps': {
                'min': float(np.min(bandwidths)),
                'max': float(np.max(bandwidths)),
                'mean': float(np.mean(bandwidths)),
                'std': float(np.std(bandwidths)),
            },
            'r_squared': {
                'min': float(np.min(r_squareds)),
                'max': float(np.max(r_squareds)),
                'mean': float(np.mean(r_squareds)),
            }
        }
    
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Saved JSON data: {json_path}")
    
    # 5. Save comprehensive summary report
    summary_path = os.path.join(output_dir, 'RESULTS_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MINIMAL P2P PROFILER - COMPREHENSIVE RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Directory: {output_dir}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("RUN CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of GPUs: {topology['num_gpus']}\n")
        f.write(f"Message Sizes: {NUM_SIZES} (from {MIN_SIZE} to {MAX_SIZE} bytes)\n")
        f.write(f"Repetitions per config: {REPEAT_COUNT}\n")
        f.write(f"Warmup iterations: {WARMUP_ITERATIONS}\n")
        f.write(f"Clock locking: {'YES' if run_config.get('lock_clocks') else 'NO'}\n")
        f.write(f"Strict serial: {'YES' if run_config.get('strict_serial') else 'NO'}\n")
        f.write(f"Total measurements: {len(measurements)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("TOPOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write(f"GPUs: {topology['num_gpus']}\n")
        f.write(f"Interconnect: {topology['interconnect_type']}\n")
        if topology['nvlink_lanes'] > 0:
            f.write(f"NVLink Lanes: {topology['nvlink_lanes']}\n")
        
        f.write("\nPCIe Configuration:\n")
        f.write(f"{'GPU':<6} {'Gen (Max)':<12} {'Gen (Cur)':<12} {'Width (Max)':<13} {'Width (Cur)':<13}\n")
        f.write("-" * 80 + "\n")
        for info in topology.get('pcie_info', []):
            f.write(f"GPU {info['gpu_id']:<3} {info['pcie_gen_max']:<12} {info['pcie_gen_current']:<12} "
                   f"{info['link_width_max']:<13} {info['link_width_current']:<13}\n")
        
        # Write topology matrix
        if topology.get('topo_matrix'):
            f.write("\nTopology Matrix (nvidia-smi topo -m):\n")
            f.write(topology['topo_matrix'])
        f.write("\n")
        
        # Write detailed results for each method
        for method, method_name in [('linear', 'LINEAR (OLS)'), ('huber', 'HUBER'), ('irls', 'IRLS')]:
            f.write("=" * 80 + "\n")
            f.write(f"REGRESSION METHOD: {method_name}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Pair':<12} {'Latency (μs)':<15} {'Bandwidth (GB/s)':<18} {'R²':<10} {'Samples':<10}\n")
            f.write("-" * 80 + "\n")
            
            for r in sorted(results, key=lambda x: (x['src'], x['dst'])):
                pair_str = f"GPU {r['src']}→{r['dst']}"
                lat_key = f'latency_us_{method}'
                bw_key = f'bandwidth_gbps_{method}'
                r2_key = f'r_squared_{method}'
                
                f.write(f"{pair_str:<12} {r[lat_key]:>12.2f}    {r[bw_key]:>15.3f}    "
                       f"{r[r2_key]:>8.4f}  {r['num_samples']:>8d}\n")
            
            # Summary statistics
            stats = json_data['summary_statistics'][method]
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"Summary Statistics ({method_name}):\n")
            f.write(f"  Latency:    min={stats['latency_us']['min']:.2f} μs, "
                   f"max={stats['latency_us']['max']:.2f} μs, "
                   f"mean={stats['latency_us']['mean']:.2f} μs, "
                   f"std={stats['latency_us']['std']:.2f} μs\n")
            f.write(f"  Bandwidth:  min={stats['bandwidth_gbps']['min']:.3f} GB/s, "
                   f"max={stats['bandwidth_gbps']['max']:.3f} GB/s, "
                   f"mean={stats['bandwidth_gbps']['mean']:.3f} GB/s, "
                   f"std={stats['bandwidth_gbps']['std']:.3f} GB/s\n")
            f.write(f"  R²:         min={stats['r_squared']['min']:.4f}, "
                   f"max={stats['r_squared']['max']:.4f}, "
                   f"mean={stats['r_squared']['mean']:.4f}\n\n")
        
        # Comparison table
        f.write("=" * 80 + "\n")
        f.write("COMPARISON: AVERAGE METRICS ACROSS ALL GPU PAIRS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Method':<15} {'Avg Latency (μs)':<20} {'Avg Bandwidth (GB/s)':<22} {'Avg R²':<10}\n")
        f.write("-" * 80 + "\n")
        
        for method, method_name in [('linear', 'Linear (OLS)'), ('huber', 'Huber'), ('irls', 'IRLS')]:
            stats = json_data['summary_statistics'][method]
            f.write(f"{method_name:<15} {stats['latency_us']['mean']:>17.2f}    "
                   f"{stats['bandwidth_gbps']['mean']:>19.3f}    "
                   f"{stats['r_squared']['mean']:>8.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ Saved comprehensive summary: {summary_path}")
    
    print(f"\n{'=' * 80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'=' * 80}")
    print(f"\nKey files:")
    print(f"  - RESULTS_SUMMARY.txt      (comprehensive human-readable report)")
    print(f"  - results.json             (structured data for analysis)")
    print(f"  - regression_results.csv   (all regression metrics)")
    print(f"  - measurements.csv         (raw kernel timing data)")
    print(f"  - topology.txt             (hardware topology)")


def cleanup_traces(output_dir: str):
    """
    Clean up trace files after extraction.
    Removes CPU traces, GPU trace directories, and any other profiler artifacts.
    """
    print(f"\n{'=' * 80}")
    print("CLEANUP: REMOVING TRACE FILES")
    print(f"{'=' * 80}")
    
    cleaned_files = []
    
    # Remove CPU trace files
    cpu_traces = glob.glob(os.path.join(output_dir, 'rank_*_CPU_trace.json'))
    for f in cpu_traces:
        try:
            os.remove(f)
            cleaned_files.append(os.path.basename(f))
        except Exception as e:
            print(f"⚠ Warning: Could not remove {f}: {e}")
    
    # Remove GPU trace directories
    gpu_dirs = glob.glob(os.path.join(output_dir, 'rank_*'))
    for d in gpu_dirs:
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                cleaned_files.append(os.path.basename(d) + '/')
            except Exception as e:
                print(f"⚠ Warning: Could not remove {d}: {e}")
    
    # Remove any other profiler artifacts
    other_patterns = ['*.pt.trace.json', '*_trace.json']
    for pattern in other_patterns:
        for f in glob.glob(os.path.join(output_dir, pattern)):
            try:
                if os.path.isfile(f):
                    os.remove(f)
                    cleaned_files.append(os.path.basename(f))
            except Exception as e:
                print(f"⚠ Warning: Could not remove {f}: {e}")
    
    if cleaned_files:
        print(f"✓ Cleaned up {len(cleaned_files)} trace file(s)/directory(ies)")
        for f in cleaned_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(cleaned_files) > 10:
            print(f"  ... and {len(cleaned_files) - 10} more")
    else:
        print("✓ No trace files to clean up")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    # Parse arguments (defaults: lock-clocks=ON, strict-serial=ON)
    parser = argparse.ArgumentParser(
        description='Minimal Point-to-Point GPU Profiler with triple regression analysis',
        epilog='''
Examples:
  # Default (recommended - maximum reliability)
  torchrun --nproc_per_node=4 profile_minimal.py
  
  # Fast mode (no sudo, less isolated)
  torchrun --nproc_per_node=4 profile_minimal.py --no-lock-clocks --no-strict-serial
  
  # Custom output directory
  torchrun --nproc_per_node=4 profile_minimal.py --output-dir /data/results
  
  # 8 GPUs
  torchrun --nproc_per_node=8 profile_minimal.py

Defaults:
  - Clock locking: ENABLED (prevents thermal throttling)
  - Strict serialization: ENABLED (maximally isolated measurements)
  - Message sizes: 10 (evenly spaced from 1 KB to 1 GB)
  - Repetitions: 5 per config
  - Uses actual GPU kernel times (not CPU dispatch times)
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--no-lock-clocks', action='store_true',
                        help='Disable GPU clock locking (default: enabled, requires sudo)')
    parser.add_argument('--no-strict-serial', action='store_true',
                        help='Disable strict serialization (default: enabled)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./p2p_results_YYYYMMDD_HHMMSS)')
    args = parser.parse_args()
    
    # Apply defaults (invert the "no-" flags)
    lock_clocks = not args.no_lock_clocks
    strict_serial = not args.no_strict_serial
    
    print("\n" + "=" * 80)
    print("MINIMAL POINT-TO-POINT GPU PROFILER")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lock clocks: {'YES' if lock_clocks else 'NO'}")
    print(f"Strict serial: {'YES' if strict_serial else 'NO'}")
    print(f"Extraction method: GPU kernel times (ncclDevKernel_SendRecv)")
    
    # Store run configuration
    run_config = {
        'lock_clocks': lock_clocks,
        'strict_serial': strict_serial,
        'num_sizes': NUM_SIZES,
        'repeat_count': REPEAT_COUNT,
        'warmup_iterations': WARMUP_ITERATIONS,
        'min_size': MIN_SIZE,
        'max_size': MAX_SIZE,
    }
    
    # 1. Detect topology (before distributed init)
    if 'RANK' not in os.environ or int(os.environ.get('RANK', 0)) == 0:
        topology = get_topology_info()
    else:
        topology = {'num_gpus': 0, 'interconnect_type': 'Unknown', 'nvlink_lanes': 0, 'pcie_info': [], 'topo_matrix': ''}
    
    # 2. Initialize distributed
    print("\n" + "=" * 80)
    print("INITIALIZING DISTRIBUTED")
    print("=" * 80)
    rank, world_size = init_distributed(backend='nccl')
    print(f"Rank {rank}/{world_size} initialized on GPU {torch.cuda.current_device()}")
    
    # 3. Lock GPU clocks if enabled (rank 0 only)
    if lock_clocks and rank == 0:
        lock_gpu_clocks(world_size)
    if dist.is_initialized():
        dist.barrier()
    
    # 4. Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'./p2p_results_{timestamp}'
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    
    # 5. Run profiling (generates traces and extracts kernel times)
    print("\n" + "=" * 80)
    print("RUNNING PROFILING")
    print("=" * 80)
    measurements, kernel_times = run_profiling(rank, world_size, output_dir, strict_serial=strict_serial)
    
    # 6. Analysis (rank 0 only)
    if rank == 0:
        if not measurements:
            print("\n⚠ ERROR: No measurements extracted from traces!")
            print("  This likely means kernel extraction failed.")
            print("  Check that ncclDevKernel_SendRecv events are in GPU traces.")
        else:
            analysis = analyze_results(measurements)
            
            # 7. Print summary
            print_summary(analysis, topology)
            
            # 8. Save results
            save_results(measurements, analysis, topology, output_dir, run_config)
            
            # 9. Cleanup: Remove trace files
            cleanup_traces(output_dir)
    
    # 10. Unlock GPU clocks if we locked them (rank 0 only)
    if lock_clocks and rank == 0:
        unlock_gpu_clocks(world_size)
    if dist.is_initialized():
        dist.barrier()
    
    # Cleanup
    cleanup_distributed()
    
    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
