"""
Profile point-to-point GPU communication using PyTorch Distributed + NCCL.

Systematically tests all GPU pairs with message sizes from 1B to 1GB,
capturing bandwidth and latency characteristics for simulator validation.
"""

import torch
import torch.distributed as dist
import time
import random
import csv
import os
import argparse
from datetime import datetime
from typing import Dict, Any

from p2p_config import (
    MESSAGE_SIZES,
    WARMUP_ITERATIONS,
    REPEAT_COUNT as DEFAULT_REPEAT_COUNT,
    RANDOM_SEED as DEFAULT_RANDOM_SEED,
    BUFFER_DTYPE,
    THERMAL_WARMUP_DURATION,
    THERMAL_TARGET_TEMP,
    TEMP_MONITOR_INTERVAL,
    DRIFT_MEASUREMENT_DURATION,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    generate_all_configs,
    get_config_summary,
    generate_linear_message_sizes,
)

# Import utilities from networking.utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from networking.utils import (
    init_distributed,
    cleanup_distributed,
    barrier_all,
    print_once,
    ProfilerContext,
    configure_profiler,
    TemperatureLogger,
    thermal_warmup,
    measure_clock_drift,
    capture_full_metadata,
    save_metadata,
)


def create_buffer_pool(message_sizes, dtype=torch.float32, device='cuda'):
    """
    Pre-allocate communication buffers efficiently.
    
    Strategy: Allocate only the largest buffer and reuse it (or slices) for smaller messages.
    This prevents OOM errors when using many message sizes with linear sampling.
    
    Args:
        message_sizes: List of message sizes in bytes
        dtype: Data type for buffers
        device: Device to allocate on
    
    Returns:
        Dictionary mapping size -> tensor
    """
    print_once(f"Allocating buffer pool for {len(message_sizes)} sizes...")
    
    bytes_per_element = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else dtype.itemsize
    buffer_pool = {}
    
    # Allocate the largest buffer first
    max_size = max(message_sizes)
    max_elements = max_size // bytes_per_element
    print_once(f"  Allocating max buffer: {max_size / (1024**3):.2f} GB ({max_elements} elements)")
    max_buffer = torch.randn(max_elements, dtype=dtype, device=device)
    
    # For all sizes, use slices of the max buffer
    for size in message_sizes:
        num_elements = size // bytes_per_element
        if num_elements > 0:
            # Use a slice of the max buffer (shares memory, no additional allocation)
            buffer_pool[size] = max_buffer[:num_elements].contiguous()
        else:
            # For very small messages (< 1 element), use a 1-element buffer
            buffer_pool[size] = max_buffer[:1].contiguous()
    
    print_once(f"Buffer pool allocated successfully (using {max_size / (1024**3):.2f} GB)")
    return buffer_pool


def run_p2p_test(rank, config, buffer_pool):
    """
    Execute a single point-to-point communication test.
    
    Args:
        rank: Current process rank
        config: Configuration dictionary with src, dst, size
        buffer_pool: Pre-allocated buffers
    """
    if rank == config['src']:
        # Sender
        tensor = buffer_pool[config['size']]
        dist.send(tensor, dst=config['dst'])
    elif rank == config['dst']:
        # Receiver
        tensor = buffer_pool[config['size']]
        dist.recv(tensor, src=config['src'])
    # Other ranks idle


def save_execution_log(execution_log, output_path):
    """Save execution log to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        if execution_log:
            writer = csv.DictWriter(f, fieldnames=execution_log[0].keys())
            writer.writeheader()
            writer.writerows(execution_log)
    
    print_once(f"Execution log saved to {output_path} ({len(execution_log)} entries)")


def main():
    """Main profiling function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Profile point-to-point GPU communication')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for traces and logs')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help='Random seed for config ordering')
    parser.add_argument('--repeat', type=int, default=DEFAULT_REPEAT_COUNT,
                        help='Number of repetitions per config (default: 10)')
    parser.add_argument('--linear-sizes', type=int, default=None,
                        help='Use N linearly-spaced message sizes instead of powers of 2 (default: None, uses 31 power-of-2 sizes). Recommended: 50 for better linear regression.')
    parser.add_argument('--no-warmup', action='store_true',
                        help='Skip thermal warmup and warmup iterations (faster for testing)')
    args = parser.parse_args()
    
    OUTPUT_DIR = args.output_dir
    RANDOM_SEED = args.seed
    REPEAT_COUNT = args.repeat
    SKIP_WARMUP = args.no_warmup
    
    # Override MESSAGE_SIZES if linear sampling is requested
    if args.linear_sizes is not None:
        MESSAGE_SIZES = generate_linear_message_sizes(args.linear_sizes)
        print(f"Using {len(MESSAGE_SIZES)} linearly-spaced message sizes ({MESSAGE_SIZES[0]} to {MESSAGE_SIZES[-1]} bytes)")
    else:
        print(f"Using {len(MESSAGE_SIZES)} power-of-2 message sizes ({MESSAGE_SIZES[0]} to {MESSAGE_SIZES[-1]} bytes)")
    
    # ========================================================================
    # 1. INITIALIZATION
    # ========================================================================
    
    print_once("=" * 80)
    print_once("POINT-TO-POINT COMMUNICATION PROFILING")
    print_once("=" * 80)
    
    # Initialize distributed
    rank, world_size = init_distributed(backend='nccl')
    print(f"Rank {rank}/{world_size} initialized")
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Create output directory (all ranks in case of race condition)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    barrier_all()
    
    # Save run info (rank 0 only)
    if rank == 0:
        run_info_path = os.path.join(OUTPUT_DIR, 'run_info.txt')
        with open(run_info_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("POINT-TO-POINT COMMUNICATION PROFILING - RUN INFO\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Output Directory: {OUTPUT_DIR}\n\n")
            f.write("Configuration:\n")
            f.write(f"  GPUs (world size): {world_size}\n")
            f.write(f"  Random seed: {RANDOM_SEED}\n")
            f.write(f"  Message sizes: {len(MESSAGE_SIZES)} (from {MESSAGE_SIZES[0]} to {MESSAGE_SIZES[-1]} bytes)\n")
            f.write(f"  Warmup iterations: {WARMUP_ITERATIONS}\n")
            f.write(f"  Repeat count: {REPEAT_COUNT}\n")
            f.write(f"  Thermal warmup: {THERMAL_WARMUP_DURATION}s to {THERMAL_TARGET_TEMP}°C\n")
            f.write(f"  Drift measurement: {DRIFT_MEASUREMENT_DURATION}s\n\n")
            f.write("=" * 80 + "\n")
    
    # ========================================================================
    # 2. TEMPERATURE MONITORING (Start)
    # ========================================================================
    
    if rank == 0:
        print_once(f"\nStarting temperature monitoring...")
        temp_logger = TemperatureLogger(
            gpu_ids=list(range(world_size)),
            interval=TEMP_MONITOR_INTERVAL,
            metrics=['temperature', 'power', 'clock_graphics']
        )
        temp_logger.start()
    
    # ========================================================================
    # 3. THERMAL WARMUP
    # ========================================================================
    
    if not SKIP_WARMUP:
        print_once(f"\nThermal warmup...")
        thermal_warmup(
            device=f'cuda:{rank}',
            duration_seconds=THERMAL_WARMUP_DURATION,
            target_temp_min=THERMAL_TARGET_TEMP
        )
        
        barrier_all()
        print_once("All GPUs warmed up")
    else:
        print_once(f"\nSkipping thermal warmup (--no-warmup enabled)")
        barrier_all()
    
    # ========================================================================
    # 4. INITIAL CLOCK DRIFT MEASUREMENT
    # ========================================================================
    
    print_once(f"\nMeasuring initial clock drift ({DRIFT_MEASUREMENT_DURATION}s)...")
    drift_start = measure_clock_drift(rank, duration_seconds=DRIFT_MEASUREMENT_DURATION)
    
    if rank == 0:
        drift_ppm = (drift_start['drift_ratio'] - 1.0) * 1e6
        print_once(f"Initial clock drift: {drift_ppm:.2f} ppm")
    
    # ========================================================================
    # 5. GENERATE CONFIGURATIONS
    # ========================================================================
    
    print_once(f"\nGenerating configurations...")
    configs = generate_all_configs(world_size, MESSAGE_SIZES)
    
    # Shuffle for pseudo-random ordering
    random.shuffle(configs)
    
    # Reassign config IDs based on shuffled order
    for i, config in enumerate(configs):
        config['config_id'] = i
    
    print_once(f"Total configurations: {len(configs)}")
    print_once(f"Message sizes: {len(MESSAGE_SIZES)} ({MESSAGE_SIZES[0]} to {MESSAGE_SIZES[-1]} bytes)")
    print_once(f"GPU pairs: {world_size * (world_size - 1)} (both directions)")
    
    # ========================================================================
    # 6. ALLOCATE BUFFER POOL
    # ========================================================================
    
    print_once(f"\nAllocating communication buffers...")
    buffer_pool = create_buffer_pool(MESSAGE_SIZES, dtype=BUFFER_DTYPE, device=f'cuda:{rank}')
    
    # ========================================================================
    # 7. WARMUP PHASE (No profiling)
    # ========================================================================
    
    if not SKIP_WARMUP:
        print_once(f"\nWarmup phase ({WARMUP_ITERATIONS} iterations per config)...")
        print_once(f"Total warmup operations: {len(configs) * WARMUP_ITERATIONS}")
        
        for i, config in enumerate(configs):
            if rank == 0 and i % 100 == 0:
                print(f"  Warmup progress: {i}/{len(configs)} configs...")
            
            for _ in range(WARMUP_ITERATIONS):
                run_p2p_test(rank, config, buffer_pool)
        
        barrier_all()
        torch.cuda.synchronize()
        print_once("Warmup complete")
    else:
        print_once(f"\nSkipping warmup iterations (--no-warmup enabled)")
        barrier_all()
    
    # ========================================================================
    # 8. PROFILED EXECUTION
    # ========================================================================
    
    print_once(f"\nStarting profiled execution...")
    print_once(f"Repetitions: {REPEAT_COUNT}")
    print_once(f"Total profiled operations: {len(configs) * REPEAT_COUNT}")
    
    # Configure profiler schedule
    # Set repeat to match total number of step() calls (once per config per repetition)
    total_steps = len(configs) * REPEAT_COUNT
    schedule = configure_profiler(
        wait_steps=0,
        warmup_steps=0,
        active_steps=1,
        repeat_count=total_steps
    )
    
    # Initialize execution log (rank 0 only)
    execution_log = [] if rank == 0 else None
    
    # Profile with both CPU and GPU tracing
    # Note: with_stack=False due to Python 3.13 incompatibility with PyTorch profiler stack tracking
    with ProfilerContext(
        rank=rank,
        output_dir=OUTPUT_DIR,
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        
        for rep in range(REPEAT_COUNT):
            if rank == 0:
                print(f"\nRepetition {rep+1}/{REPEAT_COUNT}")
            
            for i, config in enumerate(configs):
                if rank == 0 and i % 100 == 0 and i > 0:
                    print(f"  Progress: {i}/{len(configs)} configs...")
                
                # Create profiler annotation name
                name = (f"p2p_cfg{config['config_id']:04d}_"
                        f"s{config['src']}d{config['dst']}_"
                        f"sz{config['size']}_"
                        f"r{rep}")
                
                # Record start time (for execution log)
                start_time = time.perf_counter_ns()
                
                # Execute with profiler annotation
                with torch.profiler.record_function(name):
                    run_p2p_test(rank, config, buffer_pool)
                
                # Record end time
                end_time = time.perf_counter_ns()
                
                # Log execution (rank 0 only)
                if rank == 0:
                    execution_log.append({
                        'config_id': config['config_id'],
                        'src': config['src'],
                        'dst': config['dst'],
                        'size': config['size'],
                        'repetition': rep,
                        'timestamp_start_ns': start_time,
                        'timestamp_end_ns': end_time,
                        'duration_ns': end_time - start_time,
                    })
                
                # Step profiler after each config
                prof.step()
            
            # Synchronize after each repetition
            torch.cuda.synchronize()
    
    print_once("Profiled execution complete")
    
    # ========================================================================
    # 9. FINAL CLOCK DRIFT MEASUREMENT
    # ========================================================================
    
    print_once(f"\nMeasuring final clock drift ({DRIFT_MEASUREMENT_DURATION}s)...")
    drift_end = measure_clock_drift(rank, duration_seconds=DRIFT_MEASUREMENT_DURATION)
    
    if rank == 0:
        drift_ppm_end = (drift_end['drift_ratio'] - 1.0) * 1e6
        print_once(f"Final clock drift: {drift_ppm_end:.2f} ppm")
        
        # Check if drift changed
        drift_change = abs(drift_end['drift_ratio'] - drift_start['drift_ratio']) * 1e6
        if drift_change > 1.0:
            print_once(f"⚠ Warning: Drift changed by {drift_change:.2f} ppm during experiment")
        else:
            print_once(f"✓ Drift stable (changed by {drift_change:.2f} ppm)")
    
    # ========================================================================
    # 10. STOP TEMPERATURE MONITORING
    # ========================================================================
    
    if rank == 0:
        print_once(f"\nStopping temperature monitoring...")
        temp_logger.stop()
        temp_logger.save(os.path.join(OUTPUT_DIR, 'temperatures.csv'))
        temp_logger.print_summary()
        
        # Check for throttling
        throttling = temp_logger.check_throttling(temp_threshold=80.0)
        if throttling:
            print_once(f"⚠ Warning: {len(throttling)} temperature throttling events detected")
    
    # ========================================================================
    # 11. SAVE EXECUTION LOG
    # ========================================================================
    
    if rank == 0:
        print_once(f"\nSaving execution log...")
        save_execution_log(execution_log, os.path.join(OUTPUT_DIR, 'execution_log.csv'))
    
    # ========================================================================
    # 12. SAVE METADATA
    # ========================================================================
    
    print_once(f"\nSaving metadata...")
    metadata = capture_full_metadata(rank, gpu_id=rank)
    
    # Add drift measurements
    metadata['drift'] = {
        'start': drift_start,
        'end': drift_end,
    }
    
    # Add experiment info
    metadata['experiment'] = {
        'name': 'point_to_point_profiling',
        'timestamp': datetime.now().isoformat(),
        'output_dir': OUTPUT_DIR,
        'random_seed': RANDOM_SEED,
        'num_configs': len(configs),
        'repetitions': REPEAT_COUNT,
        'total_tests': len(configs) * REPEAT_COUNT,
    }
    
    save_metadata(metadata, os.path.join(OUTPUT_DIR, f'metadata_rank_{rank}.yml'))
    
    # ========================================================================
    # 13. CLEANUP
    # ========================================================================
    
    barrier_all()
    print_once("\n" + "=" * 80)
    print_once("PROFILING COMPLETE")
    print_once("=" * 80)
    print_once(f"\nOutput directory: {OUTPUT_DIR}")
    print_once(f"Files per rank:")
    print_once(f"  - rank_N.pt.trace.json (GPU trace with annotations)")
    print_once(f"  - rank_N_CPU_trace.json (CPU trace)")
    print_once(f"  - metadata_rank_N.yml (per-rank metadata)")
    
    if rank == 0:
        print_once(f"\nShared files (rank 0):")
        print_once(f"  - execution_log.csv (config execution order & timing)")
        print_once(f"  - temperatures.csv (temperature/power/clock time-series)")
    
    print_once("=" * 80)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()

