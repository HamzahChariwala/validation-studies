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
from datetime import datetime
from typing import Dict, Any

from p2p_config import (
    MESSAGE_SIZES,
    WARMUP_ITERATIONS,
    REPEAT_COUNT,
    RANDOM_SEED,
    BUFFER_DTYPE,
    THERMAL_WARMUP_DURATION,
    THERMAL_TARGET_TEMP,
    TEMP_MONITOR_INTERVAL,
    DRIFT_MEASUREMENT_DURATION,
    OUTPUT_DIR,
    generate_all_configs,
    get_config_summary,
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
    Pre-allocate all communication buffers.
    
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
    
    for size in message_sizes:
        num_elements = size // bytes_per_element
        # Random non-zero data (prevents hardware sparse optimizations)
        buffer = torch.randn(num_elements, dtype=dtype, device=device)
        buffer_pool[size] = buffer
    
    print_once(f"Buffer pool allocated successfully")
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
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    
    print_once(f"\nThermal warmup...")
    thermal_warmup(
        device=f'cuda:{rank}',
        duration=THERMAL_WARMUP_DURATION,
        target_temp_min=THERMAL_TARGET_TEMP
    )
    
    barrier_all()
    print_once("All GPUs warmed up")
    
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
    
    # ========================================================================
    # 8. PROFILED EXECUTION
    # ========================================================================
    
    print_once(f"\nStarting profiled execution...")
    print_once(f"Repetitions: {REPEAT_COUNT}")
    print_once(f"Total profiled operations: {len(configs) * REPEAT_COUNT}")
    
    # Configure profiler schedule
    schedule = configure_profiler(
        wait_steps=0,
        warmup_steps=0,
        active_steps=1,
        repeat_count=REPEAT_COUNT
    )
    
    # Initialize execution log (rank 0 only)
    execution_log = [] if rank == 0 else None
    
    # Profile with both CPU and GPU tracing
    with ProfilerContext(
        rank=rank,
        output_dir=OUTPUT_DIR,
        schedule=schedule,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
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
            
            # Step profiler after each repetition
            prof.step()
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

