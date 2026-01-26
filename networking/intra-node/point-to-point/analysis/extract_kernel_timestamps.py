#!/usr/bin/env python3
"""
Extract kernel timestamps from GPU traces and add them to trace_analysis.json.

This script:
1. Parses CPU traces to find rf_ids for zeroth repetition of each config
2. Parses GPU traces to find ncclDevKernel_SendRecv events and their timestamps
3. Adds 'zeroth_timestamp' to each config in trace_analysis.json
4. Adds 'trace_start_timestamps' with the first timestamp seen on each GPU

Usage:
    python3 extract_kernel_timestamps.py <trace_directory>
    
Example:
    python3 extract_kernel_timestamps.py ../traces/run_20260121_164227/
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import re


def parse_cpu_trace(cpu_trace_path, rank):
    """
    Parse CPU trace to extract rf_ids for zeroth repetition configs.
    
    Returns:
        dict: config_name -> rf_id
    """
    print(f"  Parsing CPU trace for rank {rank}...")
    
    with open(cpu_trace_path, 'r') as f:
        cpu_trace = json.load(f)
    
    nodes = cpu_trace.get('nodes', [])
    config_rf_ids = {}
    
    # Find all p2p_cfg nodes with _r0 (zeroth repetition)
    for node in nodes:
        name = node.get('name', '')
        
        # Match pattern: p2p_cfg####_s#d#_sz#######_r0
        if 'p2p_cfg' in name and '_r0' in name:
            # Extract rf_id from attrs
            attrs = node.get('attrs', [])
            rf_id = None
            for attr in attrs:
                if attr.get('name') == 'rf_id':
                    rf_id = attr.get('value')
                    break
            
            if rf_id is not None:
                # Extract config name without repetition suffix
                # e.g., p2p_cfg0000_s1d2_sz547827461_r0 -> p2p_cfg0000_s1d2_sz547827461
                config_name = '_'.join(name.split('_')[:-1])
                config_rf_ids[config_name] = rf_id
    
    print(f"    Found {len(config_rf_ids)} zeroth-repetition configs")
    return config_rf_ids


def parse_gpu_traces(gpu_trace_dir, config_rf_ids, rank):
    """
    Parse GPU traces to find ncclDevKernel_SendRecv events for given rf_ids.
    
    Strategy:
    1. Find record_function events with matching rf_ids
    2. Get their time ranges (ts to ts+dur)
    3. Find ncclDevKernel_SendRecv events within those time ranges
    
    Returns:
        dict: config_name -> {'timestamp': ts, 'duration': dur}
        float: first_timestamp (earliest timestamp seen in any trace)
        float: base_time_nanoseconds (for converting to absolute time)
    """
    print(f"  Parsing GPU traces for rank {rank}...")
    
    gpu_trace_files = sorted([f for f in os.listdir(gpu_trace_dir) if f.endswith('.json')])
    print(f"    Found {len(gpu_trace_files)} GPU trace files")
    
    # Build rf_id -> config_name mapping
    rf_id_to_config = {rf_id: config_name for config_name, rf_id in config_rf_ids.items()}
    
    # Track time ranges for each config
    config_time_ranges = {}  # config_name -> (start_ts, end_ts)
    
    # Track kernel timestamps
    config_timestamps = {}
    first_timestamp = None
    base_time_ns = None
    
    # Phase 1: Find record_function events and their time ranges
    print(f"    Phase 1: Finding record_function time ranges...")
    for i, filename in enumerate(gpu_trace_files):
        if i % 500 == 0 and i > 0:
            print(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
        filepath = os.path.join(gpu_trace_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                gpu_trace = json.load(f)
            
            # Get baseTimeNanoseconds from first trace (for absolute time conversion)
            if base_time_ns is None:
                base_time_ns = gpu_trace.get('baseTimeNanoseconds')
            
            events = gpu_trace.get('traceEvents', [])
            
            for event in events:
                # Track first timestamp
                if 'ts' in event:
                    ts = event['ts']
                    if first_timestamp is None or ts < first_timestamp:
                        first_timestamp = ts
                
                args = event.get('args', {})
                rf_id = args.get('Record function id')
                
                # If this is a record_function we're looking for
                if rf_id in rf_id_to_config:
                    config_name = rf_id_to_config[rf_id]
                    
                    # Only record if we haven't seen this config yet (zeroth repetition)
                    if config_name not in config_time_ranges:
                        start_ts = event.get('ts')
                        duration = event.get('dur', 0)
                        end_ts = start_ts + duration
                        
                        config_time_ranges[config_name] = (start_ts, end_ts)
        
        except Exception as e:
            print(f"      Warning: Failed to parse {filename}: {e}")
            continue
    
    print(f"    Found time ranges for {len(config_time_ranges)} configs")
    
    # Phase 2: Find ncclDevKernel_SendRecv events within those time ranges
    print(f"    Phase 2: Finding ncclDevKernel_SendRecv events...")
    
    for i, filename in enumerate(gpu_trace_files):
        if i % 500 == 0 and i > 0:
            print(f"      Progress: {i}/{len(gpu_trace_files)} files...")
        
        filepath = os.path.join(gpu_trace_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                gpu_trace = json.load(f)
            
            events = gpu_trace.get('traceEvents', [])
            
            for event in events:
                name = event.get('name', '')
                
                if 'ncclDevKernel_SendRecv' in name:
                    kernel_ts = event.get('ts')
                    kernel_dur = event.get('dur')
                    
                    # Check if this kernel falls within any config's time range
                    for config_name, (start_ts, end_ts) in config_time_ranges.items():
                        if start_ts <= kernel_ts <= end_ts:
                            # Only record if we haven't seen this config yet
                            if config_name not in config_timestamps:
                                config_timestamps[config_name] = {
                                    'timestamp': kernel_ts,
                                    'duration': kernel_dur
                                }
                            break
        
        except Exception as e:
            continue
    
    print(f"    Found timestamps for {len(config_timestamps)} configs")
    return config_timestamps, first_timestamp, base_time_ns


def extract_timestamps_for_all_ranks(trace_dir):
    """
    Extract kernel timestamps for all ranks.
    
    Returns:
        dict: config_name -> timestamp (from source GPU)
        dict: rank -> first_timestamp
    """
    trace_dir = Path(trace_dir)
    
    print("\n" + "="*80)
    print("EXTRACTING KERNEL TIMESTAMPS FROM GPU TRACES")
    print("="*80)
    
    # Find all ranks
    cpu_trace_files = sorted(trace_dir.glob('rank_*_CPU_trace.json'))
    ranks = [int(f.stem.split('_')[1]) for f in cpu_trace_files]
    
    print(f"\nFound {len(ranks)} ranks: {ranks}")
    
    # Store results
    all_config_timestamps = {}
    trace_start_timestamps = {}
    base_time_nanoseconds = {}
    
    # Process each rank
    for rank in ranks:
        print(f"\nProcessing rank {rank}...")
        
        cpu_trace_path = trace_dir / f'rank_{rank}_CPU_trace.json'
        gpu_trace_dir = trace_dir / f'rank_{rank}'
        
        if not cpu_trace_path.exists():
            print(f"  Warning: CPU trace not found for rank {rank}")
            continue
        
        if not gpu_trace_dir.exists():
            print(f"  Warning: GPU trace directory not found for rank {rank}")
            continue
        
        # Parse CPU trace to get rf_ids
        config_rf_ids = parse_cpu_trace(cpu_trace_path, rank)
        
        # Parse GPU traces to get timestamps
        config_timestamps, first_timestamp, base_time_ns = parse_gpu_traces(gpu_trace_dir, config_rf_ids, rank)
        
        # Store results
        all_config_timestamps[rank] = config_timestamps
        trace_start_timestamps[rank] = first_timestamp
        base_time_nanoseconds[rank] = base_time_ns
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    
    # Summary
    for rank in ranks:
        num_configs = len(all_config_timestamps.get(rank, {}))
        first_ts = trace_start_timestamps.get(rank)
        base_time = base_time_nanoseconds.get(rank)
        print(f"Rank {rank}: {num_configs} configs, first timestamp: {first_ts}, base_time: {base_time}")
    
    return all_config_timestamps, trace_start_timestamps, base_time_nanoseconds


def update_trace_analysis_json(trace_dir, all_config_timestamps, trace_start_timestamps, base_time_nanoseconds):
    """
    Update trace_analysis.json with kernel timestamps.
    """
    trace_dir = Path(trace_dir)
    trace_analysis_path = trace_dir / 'trace_analysis.json'
    
    print("\n" + "="*80)
    print("UPDATING TRACE_ANALYSIS.JSON")
    print("="*80)
    
    # Load existing trace_analysis.json
    print(f"\nLoading {trace_analysis_path}...")
    with open(trace_analysis_path, 'r') as f:
        trace_analysis = json.load(f)
    
    print(f"Loaded {len(trace_analysis)} configs")
    
    # Add metadata as top-level keys
    trace_analysis['_trace_start_timestamps'] = trace_start_timestamps
    trace_analysis['_base_time_nanoseconds'] = base_time_nanoseconds
    
    # Update each config with zeroth_timestamp
    configs_updated = 0
    configs_missing = 0
    
    for config_name, config_data in trace_analysis.items():
        if config_name.startswith('_'):
            continue  # Skip metadata keys
        
        # Get metadata
        metadata = config_data.get('metadata', {})
        src = metadata.get('src')
        
        if src is None:
            continue
        
        # Get timestamp from source GPU
        rank_timestamps = all_config_timestamps.get(src, {})
        timestamp_data = rank_timestamps.get(config_name)
        
        if timestamp_data:
            # Add zeroth_timestamp to metadata
            metadata['zeroth_timestamp'] = timestamp_data['timestamp']
            metadata['zeroth_duration'] = timestamp_data['duration']
            configs_updated += 1
        else:
            configs_missing += 1
    
    print(f"\nConfigs updated: {configs_updated}")
    print(f"Configs missing timestamps: {configs_missing}")
    
    # Save updated trace_analysis.json
    output_path = trace_analysis_path
    print(f"\nSaving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(trace_analysis, f, indent=2)
    
    print(f"✓ Saved successfully")
    
    return configs_updated, configs_missing


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 extract_kernel_timestamps.py <trace_directory>")
        print("\nExample:")
        print("  python3 extract_kernel_timestamps.py ../traces/run_20260121_164227/")
        sys.exit(1)
    
    trace_dir = sys.argv[1]
    
    if not os.path.exists(trace_dir):
        print(f"Error: Directory not found: {trace_dir}")
        sys.exit(1)
    
    # Extract timestamps
    all_config_timestamps, trace_start_timestamps, base_time_nanoseconds = extract_timestamps_for_all_ranks(trace_dir)
    
    # Update trace_analysis.json
    configs_updated, configs_missing = update_trace_analysis_json(
        trace_dir, all_config_timestamps, trace_start_timestamps, base_time_nanoseconds
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Configs updated with timestamps: {configs_updated}")
    print(f"Configs missing timestamps: {configs_missing}")
    
    if configs_missing > 0:
        print(f"\n⚠ Warning: {configs_missing} configs are missing timestamps")
        print("  This may be expected if not all GPUs participated in all configs")
    
    print("\n✓ Kernel timestamp extraction complete!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

