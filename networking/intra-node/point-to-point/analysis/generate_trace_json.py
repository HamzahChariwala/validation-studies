#!/usr/bin/env python3
"""
Generate JSON output from CPU and GPU traces.

Output format:
{
  "p2p_cfg0000_s3d0_sz2097152": {
    "metadata": {"src": 3, "dst": 0, "size": 2097152},
    "cpu_duration": [null, 113.267, null, ...],  # in milliseconds
    "gpu_duration": {
      "c10d::recv_": [null, 112.729, ...],
      "ncclDevKernel_SendRecv": [null, 1.723, ...],
      ...
    }
  }
}
"""

import json
import gzip
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_json_file(path: Path) -> Dict:
    """Load JSON file (handles .gz compression)."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)


def parse_config_name(name: str) -> Optional[Dict]:
    """
    Parse p2p_cfg name.
    Format: p2p_cfg{id:04d}_s{src}d{dst}_sz{size}_r{rep}
    Returns: dict with src, dst, size, repetition, and config_key (without rep)
    """
    pattern = r'p2p_cfg(\d+)_s(\d+)d(\d+)_sz(\d+)_r(\d+)'
    match = re.match(pattern, name)
    
    if match:
        config_id = int(match.group(1))
        src = int(match.group(2))
        dst = int(match.group(3))
        size = int(match.group(4))
        rep = int(match.group(5))
        
        # Config key without repetition
        config_key = f"p2p_cfg{config_id:04d}_s{src}d{dst}_sz{size}"
        
        return {
            'config_id': config_id,
            'src': src,
            'dst': dst,
            'size': size,
            'repetition': rep,
            'config_key': config_key
        }
    return None


def extract_rf_id(node: Dict) -> Optional[int]:
    """Extract rf_id from CPU trace node."""
    for attr in node.get('attrs', []):
        if attr['name'] == 'rf_id':
            return attr.get('value')
    return None


def find_all_descendants(node_id: int, nodes_by_id: Dict[int, Dict]) -> List[Dict]:
    """Recursively find all descendants using ctrl_deps."""
    descendants = []
    for node in nodes_by_id.values():
        if node.get('ctrl_deps') == node_id:
            descendants.append(node)
            descendants.extend(find_all_descendants(node['id'], nodes_by_id))
    return descendants


def truncate_op_name(name: str, max_len: int = 50) -> str:
    """Truncate overly long operation names."""
    if len(name) <= max_len:
        return name
    
    # For function signatures, try to keep the function name and truncate params
    if '(' in name:
        func_name = name.split('(')[0]
        if len(func_name) <= max_len:
            return func_name
    
    return name[:max_len]


def map_p2p_to_gpu_events(p2p_cfg_name: str, cpu_trace: Dict, gpu_trace: Dict) -> List[Dict]:
    """Map a p2p_cfg to all related GPU events using hybrid CPU+GPU trace approach."""
    cpu_nodes = cpu_trace['nodes']
    
    # Find p2p_cfg node in CPU trace
    p2p_node = None
    for node in cpu_nodes:
        if node['name'] == p2p_cfg_name:
            p2p_node = node
            break
    
    if not p2p_node:
        return []
    
    # Find all descendants
    nodes_by_id = {n['id']: n for n in cpu_nodes}
    descendants = find_all_descendants(p2p_node['id'], nodes_by_id)
    
    # Get all rf_ids
    all_rf_ids = {extract_rf_id(p2p_node)}
    for desc in descendants:
        rf_id = extract_rf_id(desc)
        if rf_id is not None:
            all_rf_ids.add(rf_id)
    
    # Find GPU events with matching Record function ids
    gpu_events = gpu_trace['traceEvents']
    external_ids = set()
    
    for event in gpu_events:
        gpu_rf_id = event.get('args', {}).get('Record function id')
        if gpu_rf_id in all_rf_ids:
            ext_id = event.get('args', {}).get('External id')
            if ext_id is not None:
                external_ids.add(ext_id)
    
    # Collect ALL GPU events with those External IDs
    related_events = []
    for event in gpu_events:
        ext_id = event.get('args', {}).get('External id')
        if ext_id in external_ids:
            related_events.append(event)
    
    return related_events


def process_rank_traces(
    rank: int,
    run_dir: Path,
    max_traces: int,
    max_reps: int
) -> Dict:
    """
    Process traces for a single rank.
    
    Returns: Dictionary mapping config_key to data
    """
    print(f"\nProcessing rank {rank}...")
    
    # Load CPU trace
    cpu_trace_path = run_dir / f'rank_{rank}_CPU_trace.json'
    if not cpu_trace_path.exists():
        print(f"  ⚠️  CPU trace not found: {cpu_trace_path}")
        return {}
    
    cpu_trace = load_json_file(cpu_trace_path)
    
    # Get GPU trace files
    gpu_trace_dir = run_dir / f'rank_{rank}'
    if not gpu_trace_dir.exists():
        print(f"  ⚠️  GPU trace directory not found: {gpu_trace_dir}")
        return {}
    
    gpu_trace_files = sorted(gpu_trace_dir.glob('*.pt.trace.json*'))
    if max_traces is not None:
        gpu_trace_files = gpu_trace_files[:max_traces]
    
    print(f"  Found {len(gpu_trace_files)} GPU trace files")
    
    # Load execution log for CPU durations
    exec_log_path = run_dir / 'execution_log.csv'
    cpu_durations = {}
    if exec_log_path.exists():
        import pandas as pd
        df = pd.read_csv(exec_log_path)
        # Convert nanoseconds to milliseconds
        for _, row in df.iterrows():
            config_key = f"p2p_cfg{int(row['config_id']):04d}_s{int(row['src'])}d{int(row['dst'])}_sz{int(row['size'])}"
            rep = int(row['repetition'])
            duration_ms = row['duration_ns'] / 1_000_000.0  # ns to ms
            
            if config_key not in cpu_durations:
                cpu_durations[config_key] = {}
            cpu_durations[config_key][rep] = duration_ms
    
    # Process each GPU trace
    results = defaultdict(lambda: {
        'metadata': {},
        'cpu_duration': [None] * max_reps,
        'gpu_duration': defaultdict(lambda: [None] * max_reps)
    })
    
    for i, gpu_trace_file in enumerate(gpu_trace_files):
        if i % 50 == 0 and i > 0:
            print(f"  Processed {i}/{len(gpu_trace_files)} files...")
        
        try:
            gpu_trace = load_json_file(gpu_trace_file)
        except Exception as e:
            print(f"  ⚠️  Failed to load {gpu_trace_file.name}: {e}")
            continue
        
        # Find p2p_cfg event
        gpu_events = gpu_trace.get('traceEvents', [])
        p2p_cfg_event = None
        for event in gpu_events:
            if event.get('cat') == 'user_annotation' and 'p2p_cfg' in event.get('name', ''):
                p2p_cfg_event = event
                break
        
        if not p2p_cfg_event:
            continue
        
        # Parse config
        config = parse_config_name(p2p_cfg_event['name'])
        if not config:
            continue
        
        config_key = config['config_key']
        rep = config['repetition']
        
        # Only process if this rank is the sender
        if rank != config['src']:
            continue
        
        # Set metadata
        if not results[config_key]['metadata']:
            results[config_key]['metadata'] = {
                'src': config['src'],
                'dst': config['dst'],
                'size': config['size']
            }
        
        # Set CPU duration
        if config_key in cpu_durations and rep in cpu_durations[config_key]:
            results[config_key]['cpu_duration'][rep] = cpu_durations[config_key][rep]
        
        # Map to all related GPU events
        related_events = map_p2p_to_gpu_events(p2p_cfg_event['name'], cpu_trace, gpu_trace)
        
        # Group by operation name and record durations
        for event in related_events:
            # Skip events without duration
            if 'dur' not in event:
                continue
            
            op_name = truncate_op_name(event.get('name', 'unknown'))
            duration_ms = event.get('dur') / 1000.0  # μs to ms
            
            results[config_key]['gpu_duration'][op_name][rep] = duration_ms
    
    print(f"  Extracted {len(results)} unique configurations")
    
    # Convert defaultdicts to regular dicts for JSON serialization
    final_results = {}
    for config_key, data in results.items():
        final_results[config_key] = {
            'metadata': data['metadata'],
            'cpu_duration': data['cpu_duration'],
            'gpu_duration': {
                op_name: durations 
                for op_name, durations in data['gpu_duration'].items()
            }
        }
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description='Generate JSON from CPU/GPU traces'
    )
    parser.add_argument(
        'run_dir',
        type=str,
        help='Path to run directory'
    )
    parser.add_argument(
        '--max-traces',
        type=int,
        default=None,
        help='Maximum traces to process per rank (default: all)'
    )
    parser.add_argument(
        '--max-reps',
        type=int,
        default=10,
        help='Maximum repetitions expected (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON path (default: run_dir/trace_analysis.json)'
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    print(f"Processing traces from: {run_dir}")
    print(f"Max traces per rank: {args.max_traces or 'all'}")
    print(f"Max repetitions: {args.max_reps}")
    
    # Process all ranks and merge results
    all_results = {}
    
    for rank in range(4):  # Ranks 0-3
        rank_results = process_rank_traces(rank, run_dir, args.max_traces, args.max_reps)
        
        # Merge with existing results
        for config_key, data in rank_results.items():
            all_results[config_key] = data
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / 'trace_analysis.json'
    
    # Save JSON
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Wrote {len(all_results)} configurations to {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

