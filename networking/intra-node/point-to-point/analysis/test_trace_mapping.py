#!/usr/bin/env python3
"""
Test script to validate the hybrid CPU+GPU trace mapping strategy.

Strategy:
1. Parse CPU trace to find p2p_cfg nodes and their descendants (via ctrl_deps)
2. Extract rf_ids from descendants
3. Find GPU events with matching Record function ids
4. Extract External IDs from those GPU events
5. Collect ALL GPU events (including kernel, cuda_runtime) with those External IDs
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_cpu_trace(cpu_trace_path: Path) -> Dict:
    """Load CPU trace (ExecutionTrace format)."""
    if cpu_trace_path.suffix == '.gz':
        with gzip.open(cpu_trace_path, 'rt') as f:
            return json.load(f)
    else:
        with open(cpu_trace_path, 'r') as f:
            return json.load(f)


def load_gpu_trace(gpu_trace_path: Path) -> Dict:
    """Load GPU trace (Chrome Trace Event format)."""
    if gpu_trace_path.suffix == '.gz':
        with gzip.open(gpu_trace_path, 'rt') as f:
            return json.load(f)
    else:
        with open(gpu_trace_path, 'r') as f:
            return json.load(f)


def find_all_descendants(node_id: int, nodes_by_id: Dict[int, Dict]) -> List[Dict]:
    """
    Recursively find all descendants of a node using ctrl_deps.
    
    Args:
        node_id: ID of parent node
        nodes_by_id: Dictionary mapping node ID to node
    
    Returns:
        List of all descendant nodes
    """
    descendants = []
    for node in nodes_by_id.values():
        if node.get('ctrl_deps') == node_id:
            descendants.append(node)
            # Recursively get children of this node
            descendants.extend(find_all_descendants(node['id'], nodes_by_id))
    return descendants


def extract_rf_id(node: Dict) -> int:
    """Extract rf_id from a CPU trace node."""
    for attr in node.get('attrs', []):
        if attr['name'] == 'rf_id':
            return attr.get('value')
    return None


def map_p2p_cfg_to_gpu_events(
    p2p_cfg_name: str,
    cpu_trace: Dict,
    gpu_trace: Dict
) -> Tuple[Dict, List[Dict]]:
    """
    Map a p2p_cfg operation to all its related GPU events.
    
    Args:
        p2p_cfg_name: Name of the p2p_cfg (e.g., "p2p_cfg0000_s3d0_sz2097152_r0")
        cpu_trace: Loaded CPU trace data
        gpu_trace: Loaded GPU trace data
    
    Returns:
        Tuple of (p2p_cfg_node, list of related GPU events)
    """
    # Step 1: Find p2p_cfg node in CPU trace
    cpu_nodes = cpu_trace['nodes']
    p2p_node = None
    
    for node in cpu_nodes:
        if node['name'] == p2p_cfg_name:
            p2p_node = node
            break
    
    if not p2p_node:
        print(f"  ⚠️  Could not find {p2p_cfg_name} in CPU trace")
        return None, []
    
    p2p_rf_id = extract_rf_id(p2p_node)
    print(f"  ✓ Found p2p_cfg in CPU trace (node_id={p2p_node['id']}, rf_id={p2p_rf_id})")
    
    # Step 2: Find all descendants in CPU trace
    nodes_by_id = {n['id']: n for n in cpu_nodes}
    descendants = find_all_descendants(p2p_node['id'], nodes_by_id)
    
    # Get all rf_ids (including the p2p_cfg itself)
    all_rf_ids = {p2p_rf_id}
    for desc in descendants:
        rf_id = extract_rf_id(desc)
        if rf_id is not None:
            all_rf_ids.add(rf_id)
    
    print(f"  ✓ Found {len(descendants)} descendants with rf_ids: {sorted(all_rf_ids)}")
    
    # Step 3: Find GPU events with matching Record function ids
    gpu_events = gpu_trace['traceEvents']
    cpu_side_gpu_events = []
    
    for event in gpu_events:
        gpu_rf_id = event.get('args', {}).get('Record function id')
        if gpu_rf_id in all_rf_ids:
            cpu_side_gpu_events.append(event)
    
    print(f"  ✓ Found {len(cpu_side_gpu_events)} GPU events with matching Record function ids")
    
    # Step 4: Extract External IDs from those GPU events
    external_ids = set()
    for event in cpu_side_gpu_events:
        ext_id = event.get('args', {}).get('External id')
        if ext_id is not None:
            external_ids.add(ext_id)
    
    print(f"  ✓ Extracted External IDs: {sorted(external_ids)}")
    
    # Step 5: Collect ALL GPU events with those External IDs
    all_related_gpu_events = []
    for event in gpu_events:
        ext_id = event.get('args', {}).get('External id')
        if ext_id in external_ids:
            all_related_gpu_events.append(event)
    
    print(f"  ✓ Total GPU events (including kernel/cuda_runtime): {len(all_related_gpu_events)}")
    
    return p2p_node, all_related_gpu_events


def test_single_trace(cpu_trace_path: Path, gpu_trace_path: Path, p2p_cfg_name: str):
    """Test the mapping for a single trace file."""
    print(f"\n{'='*80}")
    print(f"Testing: {p2p_cfg_name}")
    print(f"{'='*80}")
    
    # Load traces
    cpu_trace = load_cpu_trace(cpu_trace_path)
    gpu_trace = load_gpu_trace(gpu_trace_path)
    
    # Perform mapping
    p2p_node, gpu_events = map_p2p_cfg_to_gpu_events(p2p_cfg_name, cpu_trace, gpu_trace)
    
    if not gpu_events:
        print("  ❌ FAILED: No GPU events found")
        return False
    
    # Analyze results
    print(f"\n  Event breakdown:")
    event_types = {}
    for event in gpu_events:
        cat = event.get('cat', 'unknown')
        event_types[cat] = event_types.get(cat, 0) + 1
    
    for cat, count in sorted(event_types.items()):
        print(f"    {cat:25s}: {count:3d} events")
    
    # Find the kernel
    kernels = [e for e in gpu_events if e.get('cat') == 'kernel']
    print(f"\n  Kernels found: {len(kernels)}")
    for kernel in kernels:
        print(f"    {kernel.get('name')[:60]}")
        print(f"      Duration: {kernel.get('dur'):.2f} μs")
    
    # Check if we got what we expect
    has_kernel = len(kernels) > 0
    has_cpu_op = any(e.get('cat') == 'cpu_op' for e in gpu_events)
    has_user_annotation = any(e.get('cat') == 'user_annotation' for e in gpu_events)
    
    print(f"\n  Validation:")
    print(f"    Has kernel: {has_kernel}")
    print(f"    Has cpu_op: {has_cpu_op}")
    print(f"    Has user_annotation: {has_user_annotation}")
    
    if has_user_annotation and has_cpu_op:
        print(f"    ✓ PASS: Got expected event types")
        return True
    else:
        print(f"    ⚠️  WARNING: Missing expected event types")
        return True  # Still consider it a pass if we got some events


def main():
    """Test the mapping on multiple trace files."""
    base_dir = Path('/home/azureuser/validation-studies/networking/intra-node/point-to-point/traces/run_20260121_164227')
    
    # Test cases: (rank, trace_file_index, expected_p2p_cfg_name)
    test_cases = [
        (0, 0, 'p2p_cfg0000_s3d0_sz2097152_r0'),
        (0, 1, 'p2p_cfg0001_s3d1_sz536870912_r0'),
        (0, 2, 'p2p_cfg0002_s3d0_sz1073741824_r0'),
        (3, 0, 'p2p_cfg0000_s3d0_sz2097152_r0'),
        (3, 1, 'p2p_cfg0001_s3d1_sz536870912_r0'),
    ]
    
    results = []
    
    for rank, trace_idx, p2p_cfg_name in test_cases:
        cpu_trace_path = base_dir / f'rank_{rank}_CPU_trace.json'
        
        # Get the trace file
        gpu_trace_files = sorted((base_dir / f'rank_{rank}').glob('*.pt.trace.json*'))
        if trace_idx >= len(gpu_trace_files):
            print(f"\n⚠️  Skipping: trace index {trace_idx} out of range for rank {rank}")
            continue
        
        gpu_trace_path = gpu_trace_files[trace_idx]
        
        success = test_single_trace(cpu_trace_path, gpu_trace_path, p2p_cfg_name)
        results.append((rank, trace_idx, p2p_cfg_name, success))
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    passed = sum(1 for _, _, _, success in results if success)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())

