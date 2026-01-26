"""
Parse matmul operations from CPU and device traces.
Extracts operation names, tensor sizes, and execution durations.
Maps operations to their configuration parameters (size, precision, layout, batch).
"""

import json
import glob
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Import config generator to match operations to configs
sys.path.insert(0, str(Path(__file__).parent.parent))
from matmul_config import generate_experiment_configs


def load_cpu_trace(trace_path: str) -> Dict:
    """Load Chakra ET CPU trace."""
    print(f"Loading CPU trace: {trace_path}")
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data


def load_device_traces(trace_dir: str) -> List[Dict]:
    """Load all Kineto device traces."""
    trace_files = glob.glob(f"{trace_dir}/*.pt.trace.json")
    print(f"Found {len(trace_files)} device traces")
    
    traces = []
    for trace_file in sorted(trace_files):
        print(f"Loading device trace: {trace_file}")
        with open(trace_file, 'r') as f:
            data = json.load(f)
        traces.append(data)
    
    return traces


def is_matmul_op(op_name: str) -> bool:
    """Check if operation name represents a matrix multiplication."""
    matmul_patterns = [
        'matmul',
        'mm',
        'bmm',
        'addmm',
        'addbmm',
        'baddbmm',
    ]
    op_lower = op_name.lower()
    return any(pattern in op_lower for pattern in matmul_patterns)


def extract_rf_id(node: Dict) -> int:
    """Extract rf_id from CPU trace node attributes."""
    for attr in node.get('attrs', []):
        if attr.get('name') == 'rf_id':
            return attr.get('value')
    return None


def extract_tensor_shapes(node: Dict) -> Tuple:
    """Extract input tensor shapes from CPU trace node."""
    inputs = node.get('inputs', {})
    shapes = inputs.get('shapes', [])
    
    # Filter out empty shapes and get the actual tensor dimensions
    tensor_shapes = []
    for shape in shapes:
        if isinstance(shape, list) and len(shape) > 0:
            # Only keep non-empty shapes
            if all(isinstance(dim, int) for dim in shape):
                tensor_shapes.append(tuple(shape))
    
    return tuple(tensor_shapes) if tensor_shapes else None


def match_op_to_config_by_order(op_index: int, num_matmuls_per_config: int, configs: List[Dict]) -> Dict:
    """
    Match an operation to its configuration based on execution order.
    
    Since configs are executed sequentially and we run each config multiple times,
    we can determine which config an operation belongs to based on its index.
    
    Args:
        op_index: The index of this matmul operation in the trace
        num_matmuls_per_config: Number of matmul ops per config execution
        configs: List of all configurations
    
    Returns:
        The matching configuration dict
    """
    # Each config produces num_matmuls_per_config operations (e.g., matmul + mm)
    # Operations are grouped: all repeats of config 0, then all repeats of config 1, etc.
    
    # Determine which config this operation belongs to
    config_idx = (op_index // num_matmuls_per_config) % len(configs)
    
    return configs[config_idx] if config_idx < len(configs) else None


def parse_cpu_trace(cpu_trace: Dict, configs: List[Dict]) -> List[Dict]:
    """Parse CPU trace to find all matmul operations and match to configs."""
    matmul_ops = []
    
    nodes = cpu_trace.get('nodes', [])
    print(f"Parsing {len(nodes)} nodes from CPU trace...")
    
    matmul_count = 0
    for node in nodes:
        op_name = node.get('name', '')
        
        if is_matmul_op(op_name):
            rf_id = extract_rf_id(node)
            tensor_shapes = extract_tensor_shapes(node)
            
            if rf_id is not None and tensor_shapes:
                # Match operation to its configuration based on order
                # Pattern: each step runs all configs, each config produces ~2 ops (matmul + mm/bmm)
                # So: ops (0,1) = config 0 step 0, ops (2,3) = config 1 step 0, etc.
                config_idx = (matmul_count // 2) % len(configs)
                config = configs[config_idx] if config_idx < len(configs) else None
                
                matmul_ops.append({
                    'op_name': op_name,
                    'rf_id': rf_id,
                    'tensor_shapes': tensor_shapes,
                    'node_id': node.get('id'),
                    'config': config,
                    'op_index': matmul_count
                })
                
                matmul_count += 1
    
    print(f"Found {len(matmul_ops)} matmul operations in CPU trace")
    print(f"Operations per config: {matmul_count / len(configs) if configs else 0:.1f}")
    return matmul_ops


def build_rfid_to_duration_map(device_traces: List[Dict]) -> Dict[int, List[float]]:
    """Build mapping from record function id to list of durations across all traces."""
    rfid_durations = defaultdict(list)
    
    for trace_idx, trace in enumerate(device_traces):
        events = trace.get('traceEvents', [])
        
        for event in events:
            args = event.get('args', {})
            rf_id = args.get('Record function id')
            duration = event.get('dur')  # Duration in microseconds
            
            if rf_id is not None and duration is not None:
                rfid_durations[rf_id].append(duration)
    
    print(f"Built duration map for {len(rfid_durations)} unique rf_ids")
    return rfid_durations


def compute_statistics(durations: List[float]) -> Dict[str, float]:
    """Compute mean, range, and standard deviation of durations."""
    if not durations:
        return {
            'mean': 0.0,
            'min': 0.0,
            'max': 0.0,
            'range': 0.0,
            'std': 0.0,
            'count': 0
        }
    
    arr = np.array(durations)
    return {
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'range': float(np.max(arr) - np.min(arr)),
        'std': float(np.std(arr)),
        'count': len(durations)
    }


def create_database(matmul_ops: List[Dict], rfid_durations: Dict[int, List[float]]) -> Dict:
    """Create JSON database with matmul operations and their durations."""
    database = {}
    skipped = 0
    
    for op in matmul_ops:
        rf_id = op['rf_id']
        op_name = op['op_name']
        tensor_shapes = op['tensor_shapes']
        config = op['config']
        
        if config is None:
            skipped += 1
            continue
        
        # Create a detailed key including all config parameters
        shapes_str = '_'.join(['x'.join(map(str, shape)) for shape in tensor_shapes])
        transpose_str = ""
        if config['transpose_a'] or config['transpose_b']:
            transpose_parts = []
            if config['transpose_a']:
                transpose_parts.append("At")
            if config['transpose_b']:
                transpose_parts.append("Bt")
            transpose_str = "_" + "_".join(transpose_parts)
        
        # Key format: opname_size_dtype_transpose_batchsize
        key = f"{op_name}_{shapes_str}_{config['dtype']}{transpose_str}"
        if config['batch_size'] > 1:
            key += f"_B{config['batch_size']}"
        
        # Get durations for this rf_id
        durations = rfid_durations.get(rf_id, [])
        
        # If this key already exists, append durations
        if key in database:
            database[key]['durations'].extend(durations)
            database[key]['rf_ids'].append(rf_id)
        else:
            database[key] = {
                'op_name': op_name,
                'tensor_shapes': [list(shape) for shape in tensor_shapes] if tensor_shapes else None,
                'config': {
                    'm': config['m'],
                    'n': config['n'],
                    'k': config['k'],
                    'dtype': config['dtype'],
                    'transpose_a': config['transpose_a'],
                    'transpose_b': config['transpose_b'],
                    'batch_size': config['batch_size'],
                    'config_id': config['config_id']
                },
                'rf_ids': [rf_id],
                'durations': durations.copy()
            }
    
    # Compute statistics for each entry
    for key, entry in database.items():
        stats = compute_statistics(entry['durations'])
        entry['statistics'] = stats
    
    print(f"Created database with {len(database)} unique operation configurations")
    if skipped > 0:
        print(f"Skipped {skipped} operations that couldn't be matched to configs")
    return database


def main():
    """Main execution function."""
    trace_dir = Path(__file__).parent / "traces_matmul"
    
    # Generate the same config list that was used during profiling
    print("Generating configuration list...")
    configs = generate_experiment_configs()
    print(f"Generated {len(configs)} configurations\n")
    
    # Load traces
    cpu_trace = load_cpu_trace(trace_dir / "CPU_trace.json")
    device_traces = load_device_traces(str(trace_dir))
    
    # Parse CPU trace for matmul operations
    matmul_ops = parse_cpu_trace(cpu_trace, configs)
    
    # Build duration map from device traces
    rfid_durations = build_rfid_to_duration_map(device_traces)
    
    # Create database
    database = create_database(matmul_ops, rfid_durations)
    
    # Save database
    output_file = Path(__file__).parent / "matmul_database.json"
    print(f"\nSaving database to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(database, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total unique matmul configurations: {len(database)}")
    
    # Show a few examples
    print("\nExample entries:")
    for i, (key, entry) in enumerate(list(database.items())[:5]):
        stats = entry['statistics']
        cfg = entry['config']
        print(f"\n{i+1}. {key}")
        print(f"   Operation: {entry['op_name']}")
        print(f"   Config: {cfg['m']}×{cfg['k']} @ {cfg['k']}×{cfg['n']}, "
              f"dtype={cfg['dtype']}, transpose_a={cfg['transpose_a']}, "
              f"transpose_b={cfg['transpose_b']}, batch={cfg['batch_size']}")
        print(f"   Count: {stats['count']} measurements")
        print(f"   Duration: {stats['mean']:.2f} ± {stats['std']:.2f} μs (range: {stats['min']:.2f} - {stats['max']:.2f})")
    
    print(f"\nFull database saved to: {output_file}")


if __name__ == "__main__":
    main()

