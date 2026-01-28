#!/usr/bin/env python3
"""
Build unified matmul operation database from CPU and GPU traces.

This script:
1. Loads configs from matmul_config.py in execution order
2. Parses CPU and GPU traces
3. Matches operations to configs using sequential + validation
4. Extracts kernel and timing data only
5. Computes statistics across repeats
6. Exports compact JSON database
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

# Add paths for imports - need absolute path to repo root
script_dir = Path(__file__).resolve().parent  # .../analysis
matmul_dir = script_dir.parent                 # .../matmul
compute_dir = matmul_dir.parent                # .../compute
repo_root = compute_dir.parent                 # .../validation-studies

# Add to path
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(matmul_dir))

# Now import
from common.tooling import (
    load_trace,
    op_to_kernel_tracing,
    operation_metadata_scrape,
    get_gpu_op_identifier,
    kernel_metadata_scrape,
)

from matmul_config import generate_experiment_configs


# ============================================================================
# Helper Functions
# ============================================================================

def is_transposed(shape: List[int], stride: List[int]) -> bool:
    """
    Detect if tensor is transposed based on stride pattern.
    
    Row-major (not transposed): last stride == 1
    Column-major (transposed): last stride != 1
    
    Args:
        shape: Tensor shape
        stride: Tensor stride
    
    Returns:
        True if transposed, False otherwise
    """
    if not stride or len(stride) == 0:
        return False
    return stride[-1] != 1


def extract_matmul_operations(cpu_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all matrix multiplication operations from CPU trace.
    
    Returns ONLY top-level aten::matmul operations.
    Note: aten::matmul internally calls aten::mm (2D) or aten::bmm (batched).
    We only want the parent matmul calls that correspond to our actual torch.matmul() calls.
    """
    matmul_ops = []
    
    for node in cpu_trace.get('nodes', []):
        name = node.get('name', '')
        # ONLY extract aten::matmul (the top-level operation we actually call)
        # Skip aten::mm and aten::bmm as they are called internally
        if name == 'aten::matmul':
            matmul_ops.append(node)
    
    return matmul_ops


def extract_op_parameters(cpu_op: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract parameters from a CPU trace operation.
    
    Returns:
        Dictionary with shapes, dtypes, strides, batch_size, transpose flags
    """
    inputs = cpu_op.get('inputs', {})
    shapes = inputs.get('shapes', [])
    types = inputs.get('types', [])
    strides = inputs.get('strides', [])
    
    # Get tensor shapes and strides
    shape_a = shapes[0] if len(shapes) > 0 else []
    shape_b = shapes[1] if len(shapes) > 1 else []
    stride_a = strides[0] if len(strides) > 0 else []
    stride_b = strides[1] if len(strides) > 1 else []
    
    # Get dtype from first tensor
    dtype = types[0] if len(types) > 0 else ""
    
    # Determine batch size
    batch_size = 1
    if len(shape_a) == 3:  # Batched operation
        batch_size = shape_a[0]
    
    # Detect transposes from strides
    transpose_a = is_transposed(shape_a, stride_a)
    transpose_b = is_transposed(shape_b, stride_b)
    
    return {
        'shape_a': shape_a,
        'shape_b': shape_b,
        'stride_a': stride_a,
        'stride_b': stride_b,
        'dtype': dtype,
        'batch_size': batch_size,
        'transpose_a': transpose_a,
        'transpose_b': transpose_b
    }


def validate_match(actual: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that trace operation matches config specification.
    
    Uses config as source of truth, validates stride patterns match expected transposes.
    
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    # Dtype mapping
    dtype_map = {
        'fp32': 'Tensor(float)',
        'fp16': 'Tensor(c10::Half)',
        'bf16': 'Tensor(c10::BFloat16)'
    }
    
    # Validate dtype
    expected_dtype = dtype_map.get(config['dtype'], '')
    if actual['dtype'] != expected_dtype:
        errors.append(f"Dtype mismatch: {actual['dtype']} != {expected_dtype}")
    
    # Validate batch size
    if actual['batch_size'] != config['batch_size']:
        errors.append(f"Batch size mismatch: {actual['batch_size']} != {config['batch_size']}")
    
    # Validate transpose flags using stride pattern
    # Config is source of truth, stride pattern is validation
    if actual['transpose_a'] != config['transpose_a']:
        errors.append(
            f"Transpose A mismatch: stride={actual['stride_a']} "
            f"(detected={actual['transpose_a']}) != config={config['transpose_a']}"
        )
    
    if actual['transpose_b'] != config['transpose_b']:
        errors.append(
            f"Transpose B mismatch: stride={actual['stride_b']} "
            f"(detected={actual['transpose_b']}) != config={config['transpose_b']}"
        )
    
    # Validate dimensions (accounting for batch and transpose)
    # The profiling script creates tensors as A[m, k] and B[k, n], then transposes them.
    # The trace shows the shapes AFTER transpose, so we need to account for that.
    m, n, k = config['m'], config['n'], config['k']
    
    # Expected shapes AFTER transpose (as they appear in the trace)
    if config['batch_size'] == 1:
        # 2D matmul
        if config['transpose_a']:
            expected_a = [k, m]  # Transposed from [m, k]
        else:
            expected_a = [m, k]
        
        if config['transpose_b']:
            expected_b = [n, k]  # Transposed from [k, n]
        else:
            expected_b = [k, n]
    else:
        # 3D batched matmul
        b = config['batch_size']
        if config['transpose_a']:
            expected_a = [b, k, m]  # Transposed from [b, m, k]
        else:
            expected_a = [b, m, k]
        
        if config['transpose_b']:
            expected_b = [b, n, k]  # Transposed from [b, k, n]
        else:
            expected_b = [b, k, n]
    
    # Compare actual shapes (after transpose) with expected shapes (after transpose)
    if actual['shape_a'] != expected_a:
        errors.append(f"Shape A mismatch: {actual['shape_a']} != {expected_a}")
    
    if actual['shape_b'] != expected_b:
        errors.append(f"Shape B mismatch: {actual['shape_b']} != {expected_b}")
    
    return (len(errors) == 0, errors)


def extract_gpu_kernel_data(cpu_op_id: int, cpu_trace: Dict[str, Any], 
                            gpu_trace: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Extract GPU kernel data for a CPU operation.
    
    For aten::matmul, we need to look at child operations (aten::mm or aten::bmm)
    since that's where the actual GPU kernels are launched.
    
    Returns:
        (kernels_data, timing_data) tuple
    """
    # Find the CPU operation
    cpu_op = None
    for node in cpu_trace.get('nodes', []):
        if node.get('id') == cpu_op_id:
            cpu_op = node
            break
    
    if cpu_op is None:
        return [], {'total_kernel_time': None}
    
    # For aten::matmul, we need to find the child mm/bmm operation
    # aten::matmul internally calls aten::mm (2D) or aten::bmm (batched 3D)
    # The kernels are associated with these child operations
    op_name = cpu_op.get('name', '')
    
    if op_name == 'aten::matmul':
        # Find child mm or bmm operation
        # They should be direct children (next operations in the trace)
        child_ops = []
        for node in cpu_trace.get('nodes', []):
            node_name = node.get('name', '')
            if node_name in ['aten::mm', 'aten::bmm']:
                # Check if this is a child by comparing IDs (child should have higher ID)
                # and be close in the trace
                if node.get('id', 0) > cpu_op_id and node.get('id', 0) < cpu_op_id + 100:
                    child_ops.append(node)
        
        # Use the first child operation we find
        if child_ops:
            cpu_op = child_ops[0]
            cpu_op_id = cpu_op['id']
    
    # Now get GPU operations for this operation (or its child)
    gpu_ops = op_to_kernel_tracing(cpu_op_id, cpu_trace, gpu_trace)
    
    # FILTER: Keep only kernel operations
    kernel_ops = [op for op in gpu_ops if op.get('cat') == 'kernel']
    
    # Extract kernel data
    kernels_data = []
    total_kernel_time = 0.0
    
    for kernel_op in kernel_ops:
        identifier = get_gpu_op_identifier(kernel_op)
        metadata = kernel_metadata_scrape(identifier, gpu_trace)
        
        if metadata:
            kernel_info = {
                'name': metadata['name'],
                'duration': metadata['dur'],
                'grid': metadata.get('grid'),
                'block': metadata.get('block'),
                'shared_memory': metadata.get('shared_memory'),
                'registers_per_thread': metadata.get('registers_per_thread'),
                'occupancy': metadata.get('est_achieved_occupancy')
            }
            kernels_data.append(kernel_info)
            total_kernel_time += metadata['dur']
    
    timing_data = {
        'total_kernel_time': total_kernel_time
    }
    
    return kernels_data, timing_data


def compute_statistics(repeats_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute statistics across repeats.
    
    Args:
        repeats_data: List of repeat dictionaries with timing data
    
    Returns:
        Dictionary with mean, std, min, max (or None if no valid data)
    """
    # Extract kernel times, filtering out None values
    kernel_times = [
        r['timing']['total_kernel_time'] 
        for r in repeats_data 
        if 'timing' in r 
        and 'total_kernel_time' in r['timing']
        and r['timing']['total_kernel_time'] is not None
    ]
    
    if not kernel_times:
        return {
            'kernel_time_mean': None,
            'kernel_time_std': None,
            'kernel_time_min': None,
            'kernel_time_max': None
        }
    
    return {
        'kernel_time_mean': float(np.mean(kernel_times)),
        'kernel_time_std': float(np.std(kernel_times)),
        'kernel_time_min': float(np.min(kernel_times)),
        'kernel_time_max': float(np.max(kernel_times))
    }


def get_layout_key(transpose_a: bool, transpose_b: bool) -> str:
    """
    Convert transpose flags to layout key.
    
    Returns: "NN", "NT", "TN", or "TT"
    """
    a = "T" if transpose_a else "N"
    b = "T" if transpose_b else "N"
    return f"{a}{b}"


def build_database(trace_dir: Path, output_path: Path) -> Dict[str, Any]:
    """
    Build the complete database from traces.
    
    Args:
        trace_dir: Directory containing trace files
        output_path: Path to write database JSON
    
    Returns:
        The database dictionary
    """
    print("=" * 80)
    print("MATMUL DATABASE BUILDER")
    print("=" * 80)
    
    # Load configs - prefer saved configs from profiling run, fallback to regenerating
    print("\n[1/8] Loading configurations...")
    configs_file = trace_dir / 'configs.json'
    
    if configs_file.exists():
        print(f"      Loading configs from: {configs_file.name}")
        with open(configs_file, 'r') as f:
            configs = json.load(f)
        print(f"      ✓ Loaded {len(configs)} configurations from saved file")
    else:
        print(f"      No configs.json found, regenerating from matmul_config.py")
        print(f"      ⚠ Warning: Regenerated configs may not match actual run")
        configs = generate_experiment_configs()
        print(f"      Generated {len(configs)} configurations")
    
    
    # Find trace files
    print("\n[2/8] Locating trace files...")
    cpu_trace_path = list(trace_dir.glob("*_CPU_trace.json"))[0]
    gpu_trace_paths = sorted(trace_dir.glob("*.pt.trace.json"))
    
    print(f"      CPU trace: {cpu_trace_path.name}")
    print(f"      GPU traces: {len(gpu_trace_paths)} files")
    
    # Load CPU trace
    print("\n[3/8] Loading CPU trace...")
    cpu_trace = load_trace(str(cpu_trace_path))
    print(f"      Loaded {len(cpu_trace.get('nodes', []))} nodes")
    
    # Extract matmul operations
    print("\n[4/8] Extracting matmul operations...")
    matmul_ops = extract_matmul_operations(cpu_trace)
    print(f"      Found {len(matmul_ops)} matmul operations in CPU trace")
    
    # Calculate how many complete repeats we have in CPU trace
    ops_per_repeat = len(configs)
    complete_repeats = len(matmul_ops) // ops_per_repeat
    partial_ops = len(matmul_ops) % ops_per_repeat
    
    if complete_repeats < len(gpu_trace_paths):
        print(f"      ⚠ CPU trace incomplete:")
        print(f"        - Complete repeats: {complete_repeats}/{len(gpu_trace_paths)}")
        if partial_ops > 0:
            print(f"        - Partial repeat: {partial_ops}/{ops_per_repeat} ops")
        print(f"        - Later repeats will use GPU trace only (may have missing data)")
    
    # Initialize database structure
    database = {
        'metadata': {
            'trace_directory': str(trace_dir),
            'repeat_count': len(gpu_trace_paths),
            'total_configs': len(configs),
            'total_operations': len(matmul_ops)
        },
        'operations': {}
    }
    
    # Read metadata from trace dir if available
    metadata_file = trace_dir / 'metadata.yml'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    database['metadata'][key.strip().lower()] = value.strip()
    
    # Process by repeat (load each GPU trace once)
    print("\n[5/8] Processing traces by repeat...")
    validation_errors = []
    processing_errors = []
    
    # Initialize all config data structures first
    config_data_map = {}  # Maps config_idx to config_data
    
    for config_idx, config in enumerate(configs):
        # Build database keys
        dims_key = f"{config['m']}x{config['n']}x{config['k']}"
        precision_key = config['dtype']
        batch_key = f"batch_{config['batch_size']}"
        layout_key = get_layout_key(config['transpose_a'], config['transpose_b'])
        
        # Initialize nested structure
        if dims_key not in database['operations']:
            database['operations'][dims_key] = {}
        if precision_key not in database['operations'][dims_key]:
            database['operations'][dims_key][precision_key] = {}
        if batch_key not in database['operations'][dims_key][precision_key]:
            database['operations'][dims_key][precision_key][batch_key] = {}
        
        # Store config-level data
        config_data = {
            'config_id': config['config_id'],
            'm': config['m'],
            'n': config['n'],
            'k': config['k'],
            'transpose_a': config['transpose_a'],
            'transpose_b': config['transpose_b'],
            'repeats': [],
            'validation_passed': True  # Will update if validation fails
        }
        
        config_data_map[config_idx] = {
            'data': config_data,
            'keys': (dims_key, precision_key, batch_key, layout_key)
        }
    
    # Process each repeat (load GPU trace once per repeat)
    print("\n[6/8] Processing each repeat...")
    for repeat_idx, gpu_trace_path in enumerate(gpu_trace_paths):
        print(f"\n      Repeat {repeat_idx + 1}/{len(gpu_trace_paths)}: {gpu_trace_path.name}")
        print(f"      Loading GPU trace...")
        
        try:
            gpu_trace = load_trace(str(gpu_trace_path))
            print(f"      ✓ Loaded {len(gpu_trace.get('traceEvents', []))} events")
        except UnicodeDecodeError as e:
            print(f"      ✗ GPU trace appears corrupted (UTF-8 decode error)")
            print(f"      Skipping repeat {repeat_idx + 1}")
            processing_errors.append(f"Repeat {repeat_idx}: Corrupted trace file (UTF-8 decode error)")
            # Add empty repeats for all configs for this repeat
            for config_idx in range(len(configs)):
                repeat_data = {
                    'repeat_id': repeat_idx,
                    'status': 'error',
                    'error_message': 'Corrupted GPU trace file',
                    'kernels': [],
                    'timing': {'total_kernel_time': None}
                }
                config_data_map[config_idx]['data']['repeats'].append(repeat_data)
            continue
        except Exception as e:
            print(f"      ✗ Failed to load GPU trace: {e}")
            processing_errors.append(f"Repeat {repeat_idx}: Failed to load GPU trace: {e}")
            continue
        
        # Process all configs for this repeat
        print(f"      Processing {len(configs)} configs...")
        for config_idx, config in enumerate(configs):
            if config_idx % 200 == 0 and config_idx > 0:
                print(f"        Progress: {config_idx}/{len(configs)} configs...")
            
            try:
                # Get the CPU operation for THIS config and repeat
                cpu_op_idx = config_idx + (repeat_idx * len(configs))
                if cpu_op_idx >= len(matmul_ops):
                    # CPU trace is incomplete for this repeat
                    raise IndexError(f"CPU trace incomplete: op {cpu_op_idx} not found (trace has {len(matmul_ops)} ops)")
                
                cpu_op = matmul_ops[cpu_op_idx]
                
                # Validate parameters (only on first repeat)
                if repeat_idx == 0:
                    actual_params = extract_op_parameters(cpu_op)
                    is_valid, errors = validate_match(actual_params, config)
                    
                    if not is_valid:
                        validation_errors.append({
                            'config_id': config['config_id'],
                            'config_idx': config_idx,
                            'errors': errors
                        })
                        config_data_map[config_idx]['data']['validation_passed'] = False
                        config_data_map[config_idx]['data']['validation_errors'] = errors
                
                # Extract kernel data
                kernels_data, timing_data = extract_gpu_kernel_data(
                    cpu_op['id'], cpu_trace, gpu_trace
                )
                
                repeat_data = {
                    'repeat_id': repeat_idx,
                    'kernels': kernels_data,
                    'timing': timing_data,
                    'status': 'success'
                }
                
            except Exception as e:
                repeat_data = {
                    'repeat_id': repeat_idx,
                    'status': 'error',
                    'error_message': str(e),
                    'kernels': [],
                    'timing': {'total_kernel_time': None}
                }
                processing_errors.append(f"Config {config_idx}, Repeat {repeat_idx}: {str(e)}")
            
            config_data_map[config_idx]['data']['repeats'].append(repeat_data)
        
        print(f"      ✓ Completed repeat {repeat_idx + 1}")
    
    # Compute statistics and store in database
    print("\n[7/8] Computing statistics and finalizing database...")
    for config_idx in range(len(configs)):
        config_entry = config_data_map[config_idx]
        config_data = config_entry['data']
        dims_key, precision_key, batch_key, layout_key = config_entry['keys']
        
        # Compute statistics
        config_data['statistics'] = compute_statistics(config_data['repeats'])
        
        # Store in database
        database['operations'][dims_key][precision_key][batch_key][layout_key] = config_data
    
    print(f"      ✓ Finalized {len(configs)} configurations")
    
    # Report errors
    print("\n[8/8] Validation summary...")
    print(f"      Validation errors: {len(validation_errors)}")
    print(f"      Processing errors: {len(processing_errors)}")
    
    if validation_errors:
        print("\n      First 5 validation errors:")
        for error in validation_errors[:5]:
            print(f"        Config {error['config_id']}: {error['errors']}")
    
    if processing_errors:
        print("\n      First 5 processing errors:")
        for error in processing_errors[:5]:
            print(f"        {error}")
    
    # Write database
    print(f"\n[9/9] Writing database to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(database, f, separators=(',', ':'), indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"      Database written: {file_size_mb:.2f} MB")
    
    # Write error log if there are errors
    if validation_errors or processing_errors:
        error_log_path = output_path.parent / 'database_errors.log'
        with open(error_log_path, 'w') as f:
            f.write("VALIDATION ERRORS\n")
            f.write("=" * 80 + "\n\n")
            for error in validation_errors:
                f.write(f"Config {error['config_id']}:\n")
                for err in error['errors']:
                    f.write(f"  - {err}\n")
                f.write("\n")
            
            f.write("\nPROCESSING ERRORS\n")
            f.write("=" * 80 + "\n\n")
            for error in processing_errors:
                f.write(f"{error}\n")
        
        print(f"      Error log written: {error_log_path}")
    
    print("\n" + "=" * 80)
    print("DATABASE BUILD COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Configurations: {len(configs)}")
    print(f"  - Repeats per config: {len(gpu_trace_paths)}")
    print(f"  - Validation errors: {len(validation_errors)}")
    print(f"  - Processing errors: {len(processing_errors)}")
    print(f"  - Output: {output_path}")
    
    return database


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build matmul operation database from traces'
    )
    parser.add_argument(
        'trace_dir',
        type=str,
        help='Directory containing trace files (e.g., traces/A100/A100-580.95.05-12.4/)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for database JSON (default: ../database/<trace_name>_database.json)'
    )
    
    args = parser.parse_args()
    
    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists():
        print(f"Error: Trace directory not found: {trace_dir}")
        return 1
    
    # Default output to ../database/<trace_name>_database.json
    if args.output:
        output_path = Path(args.output)
    else:
        database_dir = Path(__file__).parent.parent / 'database'
        database_dir.mkdir(exist_ok=True)
        trace_name = trace_dir.name  # e.g., "A100-580.95.05-12.4"
        output_path = database_dir / f"{trace_name}_database.json"
    
    try:
        build_database(trace_dir, output_path)
        return 0
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

