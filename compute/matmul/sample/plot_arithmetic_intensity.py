#!/usr/bin/env python3
"""
Script to parse T4 test run database and create plots.
First plot: Arithmetic Intensity vs Throughput
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# Precision to bytes mapping
PRECISION_BYTES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'fp64': 8,
    'int8': 1,
    'int32': 4,
}


def calculate_flops(m: int, n: int, k: int) -> float:
    """
    Calculate FLOPs using the 2MNK heuristic.
    
    Args:
        m, n, k: Matrix multiplication dimensions (M x K) @ (K x N) = (M x N)
    
    Returns:
        Total FLOPs
    """
    return 2 * m * n * k


def calculate_bytes(m: int, n: int, k: int, precision: str) -> float:
    """
    Calculate total bytes accessed for matrix multiplication.
    
    Args:
        m, n, k: Matrix multiplication dimensions
        precision: Data precision (fp32, fp16, etc.)
    
    Returns:
        Total bytes accessed
    """
    bytes_per_element = PRECISION_BYTES.get(precision, 4)
    
    # Matrix A: M x K
    # Matrix B: K x N
    # Matrix C: M x N
    total_elements = m * k + k * n + m * n
    
    return total_elements * bytes_per_element


def calculate_arithmetic_intensity(m: int, n: int, k: int, precision: str) -> float:
    """
    Calculate arithmetic intensity (FLOPs/Byte).
    
    Args:
        m, n, k: Matrix multiplication dimensions
        precision: Data precision
    
    Returns:
        Arithmetic intensity in FLOPs/Byte
    """
    flops = calculate_flops(m, n, k)
    bytes_accessed = calculate_bytes(m, n, k, precision)
    
    return flops / bytes_accessed


def calculate_throughput(m: int, n: int, k: int, time_us: float) -> float:
    """
    Calculate throughput in TFLOP/s.
    
    Args:
        m, n, k: Matrix multiplication dimensions
        time_us: Execution time in microseconds
    
    Returns:
        Throughput in TFLOP/s
    """
    flops = calculate_flops(m, n, k)
    time_s = time_us / 1e6  # Convert microseconds to seconds
    tflops = flops / time_s / 1e12  # Convert to TFLOP/s
    
    return tflops


def parse_database(json_path: str) -> List[Dict]:
    """
    Parse the database JSON and extract all data points.
    
    Args:
        json_path: Path to the JSON database file
    
    Returns:
        List of data points with computed metrics
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_points = []
    
    # Navigate through the nested structure
    operations = data.get('operations', {})
    
    for size_key, size_data in operations.items():
        for precision, precision_data in size_data.items():
            for batch_key, batch_data in precision_data.items():
                for transpose_key, config in batch_data.items():
                    # Extract configuration
                    m = config['m']
                    n = config['n']
                    k = config['k']
                    transpose_a = config['transpose_a']
                    transpose_b = config['transpose_b']
                    config_id = config['config_id']
                    
                    # Calculate metrics that are constant for this config
                    arithmetic_intensity = calculate_arithmetic_intensity(m, n, k, precision)
                    
                    # Process each repeat (individual instance)
                    for repeat in config.get('repeats', []):
                        if repeat.get('status') != 'success':
                            continue
                        
                        timing = repeat.get('timing', {})
                        total_kernel_time = timing.get('total_kernel_time')
                        
                        if total_kernel_time is None or total_kernel_time <= 0:
                            continue
                        
                        # Calculate throughput for this specific instance
                        throughput = calculate_throughput(m, n, k, total_kernel_time)
                        
                        # Store data point
                        data_points.append({
                            'config_id': config_id,
                            'size_key': size_key,
                            'm': m,
                            'n': n,
                            'k': k,
                            'precision': precision,
                            'batch': batch_key,
                            'transpose_a': transpose_a,
                            'transpose_b': transpose_b,
                            'transpose_key': transpose_key,
                            'repeat_id': repeat['repeat_id'],
                            'total_kernel_time_us': total_kernel_time,
                            'arithmetic_intensity': arithmetic_intensity,
                            'throughput_tflops': throughput,
                            'flops': calculate_flops(m, n, k),
                            'bytes': calculate_bytes(m, n, k, precision),
                        })
    
    return data_points


def plot_arithmetic_intensity_vs_throughput(data_points: List[Dict], output_dir: str):
    """
    Create scatter plot of arithmetic intensity vs throughput.
    
    Args:
        data_points: List of parsed data points
        output_dir: Directory to save the plot
    """
    # T4 GPU specifications
    MEMORY_BANDWIDTH_GBS = 320  # GB/s
    PEAK_COMPUTE_FP32_TFLOPS = 8.141  # TFLOPS
    PEAK_COMPUTE_FP16_TFLOPS = 65.13  # TFLOPS
    
    # Convert memory bandwidth to TFLOP/s per FLOPs/Byte
    # 320 GB/s = 0.32 TFLOP/s per (FLOPs/Byte)
    memory_bandwidth_tflops_per_ai = MEMORY_BANDWIDTH_GBS / 1000  # 0.32
    
    # Filter out bf16 (T4 doesn't process bf16 correctly)
    filtered_points = [dp for dp in data_points if dp['precision'] != 'bf16']
    
    # Extract data for plotting
    ai_values = [dp['arithmetic_intensity'] for dp in filtered_points]
    throughput_values = [dp['throughput_tflops'] for dp in filtered_points]
    precisions = [dp['precision'] for dp in filtered_points]
    
    # Create color mapping for different precisions using magma
    # Using closer colors in the magma spectrum (0.3 to 0.85)
    unique_precisions = sorted(set(precisions))
    colors = plt.cm.magma(np.linspace(0.3, 0.85, len(unique_precisions)))
    precision_color_map = {prec: colors[i] for i, prec in enumerate(unique_precisions)}
    
    # Roofline colors from magma (matching FP32 and FP16 precision colors)
    roofline_color_fp32 = precision_color_map['fp32'] if 'fp32' in precision_color_map else plt.cm.magma(0.2)
    roofline_color_fp16 = precision_color_map['fp16'] if 'fp16' in precision_color_map else plt.cm.magma(0.9)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot each precision separately for legend
    for precision in unique_precisions:
        mask = [dp['precision'] == precision for dp in filtered_points]
        ai_prec = [ai for ai, m in zip(ai_values, mask) if m]
        throughput_prec = [tp for tp, m in zip(throughput_values, mask) if m]
        
        plt.scatter(ai_prec, throughput_prec, 
                   alpha=0.6, 
                   s=30,
                   color=precision_color_map[precision],
                   label=precision,
                   edgecolors='black',
                   linewidths=0.5)
    
    # Add roofline model lines
    ai_range = np.array([min(ai_values), max(ai_values)])
    
    # FP32 roofline
    ridge_point_fp32 = PEAK_COMPUTE_FP32_TFLOPS / memory_bandwidth_tflops_per_ai
    memory_bound_fp32 = ai_range * memory_bandwidth_tflops_per_ai
    memory_bound_fp32 = np.minimum(memory_bound_fp32, PEAK_COMPUTE_FP32_TFLOPS)
    plt.plot([0, ridge_point_fp32, max(ai_values)], 
             [0, PEAK_COMPUTE_FP32_TFLOPS, PEAK_COMPUTE_FP32_TFLOPS],
             '--', color=roofline_color_fp32, linewidth=2.5, label='FP32 Roofline', alpha=0.9)
    
    # FP16 roofline
    ridge_point_fp16 = PEAK_COMPUTE_FP16_TFLOPS / memory_bandwidth_tflops_per_ai
    memory_bound_fp16 = ai_range * memory_bandwidth_tflops_per_ai
    memory_bound_fp16 = np.minimum(memory_bound_fp16, PEAK_COMPUTE_FP16_TFLOPS)
    plt.plot([0, ridge_point_fp16, max(ai_values)], 
             [0, PEAK_COMPUTE_FP16_TFLOPS, PEAK_COMPUTE_FP16_TFLOPS],
             '--', color=roofline_color_fp16, linewidth=2.5, label='FP16 Roofline', alpha=0.9)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title('Arithmetic Intensity vs Throughput - T4 GPU\n(All Individual Instances)', fontsize=14)
    plt.legend(title='Precision', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / 'arithmetic_intensity_vs_throughput.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Also create log-scale version
    plt.figure(figsize=(12, 8))
    
    for precision in unique_precisions:
        mask = [dp['precision'] == precision for dp in filtered_points]
        ai_prec = [ai for ai, m in zip(ai_values, mask) if m]
        throughput_prec = [tp for tp, m in zip(throughput_values, mask) if m]
        
        plt.scatter(ai_prec, throughput_prec, 
                   alpha=0.6, 
                   s=30,
                   color=precision_color_map[precision],
                   label=precision,
                   edgecolors='black',
                   linewidths=0.5)
    
    # Add roofline model lines for log scale
    ai_range_log = np.logspace(np.log10(min(ai_values)), np.log10(max(ai_values)), 100)
    
    # FP32 roofline
    ridge_point_fp32 = PEAK_COMPUTE_FP32_TFLOPS / memory_bandwidth_tflops_per_ai
    memory_bound_fp32_log = ai_range_log * memory_bandwidth_tflops_per_ai
    throughput_fp32_log = np.minimum(memory_bound_fp32_log, PEAK_COMPUTE_FP32_TFLOPS)
    plt.plot(ai_range_log, throughput_fp32_log, 
             '--', color=roofline_color_fp32, linewidth=2.5, label='FP32 Roofline', alpha=0.9)
    
    # FP16 roofline
    ridge_point_fp16 = PEAK_COMPUTE_FP16_TFLOPS / memory_bandwidth_tflops_per_ai
    memory_bound_fp16_log = ai_range_log * memory_bandwidth_tflops_per_ai
    throughput_fp16_log = np.minimum(memory_bound_fp16_log, PEAK_COMPUTE_FP16_TFLOPS)
    plt.plot(ai_range_log, throughput_fp16_log, 
             '--', color=roofline_color_fp16, linewidth=2.5, label='FP16 Roofline', alpha=0.9)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title('Arithmetic Intensity vs Throughput - T4 GPU (Log-Log Scale)\n(All Individual Instances)', fontsize=14)
    plt.legend(title='Precision', fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    # Save log-scale plot
    output_path_log = Path(output_dir) / 'arithmetic_intensity_vs_throughput_log.png'
    plt.savefig(output_path_log, dpi=300, bbox_inches='tight')
    print(f"Saved log-scale plot to: {output_path_log}")
    
    plt.close('all')


def print_summary_statistics(data_points: List[Dict]):
    """
    Print summary statistics about the data.
    
    Args:
        data_points: List of parsed data points
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total data points: {len(data_points)}")
    
    # Group by precision
    precisions = {}
    for dp in data_points:
        prec = dp['precision']
        if prec not in precisions:
            precisions[prec] = []
        precisions[prec].append(dp)
    
    print(f"\nData points by precision:")
    for prec in sorted(precisions.keys()):
        count = len(precisions[prec])
        ai_values = [dp['arithmetic_intensity'] for dp in precisions[prec]]
        throughput_values = [dp['throughput_tflops'] for dp in precisions[prec]]
        
        print(f"  {prec:8s}: {count:5d} points")
        print(f"    AI range: {min(ai_values):.2f} - {max(ai_values):.2f} FLOPs/Byte")
        print(f"    Throughput range: {min(throughput_values):.3f} - {max(throughput_values):.3f} TFLOP/s")
    
    # Overall statistics
    all_ai = [dp['arithmetic_intensity'] for dp in data_points]
    all_throughput = [dp['throughput_tflops'] for dp in data_points]
    
    print(f"\nOverall statistics:")
    print(f"  Arithmetic Intensity:")
    print(f"    Min: {min(all_ai):.2f} FLOPs/Byte")
    print(f"    Max: {max(all_ai):.2f} FLOPs/Byte")
    print(f"    Mean: {np.mean(all_ai):.2f} FLOPs/Byte")
    print(f"    Median: {np.median(all_ai):.2f} FLOPs/Byte")
    
    print(f"  Throughput:")
    print(f"    Min: {min(all_throughput):.3f} TFLOP/s")
    print(f"    Max: {max(all_throughput):.3f} TFLOP/s")
    print(f"    Mean: {np.mean(all_throughput):.3f} TFLOP/s")
    print(f"    Median: {np.median(all_throughput):.3f} TFLOP/s")
    
    print("="*60 + "\n")


def main():
    """Main execution function."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    database_path = project_root / 'database' / 'T4_run_fixed_database.json'
    output_dir = script_dir
    
    print(f"Reading database from: {database_path}")
    
    # Parse the database
    data_points = parse_database(str(database_path))
    
    print(f"Parsed {len(data_points)} data points")
    
    # Print summary statistics
    print_summary_statistics(data_points)
    
    # Create plots
    print("Creating plots...")
    plot_arithmetic_intensity_vs_throughput(data_points, str(output_dir))
    
    print("\nDone!")


if __name__ == '__main__':
    main()

