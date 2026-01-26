#!/usr/bin/env python3
"""
Script to plot arithmetic intensity vs throughput with means and maxes per config.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


# Precision to bytes mapping
PRECISION_BYTES = {
    'fp32': 4,
    'fp16': 2,
    'bf16': 2,
    'fp64': 8,
    'int8': 1,
    'int32': 4,
}

# T4 GPU specifications
MEMORY_BANDWIDTH_GBS = 320  # GB/s
PEAK_COMPUTE_FP32_TFLOPS = 8.141  # TFLOPS
PEAK_COMPUTE_FP16_TFLOPS = 65.13  # TFLOPS


def calculate_flops(m: int, n: int, k: int) -> float:
    """Calculate FLOPs using the 2MNK heuristic."""
    return 2 * m * n * k


def calculate_bytes(m: int, n: int, k: int, precision: str) -> float:
    """Calculate total bytes accessed for matrix multiplication."""
    bytes_per_element = PRECISION_BYTES.get(precision, 4)
    total_elements = m * k + k * n + m * n
    return total_elements * bytes_per_element


def calculate_arithmetic_intensity(m: int, n: int, k: int, precision: str) -> float:
    """Calculate arithmetic intensity (FLOPs/Byte)."""
    flops = calculate_flops(m, n, k)
    bytes_accessed = calculate_bytes(m, n, k, precision)
    return flops / bytes_accessed


def calculate_throughput(m: int, n: int, k: int, time_us: float) -> float:
    """Calculate throughput in TFLOP/s."""
    flops = calculate_flops(m, n, k)
    time_s = time_us / 1e6
    tflops = flops / time_s / 1e12
    return tflops


def parse_database(json_path: str) -> Dict:
    """
    Parse the database JSON and group by configuration.
    
    Returns:
        Dictionary with config_key -> list of throughput values
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Group data by configuration
    configs = defaultdict(list)
    
    operations = data.get('operations', {})
    
    for size_key, size_data in operations.items():
        for precision, precision_data in size_data.items():
            # Skip bf16
            if precision == 'bf16':
                continue
                
            for batch_key, batch_data in precision_data.items():
                for transpose_key, config in batch_data.items():
                    m = config['m']
                    n = config['n']
                    k = config['k']
                    transpose_a = config['transpose_a']
                    transpose_b = config['transpose_b']
                    config_id = config['config_id']
                    
                    arithmetic_intensity = calculate_arithmetic_intensity(m, n, k, precision)
                    
                    # Collect all throughput values for this config
                    throughputs = []
                    for repeat in config.get('repeats', []):
                        if repeat.get('status') != 'success':
                            continue
                        
                        timing = repeat.get('timing', {})
                        total_kernel_time = timing.get('total_kernel_time')
                        
                        if total_kernel_time is None or total_kernel_time <= 0:
                            continue
                        
                        throughput = calculate_throughput(m, n, k, total_kernel_time)
                        throughputs.append(throughput)
                    
                    if throughputs:
                        config_key = (config_id, m, n, k, precision, transpose_a, transpose_b, arithmetic_intensity)
                        configs[config_key] = throughputs
    
    return configs


def plot_means_and_maxes(configs: Dict, output_dir: str):
    """
    Create scatter plot with means and maxes per config.
    
    Args:
        configs: Dictionary of config -> throughput values
        output_dir: Directory to save the plot
    """
    memory_bandwidth_tflops_per_ai = MEMORY_BANDWIDTH_GBS / 1000  # 0.32
    
    # Separate by precision and compute statistics
    fp32_means = []
    fp32_maxes = []
    fp32_ai = []
    
    fp16_means = []
    fp16_maxes = []
    fp16_ai = []
    
    for config_key, throughputs in configs.items():
        config_id, m, n, k, precision, transpose_a, transpose_b, ai = config_key
        
        mean_throughput = np.mean(throughputs)
        max_throughput = np.max(throughputs)
        
        if precision == 'fp32':
            fp32_ai.append(ai)
            fp32_means.append(mean_throughput)
            fp32_maxes.append(max_throughput)
        elif precision == 'fp16':
            fp16_ai.append(ai)
            fp16_means.append(mean_throughput)
            fp16_maxes.append(max_throughput)
    
    # Colors from magma
    # Means: darker colors
    color_fp32_mean = plt.cm.magma(0.25)
    color_fp16_mean = plt.cm.magma(0.65)
    # Maxes: brighter colors
    color_fp32_max = plt.cm.magma(0.45)
    color_fp16_max = plt.cm.magma(0.90)
    
    # --- Linear Scale Plot ---
    plt.figure(figsize=(14, 9))
    
    # Plot means
    plt.scatter(fp32_ai, fp32_means, alpha=0.7, s=50, color=color_fp32_mean, 
                label='FP32 Mean', edgecolors='black', linewidths=0.5, zorder=3, marker='o')
    plt.scatter(fp16_ai, fp16_means, alpha=0.7, s=50, color=color_fp16_mean, 
                label='FP16 Mean', edgecolors='black', linewidths=0.5, zorder=3, marker='o')
    
    # Plot maxes
    plt.scatter(fp32_ai, fp32_maxes, alpha=0.7, s=50, color=color_fp32_max, 
                label='FP32 Max', edgecolors='black', linewidths=0.5, zorder=4, marker='^')
    plt.scatter(fp16_ai, fp16_maxes, alpha=0.7, s=50, color=color_fp16_max, 
                label='FP16 Max', edgecolors='black', linewidths=0.5, zorder=4, marker='^')
    
    # Add roofline model lines
    ai_range = np.array([min(min(fp32_ai), min(fp16_ai)), max(max(fp32_ai), max(fp16_ai))])
    
    # FP32 roofline
    ridge_point_fp32 = PEAK_COMPUTE_FP32_TFLOPS / memory_bandwidth_tflops_per_ai
    plt.plot([0, ridge_point_fp32, max(ai_range)], 
             [0, PEAK_COMPUTE_FP32_TFLOPS, PEAK_COMPUTE_FP32_TFLOPS],
             '--', color=color_fp32_mean, linewidth=2.5, label='FP32 Roofline', alpha=0.9, zorder=2)
    
    # FP16 roofline
    ridge_point_fp16 = PEAK_COMPUTE_FP16_TFLOPS / memory_bandwidth_tflops_per_ai
    plt.plot([0, ridge_point_fp16, max(ai_range)], 
             [0, PEAK_COMPUTE_FP16_TFLOPS, PEAK_COMPUTE_FP16_TFLOPS],
             '--', color=color_fp16_mean, linewidth=2.5, label='FP16 Roofline', alpha=0.9, zorder=2)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title('Arithmetic Intensity vs Throughput - T4 GPU\n(Mean and Max per Configuration)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'arithmetic_intensity_means_maxes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # --- Log Scale Plot ---
    plt.figure(figsize=(14, 9))
    
    # Plot means
    plt.scatter(fp32_ai, fp32_means, alpha=0.7, s=50, color=color_fp32_mean, 
                label='FP32 Mean', edgecolors='black', linewidths=0.5, zorder=3, marker='o')
    plt.scatter(fp16_ai, fp16_means, alpha=0.7, s=50, color=color_fp16_mean, 
                label='FP16 Mean', edgecolors='black', linewidths=0.5, zorder=3, marker='o')
    
    # Plot maxes
    plt.scatter(fp32_ai, fp32_maxes, alpha=0.7, s=50, color=color_fp32_max, 
                label='FP32 Max', edgecolors='black', linewidths=0.5, zorder=4, marker='^')
    plt.scatter(fp16_ai, fp16_maxes, alpha=0.7, s=50, color=color_fp16_max, 
                label='FP16 Max', edgecolors='black', linewidths=0.5, zorder=4, marker='^')
    
    # Add roofline model lines for log scale
    ai_range_log = np.logspace(np.log10(min(min(fp32_ai), min(fp16_ai))), 
                                np.log10(max(max(fp32_ai), max(fp16_ai))), 1000)
    
    # FP32 roofline
    memory_bound_fp32_log = ai_range_log * memory_bandwidth_tflops_per_ai
    throughput_fp32_log = np.minimum(memory_bound_fp32_log, PEAK_COMPUTE_FP32_TFLOPS)
    plt.plot(ai_range_log, throughput_fp32_log, 
             '--', color=color_fp32_mean, linewidth=2.5, label='FP32 Roofline', alpha=0.9, zorder=2)
    
    # FP16 roofline
    memory_bound_fp16_log = ai_range_log * memory_bandwidth_tflops_per_ai
    throughput_fp16_log = np.minimum(memory_bound_fp16_log, PEAK_COMPUTE_FP16_TFLOPS)
    plt.plot(ai_range_log, throughput_fp16_log, 
             '--', color=color_fp16_mean, linewidth=2.5, label='FP16 Roofline', alpha=0.9, zorder=2)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title('Arithmetic Intensity vs Throughput - T4 GPU (Log Scale)\n(Mean and Max per Configuration)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right', ncol=2)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    output_path_log = Path(output_dir) / 'arithmetic_intensity_means_maxes_log.png'
    plt.savefig(output_path_log, dpi=300, bbox_inches='tight')
    print(f"Saved log-scale plot to: {output_path_log}")
    
    plt.close('all')


def print_summary_statistics(configs: Dict):
    """Print summary statistics about the data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"Total configurations: {len(configs)}")
    
    # Group by precision
    fp32_configs = [k for k in configs.keys() if k[4] == 'fp32']
    fp16_configs = [k for k in configs.keys() if k[4] == 'fp16']
    
    print(f"\nConfigurations by precision:")
    print(f"  FP32: {len(fp32_configs)} configs")
    print(f"  FP16: {len(fp16_configs)} configs")
    
    # Overall statistics for means
    all_fp32_means = [np.mean(configs[k]) for k in fp32_configs]
    all_fp16_means = [np.mean(configs[k]) for k in fp16_configs]
    
    all_fp32_maxes = [np.max(configs[k]) for k in fp32_configs]
    all_fp16_maxes = [np.max(configs[k]) for k in fp16_configs]
    
    print(f"\nFP32 Mean Throughput:")
    print(f"  Min: {min(all_fp32_means):.3f} TFLOP/s")
    print(f"  Max: {max(all_fp32_means):.3f} TFLOP/s")
    print(f"  Average: {np.mean(all_fp32_means):.3f} TFLOP/s")
    
    print(f"\nFP32 Max Throughput:")
    print(f"  Min: {min(all_fp32_maxes):.3f} TFLOP/s")
    print(f"  Max: {max(all_fp32_maxes):.3f} TFLOP/s")
    print(f"  Average: {np.mean(all_fp32_maxes):.3f} TFLOP/s")
    
    print(f"\nFP16 Mean Throughput:")
    print(f"  Min: {min(all_fp16_means):.3f} TFLOP/s")
    print(f"  Max: {max(all_fp16_means):.3f} TFLOP/s")
    print(f"  Average: {np.mean(all_fp16_means):.3f} TFLOP/s")
    
    print(f"\nFP16 Max Throughput:")
    print(f"  Min: {min(all_fp16_maxes):.3f} TFLOP/s")
    print(f"  Max: {max(all_fp16_maxes):.3f} TFLOP/s")
    print(f"  Average: {np.mean(all_fp16_maxes):.3f} TFLOP/s")
    
    print("="*60 + "\n")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    database_path = project_root / 'database' / 'T4_run_fixed_database.json'
    output_dir = script_dir
    
    print(f"Reading database from: {database_path}")
    
    # Parse the database and group by configuration
    configs = parse_database(str(database_path))
    
    print(f"Parsed {len(configs)} configurations")
    
    # Print summary statistics
    print_summary_statistics(configs)
    
    # Create plots
    print("Creating plots...")
    plot_means_and_maxes(configs, str(output_dir))
    
    print("\nDone!")


if __name__ == '__main__':
    main()

