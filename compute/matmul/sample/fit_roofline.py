#!/usr/bin/env python3
"""
Fit roofline model parameters using scipy.optimize with actual roofline model.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
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

# Theoretical T4 specifications
THEORETICAL_MEMORY_BANDWIDTH_GBS = 320  # GB/s
THEORETICAL_PEAK_FP32_TFLOPS = 8.141  # TFLOPS
THEORETICAL_PEAK_FP16_TFLOPS = 65.13  # TFLOPS


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


def parse_database(json_path: str) -> List[Dict]:
    """Parse the database JSON and extract all data points."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    data_points = []
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
                    
                    arithmetic_intensity = calculate_arithmetic_intensity(m, n, k, precision)
                    
                    for repeat in config.get('repeats', []):
                        if repeat.get('status') != 'success':
                            continue
                        
                        timing = repeat.get('timing', {})
                        total_kernel_time = timing.get('total_kernel_time')
                        
                        if total_kernel_time is None or total_kernel_time <= 0:
                            continue
                        
                        throughput = calculate_throughput(m, n, k, total_kernel_time)
                        
                        data_points.append({
                            'precision': precision,
                            'arithmetic_intensity': arithmetic_intensity,
                            'throughput_tflops': throughput,
                        })
    
    return data_points


def roofline_model(ai: np.ndarray, memory_bandwidth: float, peak_compute: float) -> np.ndarray:
    """
    Actual roofline model with sharp corner.
    
    Args:
        ai: Arithmetic intensity (FLOPs/Byte)
        memory_bandwidth: Memory bandwidth in TFLOP/s per (FLOPs/Byte)
        peak_compute: Peak compute in TFLOP/s
    
    Returns:
        Throughput in TFLOP/s
    """
    return np.minimum(memory_bandwidth * ai, peak_compute)


def huber_loss(residuals: np.ndarray, delta: float = 1.0) -> float:
    """
    Huber loss function (robust to outliers).
    
    Args:
        residuals: Prediction errors
        delta: Threshold for switching from quadratic to linear
    
    Returns:
        Total Huber loss
    """
    abs_residuals = np.abs(residuals)
    quadratic = abs_residuals <= delta
    linear = abs_residuals > delta
    
    loss = np.sum(0.5 * residuals[quadratic]**2)
    loss += np.sum(delta * (abs_residuals[linear] - 0.5 * delta))
    
    return loss


def objective_function(params: np.ndarray, data_points: List[Dict], delta: float = 2.0) -> float:
    """
    Objective function for optimization using Huber loss.
    
    Args:
        params: [memory_bandwidth, peak_compute_fp32, peak_compute_fp16]
        data_points: List of data points
        delta: Huber loss threshold (default 2.0 TFLOP/s)
    
    Returns:
        Total loss
    """
    memory_bandwidth = params[0]
    peak_compute_fp32 = params[1]
    peak_compute_fp16 = params[2]
    
    # Separate data by precision
    fp32_points = [dp for dp in data_points if dp['precision'] == 'fp32']
    fp16_points = [dp for dp in data_points if dp['precision'] == 'fp16']
    
    residuals = []
    
    # FP32 residuals
    if fp32_points:
        ai_fp32 = np.array([dp['arithmetic_intensity'] for dp in fp32_points])
        throughput_fp32 = np.array([dp['throughput_tflops'] for dp in fp32_points])
        predicted_fp32 = roofline_model(ai_fp32, memory_bandwidth, peak_compute_fp32)
        residuals.extend(throughput_fp32 - predicted_fp32)
    
    # FP16 residuals
    if fp16_points:
        ai_fp16 = np.array([dp['arithmetic_intensity'] for dp in fp16_points])
        throughput_fp16 = np.array([dp['throughput_tflops'] for dp in fp16_points])
        predicted_fp16 = roofline_model(ai_fp16, memory_bandwidth, peak_compute_fp16)
        residuals.extend(throughput_fp16 - predicted_fp16)
    
    residuals = np.array(residuals)
    return huber_loss(residuals, delta=delta)


def fit_roofline_parameters(data_points: List[Dict], method: str = 'Powell') -> Tuple[np.ndarray, float]:
    """
    Fit roofline parameters using scipy.optimize.
    
    Args:
        data_points: List of data points
        method: Optimization method ('Powell' or 'Nelder-Mead')
    
    Returns:
        Tuple of (fitted_parameters, final_loss)
    """
    # Initial guess (start from theoretical values)
    x0 = np.array([
        THEORETICAL_MEMORY_BANDWIDTH_GBS / 1000,  # 0.32 TFLOP/s per FLOPs/Byte
        THEORETICAL_PEAK_FP32_TFLOPS,             # 8.141 TFLOP/s
        THEORETICAL_PEAK_FP16_TFLOPS,             # 65.13 TFLOP/s
    ])
    
    # Reasonable bounds - allow wide range for best fit
    bounds = [
        (0.01, 1.0),      # Memory bandwidth: 10 GB/s to 1 TB/s
        (0.5, 20.0),      # FP32 peak: 0.5 to 20 TFLOP/s
        (1.0, 100.0),     # FP16 peak: 1 to 100 TFLOP/s
    ]
    
    print(f"\nOptimizing with method: {method}")
    print(f"Initial parameters: memory_bw={x0[0]:.4f}, fp32_peak={x0[1]:.3f}, fp16_peak={x0[2]:.3f}")
    print(f"Initial loss: {objective_function(x0, data_points):.4f}")
    
    result = minimize(
        objective_function,
        x0,
        args=(data_points,),
        method=method,
        bounds=bounds,
        options={'maxiter': 50000, 'disp': True}
    )
    
    print(f"Optimization complete!")
    print(f"Final parameters: memory_bw={result.x[0]:.4f}, fp32_peak={result.x[1]:.3f}, fp16_peak={result.x[2]:.3f}")
    print(f"Final loss: {result.fun:.4f}")
    print(f"Success: {result.success}")
    
    return result.x, result.fun


def plot_fitted_roofline(data_points: List[Dict], fitted_params: np.ndarray, 
                         method: str, output_dir: str):
    """
    Create plots showing data with theoretical and fitted rooflines.
    
    Args:
        data_points: List of data points
        fitted_params: Fitted parameters [memory_bw, peak_fp32, peak_fp16]
        method: Optimization method used
        output_dir: Directory to save plots
    """
    memory_bw_fitted = fitted_params[0]
    peak_fp32_fitted = fitted_params[1]
    peak_fp16_fitted = fitted_params[2]
    
    memory_bw_theoretical = THEORETICAL_MEMORY_BANDWIDTH_GBS / 1000
    
    # Separate data by precision
    fp32_points = [dp for dp in data_points if dp['precision'] == 'fp32']
    fp16_points = [dp for dp in data_points if dp['precision'] == 'fp16']
    
    ai_fp32 = np.array([dp['arithmetic_intensity'] for dp in fp32_points])
    throughput_fp32 = np.array([dp['throughput_tflops'] for dp in fp32_points])
    
    ai_fp16 = np.array([dp['arithmetic_intensity'] for dp in fp16_points])
    throughput_fp16 = np.array([dp['throughput_tflops'] for dp in fp16_points])
    
    # Colors from magma
    color_fp32 = plt.cm.magma(0.3)
    color_fp16 = plt.cm.magma(0.85)
    
    # Create range for roofline plotting
    ai_range = np.linspace(1, max(ai_fp16.max(), ai_fp32.max()), 1000)
    
    # --- Linear Scale Plot ---
    plt.figure(figsize=(14, 9))
    
    # Plot data points
    plt.scatter(ai_fp32, throughput_fp32, alpha=0.6, s=30, color=color_fp32, 
                label='FP32 Data', edgecolors='black', linewidths=0.5, zorder=3)
    plt.scatter(ai_fp16, throughput_fp16, alpha=0.6, s=30, color=color_fp16, 
                label='FP16 Data', edgecolors='black', linewidths=0.5, zorder=3)
    
    # Plot theoretical rooflines (dotted, lighter)
    theoretical_fp32 = roofline_model(ai_range, memory_bw_theoretical, THEORETICAL_PEAK_FP32_TFLOPS)
    theoretical_fp16 = roofline_model(ai_range, memory_bw_theoretical, THEORETICAL_PEAK_FP16_TFLOPS)
    plt.plot(ai_range, theoretical_fp32, ':', color=color_fp32, linewidth=2, 
             alpha=0.5, label='FP32 Theoretical Roofline', zorder=2)
    plt.plot(ai_range, theoretical_fp16, ':', color=color_fp16, linewidth=2, 
             alpha=0.5, label='FP16 Theoretical Roofline', zorder=2)
    
    # Plot fitted rooflines (solid)
    fitted_fp32 = roofline_model(ai_range, memory_bw_fitted, peak_fp32_fitted)
    fitted_fp16 = roofline_model(ai_range, memory_bw_fitted, peak_fp16_fitted)
    plt.plot(ai_range, fitted_fp32, '-', color=color_fp32, linewidth=2.5, 
             alpha=0.9, label=f'FP32 Fitted Roofline ({method})', zorder=4)
    plt.plot(ai_range, fitted_fp16, '-', color=color_fp16, linewidth=2.5, 
             alpha=0.9, label=f'FP16 Fitted Roofline ({method})', zorder=4)
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title(f'Fitted Roofline Model - T4 GPU ({method} Optimizer)\n' + 
              f'Fitted: BW={memory_bw_fitted*1000:.1f} GB/s, ' +
              f'FP32={peak_fp32_fitted:.2f} TFLOP/s, FP16={peak_fp16_fitted:.2f} TFLOP/s',
              fontsize=13)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / f'fitted_roofline_{method.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved linear plot to: {output_path}")
    
    # --- Log Scale Plot ---
    plt.figure(figsize=(14, 9))
    
    # Plot data points
    plt.scatter(ai_fp32, throughput_fp32, alpha=0.6, s=30, color=color_fp32, 
                label='FP32 Data', edgecolors='black', linewidths=0.5, zorder=3)
    plt.scatter(ai_fp16, throughput_fp16, alpha=0.6, s=30, color=color_fp16, 
                label='FP16 Data', edgecolors='black', linewidths=0.5, zorder=3)
    
    # Plot theoretical rooflines
    plt.plot(ai_range, theoretical_fp32, ':', color=color_fp32, linewidth=2, 
             alpha=0.5, label='FP32 Theoretical Roofline', zorder=2)
    plt.plot(ai_range, theoretical_fp16, ':', color=color_fp16, linewidth=2, 
             alpha=0.5, label='FP16 Theoretical Roofline', zorder=2)
    
    # Plot fitted rooflines
    plt.plot(ai_range, fitted_fp32, '-', color=color_fp32, linewidth=2.5, 
             alpha=0.9, label=f'FP32 Fitted Roofline ({method})', zorder=4)
    plt.plot(ai_range, fitted_fp16, '-', color=color_fp16, linewidth=2.5, 
             alpha=0.9, label=f'FP16 Fitted Roofline ({method})', zorder=4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
    plt.title(f'Fitted Roofline Model - T4 GPU (Log Scale, {method} Optimizer)\n' + 
              f'Fitted: BW={memory_bw_fitted*1000:.1f} GB/s, ' +
              f'FP32={peak_fp32_fitted:.2f} TFLOP/s, FP16={peak_fp16_fitted:.2f} TFLOP/s',
              fontsize=13)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    
    output_path_log = Path(output_dir) / f'fitted_roofline_{method.lower()}_log.png'
    plt.savefig(output_path_log, dpi=300, bbox_inches='tight')
    print(f"Saved log-scale plot to: {output_path_log}")
    
    plt.close('all')


def print_parameter_comparison(fitted_params_dict: Dict[str, np.ndarray]):
    """Print comparison table of theoretical vs fitted parameters."""
    print("\n" + "="*80)
    print("PARAMETER COMPARISON")
    print("="*80)
    
    memory_bw_theoretical = THEORETICAL_MEMORY_BANDWIDTH_GBS
    
    print(f"\n{'Parameter':<25s} {'Theoretical':>15s} {'Powell':>15s} {'Nelder-Mead':>15s}")
    print("-" * 80)
    
    # Memory bandwidth
    print(f"{'Memory Bandwidth (GB/s)':<25s} {memory_bw_theoretical:>15.2f} ", end="")
    for method in ['Powell', 'Nelder-Mead']:
        if method in fitted_params_dict:
            bw_fitted = fitted_params_dict[method][0] * 1000
            diff_pct = ((bw_fitted - memory_bw_theoretical) / memory_bw_theoretical) * 100
            print(f"{bw_fitted:>11.2f} ({diff_pct:+.1f}%) ", end="")
        else:
            print(f"{'N/A':>15s} ", end="")
    print()
    
    # FP32 peak
    print(f"{'FP32 Peak (TFLOP/s)':<25s} {THEORETICAL_PEAK_FP32_TFLOPS:>15.3f} ", end="")
    for method in ['Powell', 'Nelder-Mead']:
        if method in fitted_params_dict:
            peak_fitted = fitted_params_dict[method][1]
            diff_pct = ((peak_fitted - THEORETICAL_PEAK_FP32_TFLOPS) / THEORETICAL_PEAK_FP32_TFLOPS) * 100
            print(f"{peak_fitted:>11.3f} ({diff_pct:+.1f}%) ", end="")
        else:
            print(f"{'N/A':>15s} ", end="")
    print()
    
    # FP16 peak
    print(f"{'FP16 Peak (TFLOP/s)':<25s} {THEORETICAL_PEAK_FP16_TFLOPS:>15.3f} ", end="")
    for method in ['Powell', 'Nelder-Mead']:
        if method in fitted_params_dict:
            peak_fitted = fitted_params_dict[method][2]
            diff_pct = ((peak_fitted - THEORETICAL_PEAK_FP16_TFLOPS) / THEORETICAL_PEAK_FP16_TFLOPS) * 100
            print(f"{peak_fitted:>11.3f} ({diff_pct:+.1f}%) ", end="")
        else:
            print(f"{'N/A':>15s} ", end="")
    print()
    
    # Ridge points
    print(f"\n{'Ridge Point (FLOPs/Byte)':s}")
    ridge_fp32_theoretical = THEORETICAL_PEAK_FP32_TFLOPS / (memory_bw_theoretical / 1000)
    print(f"{'  FP32':<25s} {ridge_fp32_theoretical:>15.2f} ", end="")
    for method in ['Powell', 'Nelder-Mead']:
        if method in fitted_params_dict:
            ridge_fitted = fitted_params_dict[method][1] / fitted_params_dict[method][0]
            print(f"{ridge_fitted:>15.2f} ", end="")
        else:
            print(f"{'N/A':>15s} ", end="")
    print()
    
    ridge_fp16_theoretical = THEORETICAL_PEAK_FP16_TFLOPS / (memory_bw_theoretical / 1000)
    print(f"{'  FP16':<25s} {ridge_fp16_theoretical:>15.2f} ", end="")
    for method in ['Powell', 'Nelder-Mead']:
        if method in fitted_params_dict:
            ridge_fitted = fitted_params_dict[method][2] / fitted_params_dict[method][0]
            print(f"{ridge_fitted:>15.2f} ", end="")
        else:
            print(f"{'N/A':>15s} ", end="")
    print()
    
    print("="*80 + "\n")


def main():
    """Main execution function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    database_path = project_root / 'database' / 'T4_run_fixed_database.json'
    output_dir = script_dir
    
    print(f"Reading database from: {database_path}")
    data_points = parse_database(str(database_path))
    print(f"Parsed {len(data_points)} data points (excluding bf16)")
    
    fp32_count = sum(1 for dp in data_points if dp['precision'] == 'fp32')
    fp16_count = sum(1 for dp in data_points if dp['precision'] == 'fp16')
    print(f"  FP32: {fp32_count} points")
    print(f"  FP16: {fp16_count} points")
    
    # Fit with both optimizers
    fitted_params = {}
    
    for method in ['Powell', 'Nelder-Mead']:
        params, loss = fit_roofline_parameters(data_points, method=method)
        fitted_params[method] = params
        plot_fitted_roofline(data_points, params, method, str(output_dir))
    
    # Print comparison
    print_parameter_comparison(fitted_params)
    
    print("\nAll done!")


if __name__ == '__main__':
    main()

