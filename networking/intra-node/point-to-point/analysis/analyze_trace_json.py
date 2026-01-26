#!/usr/bin/env python3
"""
Analyze trace_analysis.json and generate plots using NCCL kernel timings.

This replaces analyze_execution_log.py and uses the actual GPU kernel durations
instead of CPU-measured times.

Usage:
    # Basic (no outlier filtering)
    python3 analyze_trace_json.py path/to/trace_analysis.json
    
    # With outlier filtering
    python3 analyze_trace_json.py path/to/trace_analysis.json --outlier-threshold 1000

Generates:
    - duration_vs_size_log.png - Duration vs size on log scale
    - duration_vs_size_linear_regression.png - Linear regression with R²
    - latency_heatmap.png - Latency (y-intercept) heatmap
    - bandwidth_heatmap.png - Bandwidth heatmap
    - individual_pairs/ - Scatter plots for each GPU pair with residual coloring
    - regression_summary.csv - Numerical results
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import linregress
from scipy.optimize import curve_fit
import csv
from sklearn.linear_model import HuberRegressor


def load_trace_json(json_path):
    """Load the trace analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_temperature_data(temp_csv_path):
    """
    Load temperature data from CSV.
    
    Returns:
        dict: gpu_id -> list of (elapsed_s, temperature) tuples
        tuple: (global_min_temp, global_max_temp)
    """
    temp_data = {}
    
    with open(temp_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gpu_id = int(row['gpu_id'])
            elapsed_s = float(row['elapsed_s'])
            temperature = float(row['temperature'])
            
            if gpu_id not in temp_data:
                temp_data[gpu_id] = []
            
            temp_data[gpu_id].append((elapsed_s, temperature))
    
    # Sort by elapsed time for each GPU
    for gpu_id in temp_data:
        temp_data[gpu_id].sort(key=lambda x: x[0])
    
    # Find global min/max temperatures
    all_temps = []
    for gpu_temps in temp_data.values():
        all_temps.extend([t[1] for t in gpu_temps])
    
    global_min_temp = min(all_temps) if all_temps else 0
    global_max_temp = max(all_temps) if all_temps else 100
    
    return temp_data, (global_min_temp, global_max_temp)


def match_kernel_to_temperature(kernel_timestamp, gpu_id, temp_data, trace_start_timestamps):
    """
    Match a kernel execution to the nearest temperature reading.
    
    Args:
        kernel_timestamp: Kernel timestamp in microseconds (from GPU trace)
        gpu_id: GPU ID
        temp_data: Temperature data dict from load_temperature_data() (elapsed_s, temp)
        trace_start_timestamps: Dict of gpu_id -> first trace timestamp in microseconds
    
    Returns:
        float: Temperature in Celsius, or None if not found
    """
    if gpu_id not in temp_data or gpu_id not in trace_start_timestamps:
        return None
    
    # Calculate elapsed time from trace start
    # Both kernel_timestamp and trace_start are in microseconds on CLOCK_MONOTONIC
    trace_start_us = trace_start_timestamps[gpu_id]
    elapsed_s = (kernel_timestamp - trace_start_us) / 1_000_000.0
    
    gpu_temps = temp_data[gpu_id]
    
    # Find nearest temperature reading by elapsed time
    min_diff = float('inf')
    nearest_temp = None
    
    for temp_elapsed_s, temperature in gpu_temps:
        diff = abs(temp_elapsed_s - elapsed_s)
        if diff < min_diff:
            min_diff = diff
            nearest_temp = temperature
    
    return nearest_temp


def linear_model(x, slope, intercept):
    """Linear model for curve fitting."""
    return slope * x + intercept


def constrained_linear_regression(x, y):
    """
    Perform linear regression with constraint that intercept > 0.
    
    Returns: slope, intercept, r_squared
    """
    # Use curve_fit with bounds to enforce intercept > 0
    # Initial guess using unconstrained regression
    init_slope, init_intercept, _, _, _ = linregress(x, y)
    
    # Ensure initial guess is within bounds
    p0 = [max(1e-10, init_slope), max(0.01, init_intercept, np.min(y))]
    
    # Bounds: slope must be positive, intercept must be > 0.001
    bounds = ([1e-10, 0.001], [np.inf, np.inf])
    
    try:
        popt, _ = curve_fit(linear_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        slope, intercept = popt
        
        # Calculate R²
        y_pred = linear_model(x, slope, intercept)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared
    except Exception as e:
        # Fallback to unconstrained if fitting fails
        print(f"Warning: Constrained fit failed, using unconstrained: {e}")
        slope, intercept, r_value, _, _ = linregress(x, y)
        # Force intercept to be positive
        if intercept <= 0:
            intercept = 0.001
        return slope, intercept, r_value ** 2


def huber_regression(x, y):
    """
    Perform Huber regression (robust to outliers).
    
    Returns: slope, intercept, r_squared
    """
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    X = x.reshape(-1, 1)
    huber.fit(X, y)
    
    slope = huber.coef_[0]
    intercept = huber.intercept_
    
    # Ensure intercept is positive
    if intercept <= 0:
        intercept = 0.001
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared


def irls_regression(x, y, max_iter=50, tol=1e-6):
    """
    Perform Iteratively Reweighted Least Squares regression.
    
    Returns: slope, intercept, r_squared
    """
    # Initialize with OLS
    X_design = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(X_design, y, rcond=None)[0]
    
    for iteration in range(max_iter):
        # Calculate residuals
        y_pred = X_design @ beta
        residuals = y - y_pred
        
        # Calculate weights using Huber's function
        # w_i = 1 if |r_i| <= k, else k/|r_i|
        k = 1.345 * np.median(np.abs(residuals))  # Robust scale estimate
        weights = np.ones_like(residuals)
        outlier_mask = np.abs(residuals) > k
        weights[outlier_mask] = k / np.abs(residuals[outlier_mask])
        
        # Weighted least squares
        W = np.diag(weights)
        beta_new = np.linalg.lstsq(X_design.T @ W @ X_design, 
                                    X_design.T @ W @ y, rcond=None)[0]
        
        # Check convergence
        if np.allclose(beta, beta_new, atol=tol):
            break
        
        beta = beta_new
    
    slope, intercept = beta
    
    # Ensure intercept is positive
    if intercept <= 0:
        intercept = 0.001
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, intercept, r_squared


def extract_kernel_data(data):
    """
    Extract NCCL kernel timing data from JSON.
    
    Returns: DataFrame with columns [src, dst, size, repetition, duration_ms]
    """
    rows = []
    
    for config_key, config_data in data.items():
        # Skip metadata keys
        if config_key.startswith('_'):
            continue
        
        meta = config_data['metadata']
        src = meta['src']
        dst = meta['dst']
        size = meta['size']
        
        # Extract config_id from config_key (e.g., p2p_cfg0003_s0d3_sz2 -> 3)
        import re
        match = re.search(r'p2p_cfg(\d+)_', config_key)
        config_id = int(match.group(1)) if match else 0
        
        # Find the NCCL kernel operation
        kernel_name = None
        for op_name in config_data['gpu_duration'].keys():
            if 'ncclDevKernel_SendRecv' in op_name:
                kernel_name = op_name
                break
        
        if not kernel_name:
            continue
        
        kernel_durations = config_data['gpu_duration'][kernel_name]
        
        # Extract each repetition
        for rep_idx, duration_ms in enumerate(kernel_durations):
            if duration_ms is not None:
                rows.append({
                    'config_id': config_id,
                    'src': src,
                    'dst': dst,
                    'size': size,
                    'repetition': rep_idx,
                    'duration_ms': duration_ms
                })
    
    return pd.DataFrame(rows)


def filter_outliers(df, threshold_ms=None):
    """
    Filter out outliers based on duration threshold.
    
    Args:
        df: DataFrame to filter
        threshold_ms: Maximum duration to keep. If None, no filtering is applied.
    
    Returns:
        Filtered DataFrame, number of outliers removed
    """
    if threshold_ms is None:
        # No filtering
        return df.copy(), 0
    
    initial_count = len(df)
    df_filtered = df[df['duration_ms'] <= threshold_ms].copy()
    removed_count = initial_count - len(df_filtered)
    
    return df_filtered, removed_count


def plot_duration_vs_size_log(df, output_path, outliers_removed=0):
    """Plot duration vs size on log2 scale with lines for each src-dst pair."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique src-dst pairs
    pairs = df.groupby(['src', 'dst'])
    n_pairs = len(pairs)
    
    # Get colors from magma
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_pairs))
    
    for (src, dst), color in zip(pairs.groups.keys(), colors):
        pair_data = pairs.get_group((src, dst))
        
        # Plot all individual points with translucency
        ax.scatter(pair_data['size'], pair_data['duration_ms'],
                  color=color, alpha=0.3, s=20, edgecolors='none')
        
        # Group by size and calculate mean for the line
        size_mean = pair_data.groupby('size')['duration_ms'].mean()
        
        # Plot line connecting only the averages
        ax.plot(size_mean.index, size_mean.values, 
               label=f'{src}→{dst}',
               color=color, alpha=0.9, linewidth=2.5, marker='o', markersize=6)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('Message Size (bytes)', fontsize=12)
    ax.set_ylabel('NCCL Kernel Duration (ms)', fontsize=12)
    
    title = 'NCCL Kernel Duration vs Message Size (Log Scale)'
    if outliers_removed > 0:
        title += f'\n(Filtered {outliers_removed} outliers)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")


def plot_duration_vs_size_linear_regression(df, output_path, outliers_removed=0):
    """Plot duration vs size with linear regression for each pair."""
    pairs = df.groupby(['src', 'dst'])
    n_pairs = len(pairs)
    
    # Calculate grid size
    n_cols = 3
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]
    
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_pairs))
    
    regression_results = []
    
    for idx, ((src, dst), color) in enumerate(zip(pairs.groups.keys(), colors)):
        ax = axes[idx]
        pair_data = pairs.get_group((src, dst))
        
        sizes = pair_data['size'].values
        durations = pair_data['duration_ms'].values
        
        # Linear regression with constraint: intercept > 0
        slope, intercept, r_squared = constrained_linear_regression(sizes, durations)
        
        regression_results.append({
            'src': src,
            'dst': dst,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        })
        
        # Scatter plot
        ax.scatter(sizes, durations, c=[color], alpha=0.6, s=30, 
                  edgecolors='black', linewidths=0.5)
        
        # Regression line
        x_line = np.array([sizes.min(), sizes.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='gray', linewidth=2.5, alpha=0.8)
        
        # Add regression info
        ax.text(0.05, 0.95, 
               f'R²={r_squared:.4f}\nm={slope:.2e}\nc={intercept:.4f}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Message Size (bytes)', fontsize=10)
        ax.set_ylabel('Duration (ms)', fontsize=10)
        ax.set_title(f'{src}→{dst}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].axis('off')
    
    title = 'Linear Regression: NCCL Kernel Duration vs Message Size'
    if outliers_removed > 0:
        title += f' (Filtered {outliers_removed} outliers)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    return pd.DataFrame(regression_results)


def plot_latency_heatmap(regression_df, output_path):
    """Create heatmap of latency (y-intercept) for each src-dst pair."""
    # Get unique sources and destinations
    sources = sorted(regression_df['src'].unique())
    destinations = sorted(regression_df['dst'].unique())
    
    # Create matrix with NaN for diagonal (src == dst)
    matrix = np.full((len(sources), len(destinations)), np.nan)
    
    for _, row in regression_df.iterrows():
        src_idx = sources.index(row['src'])
        dst_idx = destinations.index(row['dst'])
        # Only fill if src != dst
        if row['src'] != row['dst']:
            matrix[src_idx, dst_idx] = row['intercept']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use mask for annotations to avoid showing "nan"
    mask = np.isnan(matrix)
    annot_matrix = np.where(mask, '', matrix)
    
    # Format annotations to 3 decimal places
    annot_formatted = np.where(mask, '', 
                               [[f'{val:.3f}' if not np.isnan(val) else '' 
                                 for val in row] for row in matrix])
    
    sns.heatmap(matrix, annot=annot_formatted, fmt='', cmap='magma',
               xticklabels=[f'GPU {d}' for d in destinations],
               yticklabels=[f'GPU {s}' for s in sources],
               cbar_kws={'label': 'Latency (ms, y-intercept)'},
               ax=ax, mask=False)  # Don't mask in heatmap, but use custom coloring
    
    # Add cross-hatching to diagonal cells
    for i in range(min(len(sources), len(destinations))):
        if sources[i] in destinations:
            j = destinations.index(sources[i])
            ax.add_patch(plt.Rectangle((j, i), 1, 1, 
                                      fill=True, facecolor='lightgray',
                                      edgecolor='black', linewidth=1.5,
                                      hatch='///', zorder=10))
    
    ax.set_xlabel('Destination', fontsize=12)
    ax.set_ylabel('Source', fontsize=12)
    ax.set_title('Latency Heatmap (NCCL Kernel)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")


def plot_bandwidth_heatmap(regression_df, output_path):
    """Create heatmap of bandwidth (derived from slope) for each src-dst pair."""
    # Get unique sources and destinations
    sources = sorted(regression_df['src'].unique())
    destinations = sorted(regression_df['dst'].unique())
    
    # Create matrix with NaN for diagonal (src == dst)
    # slope is ms/byte
    # 1/slope = byte/ms = 10^3 byte/s = 10^-6 GB/s
    matrix = np.full((len(sources), len(destinations)), np.nan)
    
    for _, row in regression_df.iterrows():
        src_idx = sources.index(row['src'])
        dst_idx = destinations.index(row['dst'])
        # Only fill if src != dst and slope is valid
        if row['src'] != row['dst'] and row['slope'] > 0:
            bandwidth_gbps = (1.0 / row['slope']) * 1e-6  # byte/ms to GB/s
            matrix[src_idx, dst_idx] = bandwidth_gbps
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use mask for annotations to avoid showing "nan"
    mask = np.isnan(matrix)
    
    # Format annotations to 3 decimal places
    annot_formatted = np.where(mask, '', 
                               [[f'{val:.3f}' if not np.isnan(val) else '' 
                                 for val in row] for row in matrix])
    
    sns.heatmap(matrix, annot=annot_formatted, fmt='', cmap='magma',
               xticklabels=[f'GPU {d}' for d in destinations],
               yticklabels=[f'GPU {s}' for s in sources],
               cbar_kws={'label': 'Bandwidth (GB/s)'},
               ax=ax, mask=False)
    
    # Add cross-hatching to diagonal cells
    for i in range(min(len(sources), len(destinations))):
        if sources[i] in destinations:
            j = destinations.index(sources[i])
            ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                      fill=True, facecolor='lightgray',
                                      edgecolor='black', linewidth=1.5,
                                      hatch='///', zorder=10))
    
    ax.set_xlabel('Destination', fontsize=12)
    ax.set_ylabel('Source', fontsize=12)
    ax.set_title('Bandwidth Heatmap (NCCL Kernel)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")


def plot_individual_gpu_pairs(df, output_dir, outliers_removed=0):
    """Create individual scatter plots for each GPU pair with residual coloring."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pairs = df.groupby(['src', 'dst'])
    
    for (src, dst), pair_data in pairs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sizes = pair_data['size'].values
        durations = pair_data['duration_ms'].values
        
        # Linear regression with constraint: intercept > 0
        slope, intercept, r_squared = constrained_linear_regression(sizes, durations)
        
        # Calculate residuals
        predicted = slope * sizes + intercept
        residuals = durations - predicted
        
        # Log of absolute residuals for coloring
        log_abs_residual = np.log10(np.abs(residuals) + 0.01)
        
        # Normalize for coloring
        if log_abs_residual.max() > log_abs_residual.min():
            residual_normalized = (log_abs_residual - log_abs_residual.min()) / \
                                 (log_abs_residual.max() - log_abs_residual.min())
        else:
            residual_normalized = np.zeros_like(log_abs_residual)
        
        # Left plot: Log scale
        scatter1 = ax1.scatter(sizes, durations, 
                              c=residual_normalized, cmap='magma',
                              alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Message Size (bytes)', fontsize=11)
        ax1.set_ylabel('Duration (ms)', fontsize=11)
        ax1.set_title(f'GPU {src}→{dst} (Log Scale)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Normalized Log(Residual)', fontsize=9)
        
        # Right plot: Linear scale with regression
        scatter2 = ax2.scatter(sizes, durations,
                              c=residual_normalized, cmap='magma',
                              alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
        
        # Regression line
        x_line = np.array([sizes.min(), sizes.max()])
        y_line = slope * x_line + intercept
        ax2.plot(x_line, y_line, '--', color='gray', linewidth=2.5, alpha=0.8,
                label='Linear fit')
        
        ax2.set_xlabel('Message Size (bytes)', fontsize=11)
        ax2.set_ylabel('Duration (ms)', fontsize=11)
        
        title = f'GPU {src}→{dst} (Linear Scale)\nR²={r_squared:.4f}, m={slope:.2e}, c={intercept:.4f}'
        if outliers_removed > 0:
            title += f'\n({outliers_removed} outliers removed)'
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Normalized Log(Residual)', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f'gpu_{src}_to_{dst}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(pairs)} individual pair plots to {output_dir}/")


def plot_individual_gpu_pairs_huber(df, output_dir, outliers_removed=0):
    """Create individual scatter plots for each GPU pair using Huber regression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pairs = df.groupby(['src', 'dst'])
    
    for (src, dst), pair_data in pairs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sizes = pair_data['size'].values
        durations = pair_data['duration_ms'].values
        
        # Huber regression
        slope, intercept, r_squared = huber_regression(sizes, durations)
        
        # Calculate residuals
        predicted = slope * sizes + intercept
        residuals = durations - predicted
        
        # Log of absolute residuals for coloring
        log_abs_residual = np.log10(np.abs(residuals) + 0.01)
        
        # Normalize for coloring
        if log_abs_residual.max() > log_abs_residual.min():
            residual_normalized = (log_abs_residual - log_abs_residual.min()) / \
                                 (log_abs_residual.max() - log_abs_residual.min())
        else:
            residual_normalized = np.zeros_like(log_abs_residual)
        
        # Scatter plot with residual coloring
        scatter = ax.scatter(sizes, durations,
                            c=residual_normalized, cmap='magma',
                            alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
        
        # Regression line
        x_line = np.array([sizes.min(), sizes.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='gray', linewidth=2.5, alpha=0.8,
               label='Huber fit')
        
        ax.set_xlabel('Message Size (bytes)', fontsize=11)
        ax.set_ylabel('Duration (ms)', fontsize=11)
        
        title = f'GPU {src}→{dst} (Huber Regression)\nR²={r_squared:.4f}, m={slope:.2e}, c={intercept:.4f}'
        if outliers_removed > 0:
            title += f'\n({outliers_removed} outliers removed)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Log(Residual)', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f'gpu_{src}_to_{dst}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(pairs)} Huber regression plots to {output_dir}/")


def plot_individual_gpu_pairs_irls(df, output_dir, outliers_removed=0):
    """Create individual scatter plots for each GPU pair using IRLS regression."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pairs = df.groupby(['src', 'dst'])
    
    for (src, dst), pair_data in pairs:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        sizes = pair_data['size'].values
        durations = pair_data['duration_ms'].values
        
        # IRLS regression
        slope, intercept, r_squared = irls_regression(sizes, durations)
        
        # Calculate residuals
        predicted = slope * sizes + intercept
        residuals = durations - predicted
        
        # Log of absolute residuals for coloring
        log_abs_residual = np.log10(np.abs(residuals) + 0.01)
        
        # Normalize for coloring
        if log_abs_residual.max() > log_abs_residual.min():
            residual_normalized = (log_abs_residual - log_abs_residual.min()) / \
                                 (log_abs_residual.max() - log_abs_residual.min())
        else:
            residual_normalized = np.zeros_like(log_abs_residual)
        
        # Scatter plot with residual coloring
        scatter = ax.scatter(sizes, durations,
                            c=residual_normalized, cmap='magma',
                            alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
        
        # Regression line
        x_line = np.array([sizes.min(), sizes.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color='gray', linewidth=2.5, alpha=0.8,
               label='IRLS fit')
        
        ax.set_xlabel('Message Size (bytes)', fontsize=11)
        ax.set_ylabel('Duration (ms)', fontsize=11)
        
        title = f'GPU {src}→{dst} (IRLS Regression)\nR²={r_squared:.4f}, m={slope:.2e}, c={intercept:.4f}'
        if outliers_removed > 0:
            title += f'\n({outliers_removed} outliers removed)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Log(Residual)', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f'gpu_{src}_to_{dst}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(pairs)} IRLS regression plots to {output_dir}/")


def plot_individual_gpu_pairs_vs_temperature(df, output_dir, temp_csv_path, trace_data, outliers_removed=0):
    """
    Create scatter plots for each GPU pair colored by temperature.
    
    Similar to the right plot in plot_individual_gpu_pairs, but colored by GPU temperature
    instead of residuals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load temperature data
    print(f"  Loading temperature data from {temp_csv_path}...")
    temp_data, (global_min_temp, global_max_temp) = load_temperature_data(temp_csv_path)
    print(f"  Global temperature range: {global_min_temp:.1f}°C to {global_max_temp:.1f}°C")
    
    # Get trace start timestamps for elapsed time calculation
    trace_start_timestamps = trace_data.get('_trace_start_timestamps', {})
    if not trace_start_timestamps:
        print("  Warning: No _trace_start_timestamps found in trace_analysis.json")
        print("  Run extract_kernel_timestamps.py first to add timestamp information")
        return
    
    # Convert string keys to integers
    trace_start_timestamps = {int(k): v for k, v in trace_start_timestamps.items()}
    
    pairs = df.groupby(['src', 'dst'])
    
    for (src, dst), pair_data in pairs:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        sizes = pair_data['size'].values
        durations = pair_data['duration_ms'].values
        
        # Match each data point to temperature
        temperatures = []
        for idx in pair_data.index:
            row = pair_data.loc[idx]
            config_name = f"p2p_cfg{int(row['config_id']):04d}_s{int(row['src'])}d{int(row['dst'])}_sz{int(row['size'])}"
            
            # Get kernel timestamp from trace_data
            if config_name in trace_data:
                metadata = trace_data[config_name].get('metadata', {})
                kernel_timestamp = metadata.get('zeroth_timestamp')
                
                if kernel_timestamp is not None:
                    temp = match_kernel_to_temperature(
                        kernel_timestamp, 
                        int(src), 
                        temp_data, 
                        trace_start_timestamps
                    )
                    temperatures.append(temp if temp is not None else global_min_temp)
                else:
                    temperatures.append(global_min_temp)
            else:
                temperatures.append(global_min_temp)
        
        temperatures = np.array(temperatures)
        
        # Normalize temperatures to [0, 1] for opacity and colormap
        if global_max_temp > global_min_temp:
            temp_normalized = (temperatures - global_min_temp) / (global_max_temp - global_min_temp)
        else:
            temp_normalized = np.zeros_like(temperatures)
        
        # Map temperature to opacity: cooler = more transparent (0.3), hotter = more opaque (1.0)
        alpha_values = 0.3 + 0.7 * temp_normalized
        
        # Scatter plot colored by temperature with variable opacity
        scatter = ax.scatter(sizes, durations,
                           c=temperatures, cmap='coolwarm',
                           vmin=global_min_temp, vmax=global_max_temp,
                           alpha=alpha_values, s=40, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('Message Size (bytes)', fontsize=11)
        ax.set_ylabel('Duration (ms)', fontsize=11)
        
        title = f'GPU {src}→{dst} - Temperature Colored'
        if outliers_removed > 0:
            title += f'\n({outliers_removed} outliers removed)'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('GPU Temperature (°C)', fontsize=9)
        
        plt.tight_layout()
        
        output_path = output_dir / f'gpu_{src}_to_{dst}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(pairs)} temperature-colored plots to {output_dir}/")


def plot_temperature_over_time(temp_csv_path, output_dir):
    """
    Create temperature plot over time showing each GPU.
    
    Args:
        temp_csv_path: Path to temperatures.csv
        output_dir: Directory to save the plot
    """
    print(f"  Loading temperature data from {temp_csv_path}...")
    
    # Load data
    df = pd.read_csv(temp_csv_path)
    
    # Get time range
    time_min = df['elapsed_s'].min()
    time_max = df['elapsed_s'].max()
    print(f"  Time range: {time_min:.2f}s to {time_max:.2f}s ({(time_max-time_min)/60:.1f} minutes)")
    
    # Get unique GPU IDs
    gpu_ids = sorted(df['gpu_id'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use magma colormap for GPU colors
    n_gpus = len(gpu_ids)
    colors = plt.cm.magma(np.linspace(0.2, 0.9, n_gpus))
    
    # Plot temperature line for each GPU
    for idx, gpu_id in enumerate(gpu_ids):
        gpu_df = df[df['gpu_id'] == gpu_id].sort_values('elapsed_s')
        ax.plot(gpu_df['elapsed_s'], gpu_df['temperature'],
               label=f'GPU {gpu_id}',
               color=colors[idx],
               linewidth=2,
               alpha=0.8)
    
    ax.set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9)
    
    # Set title
    plt.title('GPU Temperature Over Time', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the plot
    output_path = output_dir / 'temperature_over_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved temperature plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze trace_analysis.json and generate plots'
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to trace_analysis.json'
    )
    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=None,
        help='Duration threshold in ms for outlier filtering (default: disabled)'
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Create output directory
    output_dir = json_path.parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {json_path}")
    data = load_trace_json(json_path)
    print(f"Loaded {len(data)} configurations")
    
    # Extract kernel data
    print("\nExtracting NCCL kernel timing data...")
    df = extract_kernel_data(data)
    print(f"Extracted {len(df)} kernel measurements")
    
    if len(df) == 0:
        print("Error: No kernel data found!")
        return 1
    
    # Filter outliers
    if args.outlier_threshold is not None:
        print(f"\nFiltering outliers (threshold: {args.outlier_threshold} ms)...")
        df_filtered, outliers_removed = filter_outliers(df, args.outlier_threshold)
        print(f"Removed {outliers_removed} outliers ({100*outliers_removed/len(df):.1f}%)")
        print(f"Remaining data points: {len(df_filtered)}")
    else:
        print("\nOutlier filtering disabled")
        df_filtered, outliers_removed = filter_outliers(df, None)
        print(f"Using all {len(df_filtered)} data points")
    
    # Generate plots
    print(f"\nGenerating plots in: {output_dir}/")
    
    print("\n1. Duration vs Size (Log Scale)...")
    plot_duration_vs_size_log(df_filtered, 
                             output_dir / 'duration_vs_size_log.png',
                             outliers_removed)
    
    print("\n2. Linear Regression Analysis...")
    regression_df = plot_duration_vs_size_linear_regression(
        df_filtered,
        output_dir / 'duration_vs_size_linear_regression.png',
        outliers_removed
    )
    
    # Save regression summary
    summary_path = output_dir / 'regression_summary.csv'
    # Add bandwidth column
    # slope is ms/byte, so 1/slope = byte/ms = 10^3 byte/s = 10^-6 GB/s
    regression_df['bandwidth_gbps'] = (1.0 / regression_df['slope']) * 1e-6
    regression_df.to_csv(summary_path, index=False)
    print(f"✓ Saved regression summary: {summary_path}")
    
    print("\n3. Latency Heatmap...")
    plot_latency_heatmap(regression_df, output_dir / 'latency_heatmap.png')
    
    print("\n4. Bandwidth Heatmap...")
    plot_bandwidth_heatmap(regression_df, output_dir / 'bandwidth_heatmap.png')
    
    print("\n5. Individual GPU Pair Plots...")
    individual_dir = output_dir / 'individual_pairs'
    plot_individual_gpu_pairs(df_filtered, individual_dir, outliers_removed)
    
    print("\n5a. Individual GPU Pair Plots (Huber Regression)...")
    huber_dir = output_dir / 'individual_huber'
    plot_individual_gpu_pairs_huber(df_filtered, huber_dir, outliers_removed)
    
    print("\n5b. Individual GPU Pair Plots (IRLS Regression)...")
    irls_dir = output_dir / 'individual_irls'
    plot_individual_gpu_pairs_irls(df_filtered, irls_dir, outliers_removed)
    
    # 6. Temperature-colored plots (if temperature data available)
    print("\n6. Individual GPU Pair Plots (Temperature-colored)...")
    temp_csv_path = json_path.parent / 'temperatures.csv'
    if temp_csv_path.exists():
        temp_dir = output_dir / 'individual_vs_temp'
        plot_individual_gpu_pairs_vs_temperature(
            df_filtered, temp_dir, temp_csv_path, data, outliers_removed
        )
        
        # 7. Temperature over time plot
        print("\n7. Temperature Over Time...")
        plot_temperature_over_time(temp_csv_path, output_dir)
    else:
        print(f"  Skipping: temperatures.csv not found at {temp_csv_path}")
        print(f"  (Temperature-colored plots require temperature data)")
    
    print("\n" + "="*80)
    print("✓ All plots generated successfully!")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

