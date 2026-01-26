#!/usr/bin/env python3
"""
Analyze execution log from point-to-point profiling runs.

Generates:
1. Duration vs Message Size (log scale) - separate lines per GPU pair
2. Duration vs Message Size (linear) with linear regression fits
3. Latency heatmap (using y-intercept as proxy)
4. Bandwidth heatmap (using slope as proxy)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import argparse


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
    result = stats.linregress(x, y)
    
    # Ensure initial guess is within bounds
    p0 = [max(1e-10, result.slope), max(0.01, result.intercept, np.min(y))]
    
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
        result = stats.linregress(x, y)
        slope, intercept = result.slope, result.intercept
        # Force intercept to be positive
        if intercept <= 0:
            intercept = 0.001
        return slope, intercept, result.rvalue ** 2


def load_execution_log(csv_path, outlier_threshold_us=None):
    """Load and prepare execution log data.
    
    Args:
        csv_path: Path to execution log CSV
        outlier_threshold_us: If specified, filter out durations above this threshold
        
    Returns:
        df: Filtered DataFrame
        removed_count: Number of outliers removed
    """
    df = pd.read_csv(csv_path)
    
    # Convert duration from nanoseconds to microseconds for better readability
    df['duration_us'] = df['duration_ns'] / 1000
    
    # Create a label for each src-dst pair
    df['pair'] = df.apply(lambda row: f"GPU {row['src']} → {row['dst']}", axis=1)
    
    # Filter outliers if threshold specified
    removed_count = 0
    if outlier_threshold_us is not None:
        original_count = len(df)
        df = df[df['duration_us'] <= outlier_threshold_us].copy()
        removed_count = original_count - len(df)
    
    return df, removed_count


def plot_duration_vs_size_log(df, output_path, outliers_removed=0):
    """Plot duration vs message size with log2 x-axis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by src-dst pair
    pairs = df.groupby(['src', 'dst'])
    
    # Use magma colormap
    colors = plt.cm.magma(np.linspace(0.1, 0.9, len(pairs)))
    
    for idx, ((src, dst), group) in enumerate(pairs):
        label = f"GPU {src} → GPU {dst}"
        
        # Plot all individual points with translucency
        ax.scatter(group['size'], group['duration_us'],
                  color=colors[idx], alpha=0.3, s=20, edgecolors='none')
        
        # Group by size and calculate mean for the line
        size_mean = group.groupby('size')['duration_us'].mean()
        
        # Plot line connecting only the averages
        ax.plot(size_mean.index, size_mean.values,
               label=label, color=colors[idx], alpha=0.9, 
               linewidth=2.5, marker='o', markersize=6)
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.set_xlabel('Message Size (bytes)', fontsize=12)
    ax.set_ylabel('Duration (μs)', fontsize=12)
    
    title = 'Point-to-Point Communication Duration vs Message Size (Log Scale)'
    if outliers_removed > 0:
        title += f'\n(Outliers removed: {outliers_removed})'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_duration_vs_size_linear_with_regression(df, output_path, outliers_removed=0):
    """Plot duration vs message size with linear regression."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Group by src-dst pair
    pairs = df.groupby(['src', 'dst'])
    
    # Store regression results
    regression_results = []
    
    # Use magma colormap
    colors = plt.cm.magma(np.linspace(0.2, 0.9, len(pairs)))
    
    for idx, ((src, dst), group) in enumerate(pairs):
        label = f"GPU {src} → GPU {dst}"
        color = colors[idx]
        
        # Plot data points
        ax.scatter(group['size'], group['duration_us'], 
                  alpha=0.4, s=30, color=color, label=f"{label} (data)")
        
        # Perform linear regression with constraint: intercept > 0
        slope, intercept, r_squared = constrained_linear_regression(
            group['size'].values, group['duration_us'].values
        )
        
        # Generate regression line
        x_line = np.array([group['size'].min(), group['size'].max()])
        y_line = slope * x_line + intercept
        
        # Plot regression line
        ax.plot(x_line, y_line, '--', color=color, linewidth=2, 
               label=f"{label} (fit: y={slope:.2e}x + {intercept:.2f}, R²={r_squared:.4f})")
        
        # Store results
        regression_results.append({
            'src': src,
            'dst': dst,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        })
    
    ax.set_xlabel('Message Size (bytes)', fontsize=12)
    ax.set_ylabel('Duration (μs)', fontsize=12)
    
    title = 'Point-to-Point Communication: Linear Regression Analysis'
    if outliers_removed > 0:
        title += f'\n(Outliers removed: {outliers_removed})'
    ax.set_title(title, fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    
    # Print regression statistics
    print("\nLinear Regression Results:")
    print("=" * 80)
    for result in regression_results:
        print(f"GPU {result['src']} → GPU {result['dst']}:")
        print(f"  Slope (m):      {result['slope']:.6e} μs/byte  (inverse bandwidth proxy)")
        print(f"  Intercept (c):  {result['intercept']:.2f} μs         (latency proxy)")
        print(f"  R²:             {result['r_squared']:.6f}")
        print(f"  p-value:        {result['p_value']:.6e}")
        print()
    
    return regression_results


def plot_latency_heatmap(regression_results, output_path, outliers_removed=0):
    """Create heatmap for latency (using y-intercept as proxy)."""
    # Get unique GPU IDs
    all_gpus = sorted(set([r['src'] for r in regression_results] + 
                          [r['dst'] for r in regression_results]))
    
    # Create matrix with NaN for diagonal (src == dst)
    latency_matrix = np.full((len(all_gpus), len(all_gpus)), np.nan)
    
    for result in regression_results:
        src_idx = all_gpus.index(result['src'])
        dst_idx = all_gpus.index(result['dst'])
        # Only fill if src != dst
        if result['src'] != result['dst']:
            latency_matrix[src_idx, dst_idx] = result['intercept']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotation array that shows empty strings for NaN, formatted to 3 decimal places
    mask = np.isnan(latency_matrix)
    annot_formatted = np.where(mask, '', 
                               [[f'{val:.3f}' if not np.isnan(val) else '' 
                                 for val in row] for row in latency_matrix])
    
    sns.heatmap(latency_matrix, 
                annot=annot_formatted,
                fmt='',
                cmap='magma',
                xticklabels=[f"GPU {i}" for i in all_gpus],
                yticklabels=[f"GPU {i}" for i in all_gpus],
                cbar_kws={'label': 'Latency (μs, y-intercept)'},
                square=True,
                ax=ax)
    
    # Add cross-hatching to diagonal cells
    for i in range(len(all_gpus)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                  fill=True, facecolor='lightgray',
                                  edgecolor='black', linewidth=1.5,
                                  hatch='///', zorder=10))
    
    ax.set_xlabel('Destination GPU', fontsize=12)
    ax.set_ylabel('Source GPU', fontsize=12)
    
    title = 'Communication Latency Heatmap\n(Y-intercept from linear regression)'
    if outliers_removed > 0:
        title += f'\n[Outliers removed: {outliers_removed}]'
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_bandwidth_heatmap(regression_results, output_path, outliers_removed=0):
    """Create heatmap for bandwidth (using slope as inverse bandwidth proxy)."""
    # Get unique GPU IDs
    all_gpus = sorted(set([r['src'] for r in regression_results] + 
                          [r['dst'] for r in regression_results]))
    
    # Create matrix with NaN for diagonal (src == dst)
    slope_matrix = np.full((len(all_gpus), len(all_gpus)), np.nan)
    
    for result in regression_results:
        src_idx = all_gpus.index(result['src'])
        dst_idx = all_gpus.index(result['dst'])
        # Only fill if src != dst
        if result['src'] != result['dst']:
            # Slope is in μs/byte
            # Convert to GB/s:
            #   1/slope = byte/μs
            #   byte/μs × 10^6 = byte/s (convert μs to s)
            #   byte/s / 10^9 = GB/s (convert byte to GB)
            #   Result: bandwidth = (1/slope) × 10^-3
            bandwidth_gbps = (1.0 / result['slope']) * 1e-3  # byte/μs to GB/s
            slope_matrix[src_idx, dst_idx] = bandwidth_gbps
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create annotation array that shows empty strings for NaN, formatted to 3 decimal places
    mask = np.isnan(slope_matrix)
    annot_formatted = np.where(mask, '', 
                               [[f'{val:.3f}' if not np.isnan(val) else '' 
                                 for val in row] for row in slope_matrix])
    
    sns.heatmap(slope_matrix, 
                annot=annot_formatted,
                fmt='',
                cmap='magma',
                xticklabels=[f"GPU {i}" for i in all_gpus],
                yticklabels=[f"GPU {i}" for i in all_gpus],
                cbar_kws={'label': 'Bandwidth (GB/s)'},
                square=True,
                ax=ax)
    
    # Add cross-hatching to diagonal cells
    for i in range(len(all_gpus)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                  fill=True, facecolor='lightgray',
                                  edgecolor='black', linewidth=1.5,
                                  hatch='///', zorder=10))
    
    ax.set_xlabel('Destination GPU', fontsize=12)
    ax.set_ylabel('Source GPU', fontsize=12)
    
    title = 'Communication Bandwidth Heatmap\n(Derived from linear regression slope)'
    if outliers_removed > 0:
        title += f'\n[Outliers removed: {outliers_removed}]'
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_individual_gpu_pairs(df, output_dir, duration_threshold_us=10000):
    """Create individual scatterplots for each GPU pair to examine data quality.
    
    Args:
        df: DataFrame with execution log data
        output_dir: Directory to save plots
        duration_threshold_us: Maximum duration to include (default: 10000 μs)
    """
    # Group by src-dst pair
    pairs = df.groupby(['src', 'dst'])
    
    # Create a subdirectory for individual plots
    individual_dir = output_dir / 'individual_pairs'
    individual_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating individual scatterplots for each GPU pair...")
    print(f"Filtering outliers > {duration_threshold_us} μs...")
    
    total_removed = 0
    
    for (src, dst), group in pairs:
        # Filter outliers
        original_count = len(group)
        group_filtered = group[group['duration_us'] <= duration_threshold_us].copy()
        removed_count = original_count - len(group_filtered)
        total_removed += removed_count
        
        # Perform linear regression first to calculate residuals (with constraint: intercept > 0)
        if len(group_filtered) > 1:
            slope, intercept, r_squared = constrained_linear_regression(
                group_filtered['size'].values, group_filtered['duration_us'].values
            )
            
            # Calculate predicted values and residuals
            group_filtered['predicted'] = slope * group_filtered['size'] + intercept
            group_filtered['residual'] = group_filtered['duration_us'] - group_filtered['predicted']
            group_filtered['abs_residual'] = np.abs(group_filtered['residual'])
            
            # Use log of residuals for coloring (emphasizes differences at small scales)
            # Add small epsilon to avoid log(0)
            epsilon = 0.01  # μs
            log_residual = np.log10(group_filtered['abs_residual'] + epsilon)
            
            # Normalize log residuals to 0-1 range for coloring
            if log_residual.max() > log_residual.min():
                residual_normalized = (log_residual - log_residual.min()) / (log_residual.max() - log_residual.min())
            else:
                residual_normalized = np.zeros(len(group_filtered))
        else:
            # Not enough points for regression
            residual_normalized = np.zeros(len(group_filtered))
            slope, intercept, r_value, p_value = 0, 0, 0, 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Log scale with residual coloring
        scatter1 = ax1.scatter(group_filtered['size'], group_filtered['duration_us'], 
                              c=residual_normalized, cmap='magma', 
                              alpha=0.7, s=30, edgecolors='black', linewidths=0.5)
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Message Size (bytes)', fontsize=11)
        ax1.set_ylabel('Duration (μs)', fontsize=11)
        ax1.set_title(f'GPU {src} → GPU {dst} (Log Scale)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar for left plot
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Log(Residual)\n(normalized)', fontsize=9)
        
        # Add statistics with filtering info
        filter_note = f'\n[Filtered: {removed_count} outliers]' if removed_count > 0 else ''
        stats_text = (f'Points: {len(group_filtered)} / {original_count}{filter_note}\n'
                     f'Mean: {group_filtered["duration_us"].mean():.2f} μs\n'
                     f'Std: {group_filtered["duration_us"].std():.2f} μs')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Right plot: Linear scale with regression and residual coloring
        scatter2 = ax2.scatter(group_filtered['size'], group_filtered['duration_us'], 
                              c=residual_normalized, cmap='magma',
                              alpha=0.7, s=30, edgecolors='black', linewidths=0.5,
                              label='Data (colored by residual)')
        
        # Add colorbar for right plot
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Log(Residual)\n(normalized)', fontsize=9)
        
        if len(group_filtered) > 1:
            # Plot regression line
            x_line = np.array([group_filtered['size'].min(), group_filtered['size'].max()])
            y_line = slope * x_line + intercept
            ax2.plot(x_line, y_line, '--', color='gray', linewidth=2.5, 
                    label=f'Linear fit', alpha=0.8)
            
            # Add regression statistics
            reg_text = (f'y = {slope:.2e}x + {intercept:.2f}\n'
                       f'R² = {r_squared:.6f}\n'
                       f'Max residual: {group_filtered["abs_residual"].max():.2f} μs')
            ax2.text(0.98, 0.02, reg_text, transform=ax2.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2.set_xlabel('Message Size (bytes)', fontsize=11)
        ax2.set_ylabel('Duration (μs)', fontsize=11)
        ax2.set_title(f'GPU {src} → GPU {dst} (Linear Scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        
        # Add filtering note to main title
        if removed_count > 0:
            fig.suptitle(f'Outliers > {duration_threshold_us} μs removed ({removed_count} points)', 
                        fontsize=10, y=0.98, color='red')
        
        plt.tight_layout()
        plot_path = individual_dir / f'gpu_{src}_to_{dst}.png'
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(pairs)} individual plots to: {individual_dir}")
    if total_removed > 0:
        print(f"  Total outliers removed: {total_removed} points (> {duration_threshold_us} μs)")


def save_regression_summary(regression_results, output_path):
    """Save regression results to CSV."""
    df = pd.DataFrame(regression_results)
    
    # Add derived metrics
    # Slope is in μs/byte, convert to GB/s: (1/slope) × 10^-3
    df['bandwidth_gbps'] = (1.0 / df['slope']) * 1e-3  # GB/s
    df['latency_us'] = df['intercept']
    
    # Reorder columns
    df = df[['src', 'dst', 'latency_us', 'bandwidth_gbps', 
             'slope', 'intercept', 'r_squared', 'p_value']]
    
    df.to_csv(output_path, index=False, float_format='%.6e')
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze point-to-point communication execution logs'
    )
    parser.add_argument('run_dir', type=str, 
                       help='Path to run directory (e.g., traces/run_20260121_164227)')
    parser.add_argument('--execution-log', type=str, default='execution_log.csv',
                       help='Name of execution log CSV file (default: execution_log.csv)')
    parser.add_argument('--outlier-threshold', type=float, default=10000.0,
                       help='Duration threshold in μs for outlier removal (default: 10000)')
    
    args = parser.parse_args()
    
    # Setup paths
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    csv_path = run_dir / args.execution_log
    if not csv_path.exists():
        print(f"Error: Execution log not found: {csv_path}")
        return 1
    
    # Create plots directory
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("POINT-TO-POINT COMMUNICATION ANALYSIS")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Execution log: {csv_path}")
    print(f"Output directory: {plots_dir}")
    print()
    
    # Load data
    print("Loading execution log...")
    df, removed_count = load_execution_log(csv_path, outlier_threshold_us=args.outlier_threshold)
    print(f"✓ Loaded {len(df)} measurements")
    if removed_count > 0:
        print(f"  Outliers removed: {removed_count} (> {args.outlier_threshold} μs)")
    print(f"  GPU pairs: {df.groupby(['src', 'dst']).ngroups}")
    print(f"  Message sizes: {df['size'].nunique()}")
    print(f"  Repetitions: {df.groupby(['src', 'dst', 'size']).size().iloc[0]}")
    print()
    
    # Generate plots
    print("Generating plots...")
    print()
    
    # Plot 1: Log scale
    plot_duration_vs_size_log(
        df, 
        plots_dir / 'duration_vs_size_log.png',
        outliers_removed=removed_count
    )
    
    # Plot 2: Linear with regression
    regression_results = plot_duration_vs_size_linear_with_regression(
        df,
        plots_dir / 'duration_vs_size_linear_regression.png',
        outliers_removed=removed_count
    )
    
    # Plot 3: Latency heatmap
    plot_latency_heatmap(
        regression_results,
        plots_dir / 'latency_heatmap.png',
        outliers_removed=removed_count
    )
    
    # Plot 4: Bandwidth heatmap
    plot_bandwidth_heatmap(
        regression_results,
        plots_dir / 'bandwidth_heatmap.png',
        outliers_removed=removed_count
    )
    
    # Plot 5: Individual GPU pair scatterplots
    plot_individual_gpu_pairs(
        df,
        plots_dir,
        duration_threshold_us=args.outlier_threshold
    )
    
    # Save regression summary
    save_regression_summary(
        regression_results,
        plots_dir / 'regression_summary.csv'
    )
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All plots saved to: {plots_dir}")
    print()
    print("Generated files:")
    print("  - duration_vs_size_log.png")
    print("  - duration_vs_size_linear_regression.png")
    print("  - latency_heatmap.png")
    print("  - bandwidth_heatmap.png")
    print("  - regression_summary.csv")
    print("  - individual_pairs/gpu_X_to_Y.png (one per GPU pair)")
    
    return 0


if __name__ == '__main__':
    exit(main())

