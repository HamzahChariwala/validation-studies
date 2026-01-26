#!/usr/bin/env python3
"""
Plot effective bandwidth over time for each GPU pair.

Shows how bandwidth varies over the duration of the benchmark,
with each point representing one configuration's aggregated performance.

Usage:
    python3 plot_bandwidth_timeline.py <trace_analysis.json>
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import HuberRegressor


def load_trace_json(json_path):
    """Load the trace analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_bandwidth_timeline(trace_data):
    """
    Extract bandwidth timeline data from trace analysis.
    
    Returns DataFrame with columns:
        config_id, src, dst, size, elapsed_s, effective_bandwidth_gbps, num_repetitions
    """
    # Get trace start timestamps
    trace_start_timestamps = trace_data.get('_trace_start_timestamps', {})
    if not trace_start_timestamps:
        raise ValueError("No _trace_start_timestamps found in trace_analysis.json")
    
    # Convert string keys to int
    trace_start_timestamps = {int(k): v for k, v in trace_start_timestamps.items()}
    
    records = []
    
    for config_key, config_data in trace_data.items():
        # Skip metadata keys
        if config_key.startswith('_'):
            continue
        
        metadata = config_data.get('metadata', {})
        src = metadata.get('src')
        dst = metadata.get('dst')
        size = metadata.get('size')
        zeroth_timestamp = metadata.get('zeroth_timestamp')
        
        # Extract config_id from config_key
        try:
            config_id = int(config_key.split('_')[1].replace('cfg', ''))
        except:
            config_id = None
        
        if None in [src, dst, size, zeroth_timestamp]:
            continue
        
        if src not in trace_start_timestamps:
            continue
        
        # Calculate elapsed time from trace start
        trace_start_us = trace_start_timestamps[src]
        elapsed_s = (zeroth_timestamp - trace_start_us) / 1_000_000.0
        
        # Get GPU duration data - specifically ncclDevKernel_SendRecv
        gpu_duration = config_data.get('gpu_duration', {})
        if not gpu_duration:
            continue
        
        # Get ncclDevKernel_SendRecv durations (the actual NCCL kernel we care about)
        nccl_durations = gpu_duration.get('ncclDevKernel_SendRecv', [])
        if not isinstance(nccl_durations, list):
            continue
        
        # Filter out None values and sum durations (in microseconds)
        valid_durations = [d for d in nccl_durations if d is not None]
        if len(valid_durations) == 0:
            continue
        
        total_duration_us = sum(valid_durations)
        num_kernels = len(valid_durations)
        
        # Calculate effective bandwidth
        # bandwidth = (size * num_repetitions) / total_time
        total_duration_s = total_duration_us / 1_000_000.0
        total_data_bytes = size * num_kernels
        bandwidth_gbps = (total_data_bytes / total_duration_s) / 1e9  # Convert to GB/s
        
        records.append({
            'config_id': config_id,
            'src': src,
            'dst': dst,
            'size': size,
            'elapsed_s': elapsed_s,
            'effective_bandwidth_gbps': bandwidth_gbps,
            'num_repetitions': num_kernels,
            'total_duration_s': total_duration_s
        })
    
    df = pd.DataFrame(records)
    
    if len(df) == 0:
        return df
    
    return df.sort_values(['src', 'dst', 'elapsed_s'])


def huber_regression_bandwidth(bandwidth_values):
    """
    Calculate Huber regression on bandwidth values to get robust mean.
    
    Returns: predicted bandwidth value (robust mean)
    """
    if len(bandwidth_values) < 2:
        return np.mean(bandwidth_values)
    
    # Fit Huber regression with just intercept (constant model)
    X = np.arange(len(bandwidth_values)).reshape(-1, 1)
    huber = HuberRegressor(epsilon=1.35, max_iter=1000)
    huber.fit(X, bandwidth_values)
    
    # Return the predicted value at the center
    center_idx = len(bandwidth_values) // 2
    predicted_bw = huber.predict([[center_idx]])[0]
    
    return predicted_bw


def plot_bandwidth_timeline(df, output_path):
    """
    Create bandwidth vs time plot with scatter points and horizontal Huber regression lines.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Get unique GPU pairs
    pairs = df.groupby(['src', 'dst'])
    n_pairs = len(pairs)
    
    # Generate colors using magma colormap
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_pairs))
    
    # Get time range for horizontal lines
    time_min = df['elapsed_s'].min()
    time_max = df['elapsed_s'].max()
    
    # Plot each GPU pair
    for idx, ((src, dst), pair_data) in enumerate(pairs):
        # Sort by elapsed time
        pair_data = pair_data.sort_values('elapsed_s')
        
        color = colors[idx]
        
        # Calculate Huber regression bandwidth
        bandwidth_values = pair_data['effective_bandwidth_gbps'].values
        huber_bw = huber_regression_bandwidth(bandwidth_values)
        
        # Plot scatter points
        ax.scatter(pair_data['elapsed_s'], pair_data['effective_bandwidth_gbps'],
                  s=30, alpha=0.6, color=color, edgecolors='none',
                  label=f'GPU {src}→{dst}')
        
        # Draw horizontal line at Huber regression bandwidth
        ax.axhline(y=huber_bw, color=color, linewidth=2.5, alpha=0.8,
                  linestyle='-', zorder=1)
    
    ax.set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Bandwidth (GB/s)', fontsize=12, fontweight='bold')
    ax.set_title('Effective Bandwidth Over Time by GPU Pair\n(Horizontal lines show Huber regression bandwidth)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Legend with multiple columns to save space
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
             framealpha=0.9, fontsize=9, ncol=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved bandwidth timeline plot to {output_path}")


def print_statistics(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BANDWIDTH TIMELINE STATISTICS")
    print("="*80)
    
    print(f"\nTotal configurations: {len(df)}")
    print(f"Time range: {df['elapsed_s'].min():.2f}s - {df['elapsed_s'].max():.2f}s")
    print(f"Total duration: {(df['elapsed_s'].max() - df['elapsed_s'].min()) / 60:.2f} minutes")
    
    print(f"\nOverall bandwidth statistics:")
    print(f"  Mean: {df['effective_bandwidth_gbps'].mean():.2f} GB/s")
    print(f"  Median: {df['effective_bandwidth_gbps'].median():.2f} GB/s")
    print(f"  Std: {df['effective_bandwidth_gbps'].std():.2f} GB/s")
    print(f"  Min: {df['effective_bandwidth_gbps'].min():.2f} GB/s")
    print(f"  Max: {df['effective_bandwidth_gbps'].max():.2f} GB/s")
    
    print(f"\nPer-pair statistics (with Huber regression bandwidth):")
    for (src, dst), pair_data in df.groupby(['src', 'dst']):
        bw_mean = pair_data['effective_bandwidth_gbps'].mean()
        bw_std = pair_data['effective_bandwidth_gbps'].std()
        bw_median = pair_data['effective_bandwidth_gbps'].median()
        bw_huber = huber_regression_bandwidth(pair_data['effective_bandwidth_gbps'].values)
        print(f"  GPU {src}→{dst}: Mean={bw_mean:.2f}, Huber={bw_huber:.2f}, ±{bw_std:.2f} GB/s (n={len(pair_data)})")


def main():
    parser = argparse.ArgumentParser(
        description='Plot effective bandwidth timeline from trace_analysis.json'
    )
    parser.add_argument('json_path', type=str, help='Path to trace_analysis.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: plots/bandwidth_timeline.png)')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = json_path.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'bandwidth_timeline.png'
    
    print("="*80)
    print("BANDWIDTH TIMELINE ANALYSIS")
    print("="*80)
    
    print(f"\nLoading data from: {json_path}")
    trace_data = load_trace_json(json_path)
    
    print("Extracting bandwidth timeline data...")
    df = extract_bandwidth_timeline(trace_data)
    
    if len(df) == 0:
        print("Error: No data extracted!")
        return 1
    
    print(f"✓ Extracted {len(df)} configurations")
    
    # Print statistics
    print_statistics(df)
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_bandwidth_timeline(df, output_path)
    
    # Save detailed data
    csv_path = output_path.parent / 'bandwidth_timeline_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved timeline data to {csv_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

