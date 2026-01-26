#!/usr/bin/env python3
"""
Analyze potential factors affecting bandwidth variations.

This script correlates GPU state (power, clocks, temperature) with
observed bandwidth performance to identify causes of variation.

Usage:
    python3 analyze_bandwidth_factors.py <trace_analysis.json>
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import csv


def load_trace_json(json_path):
    """Load the trace analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_temperature_csv(temp_csv_path):
    """Load temperatures.csv with all available columns."""
    df = pd.read_csv(temp_csv_path)
    print(f"\nAvailable columns in temperatures.csv: {list(df.columns)}")
    return df


def extract_config_gpu_state(trace_data, temp_df, trace_start_timestamps):
    """
    For each configuration, extract the GPU state at execution time.
    
    Returns DataFrame with columns:
        config_id, src, dst, size, duration_ms, 
        src_temp, src_power, src_clock_sm, src_clock_mem (if available),
        dst_temp, dst_power, dst_clock_sm, dst_clock_mem (if available)
    """
    records = []
    
    # Get available columns
    has_clock_graphics = 'clock_graphics' in temp_df.columns
    has_clock_sm = 'clock_sm' in temp_df.columns
    has_clock_mem = 'clock_mem' in temp_df.columns
    has_power = 'power' in temp_df.columns
    
    print(f"Temperature CSV columns available:")
    print(f"  - Power: {has_power}")
    print(f"  - Graphics Clock: {has_clock_graphics}")
    print(f"  - SM Clock: {has_clock_sm}")
    print(f"  - Memory Clock: {has_clock_mem}")
    
    configs_processed = 0
    configs_skipped_no_metadata = 0
    configs_skipped_no_timestamp = 0
    configs_skipped_no_state = 0
    
    for config_key, config_data in trace_data.items():
        # Skip metadata keys
        if config_key.startswith('_'):
            continue
        
        configs_processed += 1
            
        metadata = config_data.get('metadata', {})
        src = metadata.get('src')
        dst = metadata.get('dst')
        size = metadata.get('size')
        zeroth_duration = metadata.get('zeroth_duration')
        zeroth_timestamp = metadata.get('zeroth_timestamp')
        
        # Extract config_id from config_key (e.g., "p2p_cfg0004_s0d3_sz876523938")
        try:
            config_id = int(config_key.split('_')[1].replace('cfg', ''))
        except:
            config_id = configs_processed
        
        # Duration is in microseconds, convert to milliseconds
        duration_ms = zeroth_duration / 1000.0 if zeroth_duration is not None else None
        
        if None in [src, dst, size, duration_ms, zeroth_timestamp]:
            configs_skipped_no_timestamp += 1
            if configs_skipped_no_timestamp <= 3:  # Only print first few
                print(f"  Skipping config {config_key}: missing metadata (src={src}, dst={dst}, size={size}, dur={duration_ms}, ts={zeroth_timestamp})")
            continue
        
        # Calculate elapsed time for this config
        if src not in trace_start_timestamps:
            if configs_processed <= 3:
                print(f"  Skipping config {config_key}: src GPU {src} not in trace_start_timestamps")
            configs_skipped_no_timestamp += 1
            continue
        
        trace_start_us = trace_start_timestamps[src]
        elapsed_s = (zeroth_timestamp - trace_start_us) / 1_000_000.0
        
        # Find nearest GPU state readings for source GPU
        src_state = find_nearest_gpu_state(temp_df, src, elapsed_s, 
                                          has_power, has_clock_graphics, has_clock_sm, has_clock_mem)
        dst_state = find_nearest_gpu_state(temp_df, dst, elapsed_s,
                                          has_power, has_clock_graphics, has_clock_sm, has_clock_mem)
        
        if src_state is None or dst_state is None:
            configs_skipped_no_state += 1
            if configs_skipped_no_state <= 3:
                print(f"  Skipping config {config_key}: failed to find GPU state (src_state={src_state is not None}, dst_state={dst_state is not None})")
            continue
        
        record = {
            'config_id': config_id,
            'src': src,
            'dst': dst,
            'size': size,
            'duration_ms': duration_ms,
            'bandwidth_gbps': (size / duration_ms) * 1e-6,  # size in bytes, duration in ms
            'elapsed_s': elapsed_s,
            'src_temp': src_state['temp'],
            'dst_temp': dst_state['temp'],
        }
        
        if has_power:
            record['src_power'] = src_state['power']
            record['dst_power'] = dst_state['power']
        
        if has_clock_graphics:
            record['src_clock_graphics'] = src_state['clock_graphics']
            record['dst_clock_graphics'] = dst_state['clock_graphics']
        
        if has_clock_sm:
            record['src_clock_sm'] = src_state['clock_sm']
            record['dst_clock_sm'] = dst_state['clock_sm']
        
        if has_clock_mem:
            record['src_clock_mem'] = src_state['clock_mem']
            record['dst_clock_mem'] = dst_state['clock_mem']
        
        records.append(record)
    
    print(f"\nProcessing summary:")
    print(f"  Configs processed: {configs_processed}")
    print(f"  Configs skipped (no metadata/timestamp): {configs_skipped_no_timestamp}")
    print(f"  Configs skipped (no GPU state found): {configs_skipped_no_state}")
    print(f"  Configs successfully extracted: {len(records)}")
    
    return pd.DataFrame(records)


def find_nearest_gpu_state(temp_df, gpu_id, elapsed_s, has_power, has_clock_graphics, has_clock_sm, has_clock_mem):
    """Find the nearest GPU state reading for a given GPU and time."""
    gpu_data = temp_df[temp_df['gpu_id'] == gpu_id]
    
    if len(gpu_data) == 0:
        return None
    
    # Find nearest time point
    time_diffs = (gpu_data['elapsed_s'] - elapsed_s).abs()
    nearest_idx = time_diffs.idxmin()
    nearest_row = gpu_data.loc[nearest_idx]
    
    state = {
        'temp': nearest_row['temperature']
    }
    
    if has_power:
        state['power'] = nearest_row['power']
    if has_clock_graphics:
        state['clock_graphics'] = nearest_row['clock_graphics']
    if has_clock_sm:
        state['clock_sm'] = nearest_row['clock_sm']
    if has_clock_mem:
        state['clock_mem'] = nearest_row['clock_mem']
    
    return state


def analyze_correlations(df):
    """Calculate correlations between GPU state and bandwidth."""
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: GPU State vs Bandwidth")
    print("="*80)
    
    # Factors to correlate with bandwidth
    factors = []
    
    if 'src_temp' in df.columns:
        factors.append('src_temp')
        factors.append('dst_temp')
    
    if 'src_power' in df.columns:
        factors.append('src_power')
        factors.append('dst_power')
    
    if 'src_clock_graphics' in df.columns:
        factors.append('src_clock_graphics')
        factors.append('dst_clock_graphics')
    
    if 'src_clock_sm' in df.columns:
        factors.append('src_clock_sm')
        factors.append('dst_clock_sm')
    
    if 'src_clock_mem' in df.columns:
        factors.append('src_clock_mem')
        factors.append('dst_clock_mem')
    
    correlations = []
    
    for factor in factors:
        # Overall correlation
        pearson_r, pearson_p = pearsonr(df[factor], df['bandwidth_gbps'])
        spearman_r, spearman_p = spearmanr(df[factor], df['bandwidth_gbps'])
        
        correlations.append({
            'factor': factor,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        })
        
        print(f"\n{factor}:")
        print(f"  Pearson r:  {pearson_r:+.4f} (p={pearson_p:.4e})")
        print(f"  Spearman r: {spearman_r:+.4f} (p={spearman_p:.4e})")
    
    # Also analyze per GPU pair
    print("\n" + "="*80)
    print("PER-PAIR CORRELATIONS")
    print("="*80)
    
    for (src, dst), pair_data in df.groupby(['src', 'dst']):
        if len(pair_data) < 10:  # Skip pairs with too few samples
            continue
        
        print(f"\nGPU {src}→{dst}:")
        print(f"  Bandwidth range: {pair_data['bandwidth_gbps'].min():.2f} - {pair_data['bandwidth_gbps'].max():.2f} GB/s")
        print(f"  Bandwidth std: {pair_data['bandwidth_gbps'].std():.2f} GB/s")
        
        # Check for consistent low/high performers by message size
        size_groups = pair_data.groupby('size')['bandwidth_gbps']
        print(f"  Bandwidth variation by message size:")
        for size, bw_data in size_groups:
            if len(bw_data) >= 5:
                cv = bw_data.std() / bw_data.mean() * 100  # Coefficient of variation
                print(f"    {size:>12} bytes: {bw_data.mean():>6.2f} GB/s (CV: {cv:.1f}%)")
    
    return pd.DataFrame(correlations)


def adjust_ylim_skip_gaps(ax, y_data, min_gap_ratio=0.3):
    """
    Adjust y-axis limits to skip large empty gaps in the data.
    
    Args:
        ax: matplotlib axis
        y_data: array of y values
        min_gap_ratio: minimum gap size (as fraction of data range) to trigger adjustment
    """
    y_min, y_max = y_data.min(), y_data.max()
    y_range = y_max - y_min
    
    if y_range == 0:
        return
    
    # Sort the data to find gaps
    y_sorted = np.sort(y_data)
    
    # Find gaps between consecutive points
    gaps = np.diff(y_sorted)
    max_gap = gaps.max() if len(gaps) > 0 else 0
    
    # If there's a large gap (> min_gap_ratio of total range), adjust limits
    if max_gap > min_gap_ratio * y_range:
        # Find where the largest gap is
        gap_idx = np.argmax(gaps)
        gap_start = y_sorted[gap_idx]
        gap_end = y_sorted[gap_idx + 1]
        
        # Check if gap is in the middle of data or at edges
        lower_portion = gap_start - y_min
        upper_portion = y_max - gap_end
        
        # If the gap is near the bottom, cut the bottom
        if lower_portion < 0.1 * y_range:
            padding = 0.05 * (y_max - gap_end)
            ax.set_ylim(gap_end - padding, y_max + padding)
        # If the gap is near the top, cut the top
        elif upper_portion < 0.1 * y_range:
            padding = 0.05 * (gap_start - y_min)
            ax.set_ylim(y_min - padding, gap_start + padding)
        # Otherwise just add normal padding
        else:
            padding = 0.05 * y_range
            ax.set_ylim(y_min - padding, y_max + padding)
    else:
        # No large gaps, just add normal padding
        padding = 0.05 * y_range
        ax.set_ylim(y_min - padding, y_max + padding)


def plot_state_vs_bandwidth(df, output_dir):
    """Create scatter plots of GPU state factors vs bandwidth."""
    output_dir = Path(output_dir) / 'bandwidth_factors'
    output_dir.mkdir(exist_ok=True)
    
    factors = []
    labels = []
    
    if 'src_power' in df.columns:
        factors.extend(['src_power', 'dst_power'])
        labels.extend(['Source GPU Power (W)', 'Destination GPU Power (W)'])
    
    if 'src_clock_graphics' in df.columns:
        factors.extend(['src_clock_graphics', 'dst_clock_graphics'])
        labels.extend(['Source Graphics Clock (MHz)', 'Destination Graphics Clock (MHz)'])
    
    if 'src_clock_sm' in df.columns:
        factors.extend(['src_clock_sm', 'dst_clock_sm'])
        labels.extend(['Source SM Clock (MHz)', 'Destination SM Clock (MHz)'])
    
    if 'src_clock_mem' in df.columns:
        factors.extend(['src_clock_mem', 'dst_clock_mem'])
        labels.extend(['Source Memory Clock (MHz)', 'Destination Memory Clock (MHz)'])
    
    if 'src_temp' in df.columns:
        factors.extend(['src_temp', 'dst_temp'])
        labels.extend(['Source Temperature (°C)', 'Destination Temperature (°C)'])
    
    for factor, label in zip(factors, labels):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by GPU pair using magma colormap
        pairs = df.groupby(['src', 'dst'])
        n_pairs = len(pairs)
        colors = plt.cm.magma(np.linspace(0.2, 0.9, n_pairs))  # Avoid extremes (0.2-0.9)
        
        for idx, ((src, dst), pair_data) in enumerate(pairs):
            ax.scatter(pair_data[factor], pair_data['bandwidth_gbps'],
                      label=f'{src}→{dst}', alpha=0.6, s=20,
                      color=colors[idx])
        
        # Set log scale for x-axis only
        ax.set_xscale('log')
        
        # Adjust y-axis to skip large gaps
        adjust_ylim_skip_gaps(ax, df['bandwidth_gbps'].values)
        
        ax.set_xlabel(label, fontsize=12, fontweight='bold')
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=12, fontweight='bold')
        ax.set_title(f'Bandwidth vs {label}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        
        output_path = output_dir / f'bandwidth_vs_{factor}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")
    
    print(f"\n✓ Saved {len(factors)} factor plots to {output_dir}/")


def analyze_message_size_patterns(df, output_dir):
    """Analyze if certain message sizes consistently show different performance."""
    print("\n" + "="*80)
    print("MESSAGE SIZE PATTERN ANALYSIS")
    print("="*80)
    
    # For each GPU pair, analyze bandwidth by message size
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    # Use magma colormap for different GPU pairs
    n_pairs = len(df.groupby(['src', 'dst']))
    colors = plt.cm.magma(np.linspace(0.2, 0.9, n_pairs))
    
    for idx, ((src, dst), pair_data) in enumerate(df.groupby(['src', 'dst'])):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Get bandwidth statistics by message size
        size_stats = pair_data.groupby('size')['bandwidth_gbps'].agg(['mean', 'std', 'count'])
        
        sizes = size_stats.index.values
        means = size_stats['mean'].values
        stds = size_stats['std'].values
        counts = size_stats['count'].values
        
        # Plot with error bars using magma color for this pair
        ax.errorbar(sizes, means, yerr=stds, fmt='o-', capsize=5, markersize=6, 
                   color=colors[idx], ecolor=colors[idx], alpha=0.8)
        
        # Set log scale for x-axis only
        ax.set_xscale('log', base=2)
        
        # Adjust y-axis to skip large gaps
        adjust_ylim_skip_gaps(ax, pair_data['bandwidth_gbps'].values)
        
        ax.set_xlabel('Message Size (bytes)', fontsize=10)
        ax.set_ylabel('Bandwidth (GB/s)', fontsize=10)
        ax.set_title(f'GPU {src}→{dst}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Print analysis
        if len(size_stats) > 0:
            # Identify sizes with high variability
            high_var_sizes = size_stats[size_stats['std'] / size_stats['mean'] > 0.1]
            if len(high_var_sizes) > 0:
                print(f"\nGPU {src}→{dst} - High variability sizes (CV > 10%):")
                for size, row in high_var_sizes.iterrows():
                    cv = (row['std'] / row['mean']) * 100
                    print(f"  {size:>12} bytes: {row['mean']:.2f} ± {row['std']:.2f} GB/s (CV: {cv:.1f}%, n={int(row['count'])})")
    
    # Hide unused subplots
    for idx in range(len(df.groupby(['src', 'dst'])), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / 'bandwidth_factors' / 'bandwidth_by_message_size.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze factors affecting bandwidth variations'
    )
    parser.add_argument('json_path', type=str, help='Path to trace_analysis.json')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    temp_csv_path = json_path.parent / 'temperatures.csv'
    if not temp_csv_path.exists():
        print(f"Error: temperatures.csv not found: {temp_csv_path}")
        print("This analysis requires temperature/power/clock data.")
        return 1
    
    output_dir = json_path.parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("BANDWIDTH VARIATION FACTOR ANALYSIS")
    print("="*80)
    
    print(f"\nLoading data from: {json_path}")
    trace_data = load_trace_json(json_path)
    
    print(f"Loading temperature/power/clock data from: {temp_csv_path}")
    temp_df = load_temperature_csv(temp_csv_path)
    
    # Get trace start timestamps
    trace_start_timestamps = trace_data.get('_trace_start_timestamps', {})
    if not trace_start_timestamps:
        print("Error: No _trace_start_timestamps found in trace_analysis.json")
        print("Run extract_kernel_timestamps.py first.")
        return 1
    
    # Convert string keys to int for trace_start_timestamps
    trace_start_timestamps = {int(k): v for k, v in trace_start_timestamps.items()}
    
    print("\nExtracting GPU state at each configuration execution...")
    df = extract_config_gpu_state(trace_data, temp_df, trace_start_timestamps)
    
    if len(df) == 0:
        print("Error: No data extracted!")
        return 1
    
    print(f"✓ Extracted {len(df)} configurations with GPU state")
    
    # Analyze correlations
    corr_df = analyze_correlations(df)
    
    # Save correlation results
    corr_path = output_dir / 'bandwidth_correlation_analysis.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"\n✓ Saved correlation analysis to: {corr_path}")
    
    # Create plots
    print("\nGenerating plots...")
    plot_state_vs_bandwidth(df, output_dir)
    analyze_message_size_patterns(df, output_dir)
    
    # Save detailed data
    detail_path = output_dir / 'config_gpu_states.csv'
    df.to_csv(detail_path, index=False)
    print(f"\n✓ Saved detailed GPU state data to: {detail_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

