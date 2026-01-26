#!/usr/bin/env python3
"""
Plot bandwidth activity and power over time - simplified version.

Shows individual config executions as points and power overlay.

Usage:
    python3 plot_bandwidth_power_simple.py <trace_analysis.json>
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_trace_json(json_path):
    """Load the trace analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_temperature_data(temp_csv_path):
    """Load temperature and power data from CSV."""
    df = pd.read_csv(temp_csv_path)
    return df


def extract_config_points(trace_data):
    """
    Extract individual config execution points.
    
    Returns DataFrame with columns:
        config_id, src, dst, start_time_s, duration_s, bandwidth_mbps
    """
    trace_start_timestamps = trace_data.get('_trace_start_timestamps', {})
    if not trace_start_timestamps:
        raise ValueError("No _trace_start_timestamps found in trace_analysis.json")
    
    trace_start_timestamps = {int(k): v for k, v in trace_start_timestamps.items()}
    
    records = []
    
    for config_key, config_data in trace_data.items():
        if config_key.startswith('_'):
            continue
        
        metadata = config_data.get('metadata', {})
        src = metadata.get('src')
        dst = metadata.get('dst')
        size = metadata.get('size')
        zeroth_timestamp = metadata.get('zeroth_timestamp')
        
        try:
            config_id = int(config_key.split('_')[1].replace('cfg', ''))
        except:
            config_id = None
        
        if None in [src, dst, size, zeroth_timestamp]:
            continue
        
        if src not in trace_start_timestamps:
            continue
        
        # Calculate start time in seconds
        trace_start_us = trace_start_timestamps[src]
        start_time_s = (zeroth_timestamp - trace_start_us) / 1e6
        
        # Get duration and calculate bandwidth
        gpu_duration = config_data.get('gpu_duration', {})
        nccl_durations = gpu_duration.get('ncclDevKernel_SendRecv', [])
        
        if not isinstance(nccl_durations, list):
            continue
        
        valid_durations = [d for d in nccl_durations if d is not None]
        if len(valid_durations) == 0:
            continue
        
        # Durations are in milliseconds
        total_duration_s = sum(valid_durations) / 1000.0
        total_data_bytes = size * len(valid_durations)
        bandwidth_mbps = (total_data_bytes / total_duration_s) / 1e6
        
        records.append({
            'config_id': config_id,
            'src': src,
            'dst': dst,
            'start_time_s': start_time_s,
            'duration_s': total_duration_s,
            'bandwidth_mbps': bandwidth_mbps,
            'num_reps': len(valid_durations)
        })
    
    return pd.DataFrame(records).sort_values('start_time_s')


def plot_bandwidth_power(config_df, temp_df, output_path):
    """
    Create plot with config points and power overlay.
    """
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot config bandwidth points (black)
    ax1.scatter(config_df['start_time_s'], config_df['bandwidth_mbps'],
               s=15, alpha=0.5, color='black', edgecolors='none',
               label='Config Bandwidth')
    
    ax1.set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bandwidth (MB/s)', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Create second y-axis for power
    ax2 = ax1.twinx()
    color_power = 'darkgray'
    ax2.set_ylabel('Power Draw (W)', fontsize=12, fontweight='bold', color=color_power)
    
    # Plot power statistics
    power_stats = temp_df.groupby('elapsed_s')['power'].agg(['mean', 'min', 'max'])
    time_points = power_stats.index.values
    power_mean = power_stats['mean'].values
    power_min = power_stats['min'].values
    power_max = power_stats['max'].values
    
    # Plot mean power as line
    ax2.plot(time_points, power_mean, '-', 
            color=color_power, linewidth=2.5, alpha=0.8,
            label=f'Mean Power ({np.mean(power_mean):.1f} W)')
    
    # Plot fins for min/max range
    ax2.fill_between(time_points, power_min, power_max,
                     color=color_power, alpha=0.2,
                     label='Power Range')
    
    ax2.tick_params(axis='y', labelcolor=color_power)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    plt.title('Config Bandwidth and GPU Power Over Time', 
             fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved bandwidth-power plot to {output_path}")


def print_statistics(config_df, temp_df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BANDWIDTH AND POWER STATISTICS")
    print("="*80)
    
    print(f"\nConfigurations:")
    print(f"  Total: {len(config_df)}")
    print(f"  Time range: {config_df['start_time_s'].min():.2f}s - {config_df['start_time_s'].max():.2f}s")
    print(f"  Total benchmark duration: {temp_df['elapsed_s'].max():.2f}s ({temp_df['elapsed_s'].max()/60:.2f} minutes)")
    
    print(f"\nBandwidth per config:")
    print(f"  Mean: {config_df['bandwidth_mbps'].mean():.2f} MB/s")
    print(f"  Median: {config_df['bandwidth_mbps'].median():.2f} MB/s")
    print(f"  Min: {config_df['bandwidth_mbps'].min():.2f} MB/s")
    print(f"  Max: {config_df['bandwidth_mbps'].max():.2f} MB/s")
    
    print(f"\nPower:")
    print(f"  Mean: {temp_df['power'].mean():.2f} W")
    print(f"  Range: {temp_df['power'].min():.2f} - {temp_df['power'].max():.2f} W")


def main():
    parser = argparse.ArgumentParser(
        description='Plot bandwidth and power timeline (simplified)'
    )
    parser.add_argument('json_path', type=str, help='Path to trace_analysis.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    temp_csv_path = json_path.parent / 'temperatures.csv'
    if not temp_csv_path.exists():
        print(f"Error: temperatures.csv not found: {temp_csv_path}")
        return 1
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = json_path.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'bandwidth_power_timeline.png'
    
    print("="*80)
    print("BANDWIDTH AND POWER TIMELINE ANALYSIS")
    print("="*80)
    
    print(f"\nLoading data...")
    trace_data = load_trace_json(json_path)
    temp_df = load_temperature_data(temp_csv_path)
    
    print("Extracting config data...")
    config_df = extract_config_points(trace_data)
    
    if len(config_df) == 0:
        print("Error: No config data extracted!")
        return 1
    
    print(f"✓ Extracted {len(config_df)} configurations")
    
    # Print statistics
    print_statistics(config_df, temp_df)
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_bandwidth_power(config_df, temp_df, output_path)
    
    # Save data
    csv_path = output_path.parent / 'bandwidth_power_configs.csv'
    config_df.to_csv(csv_path, index=False)
    print(f"✓ Saved config data to {csv_path}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

