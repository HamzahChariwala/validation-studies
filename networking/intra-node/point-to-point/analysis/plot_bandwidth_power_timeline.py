#!/usr/bin/env python3
"""
Plot aggregate bandwidth and power over time.

Shows time-averaged bandwidth across all concurrent operations
and overlays power draw statistics.

Usage:
    python3 plot_bandwidth_power_timeline.py <trace_analysis.json>
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


def load_temperature_data(temp_csv_path):
    """Load temperature and power data from CSV."""
    df = pd.read_csv(temp_csv_path)
    return df


def extract_execution_timeline(trace_data):
    """
    Extract when each config was executing with its bandwidth.
    
    Returns DataFrame with columns:
        config_id, src, dst, start_time_s, end_time_s, bandwidth_mbps
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
        zeroth_duration = metadata.get('zeroth_duration')
        
        # Extract config_id
        try:
            config_id = int(config_key.split('_')[1].replace('cfg', ''))
        except:
            config_id = None
        
        if None in [src, dst, size, zeroth_timestamp, zeroth_duration]:
            continue
        
        if src not in trace_start_timestamps:
            continue
        
        # Calculate elapsed time from trace start (in seconds)
        trace_start_us = trace_start_timestamps[src]
        start_time_s = (zeroth_timestamp - trace_start_us) / 1_000_000.0
        
        # Get GPU duration data - specifically ncclDevKernel_SendRecv
        gpu_duration = config_data.get('gpu_duration', {})
        if not gpu_duration:
            continue
        
        # Get ncclDevKernel_SendRecv durations
        nccl_durations = gpu_duration.get('ncclDevKernel_SendRecv', [])
        if not isinstance(nccl_durations, list):
            continue
        
        # Filter out None values and sum durations (in microseconds)
        valid_durations = [d for d in nccl_durations if d is not None]
        if len(valid_durations) == 0:
            continue
        
        # Total execution time in seconds 
        # NOTE: ncclDevKernel_SendRecv durations are in MILLISECONDS
        total_duration_s = sum(valid_durations) / 1000.0  # ms to seconds
        end_time_s = start_time_s + total_duration_s
        
        # Calculate bandwidth in MB/s
        # bandwidth = total_bytes / total_seconds / 1e6 (for MB/s)
        total_data_bytes = size * len(valid_durations)
        bandwidth_mbps = (total_data_bytes / total_duration_s) / 1e6  # bytes/s to MB/s
        
        records.append({
            'config_id': config_id,
            'src': src,
            'dst': dst,
            'start_time_s': start_time_s,
            'end_time_s': end_time_s,
            'duration_s': total_duration_s,
            'bandwidth_mbps': bandwidth_mbps,
            'size': size,
            'num_reps': len(valid_durations)
        })
    
    return pd.DataFrame(records)


def calculate_aggregate_bandwidth(execution_df, temp_df, time_bin_s=1.0):
    """
    Calculate aggregate data throughput over time bins.
    
    For each time bin, calculate total data transferred / time_bin_s.
    Uses the full temperature monitoring duration for time range.
    """
    if len(execution_df) == 0:
        return pd.DataFrame()
    
    # Determine time range from temperature data (full benchmark duration)
    start_time = temp_df['elapsed_s'].min()
    end_time = temp_df['elapsed_s'].max()
    
    # Create time bins
    time_bins = np.arange(start_time, end_time + time_bin_s, time_bin_s)
    
    aggregate_bw = []
    
    for t in time_bins:
        # Find all executions active during this time bin
        # An execution is active if any part of it overlaps with [t, t+time_bin_s]
        bin_start = t
        bin_end = t + time_bin_s
        
        active = execution_df[
            (execution_df['start_time_s'] < bin_end) & 
            (execution_df['end_time_s'] > bin_start)
        ]
        
        # For each active execution, calculate how much data was transferred in this bin
        total_bytes_in_bin = 0
        for _, row in active.iterrows():
            # Calculate overlap between execution and bin
            overlap_start = max(row['start_time_s'], bin_start)
            overlap_end = min(row['end_time_s'], bin_end)
            overlap_duration = overlap_end - overlap_start
            
            # Calculate data transferred during overlap
            # Assume uniform data rate throughout execution
            total_data = row['size'] * row['num_reps']
            data_rate = total_data / row['duration_s']  # bytes/second
            bytes_in_bin = data_rate * overlap_duration
            total_bytes_in_bin += bytes_in_bin
        
        # Convert to MB/s
        bandwidth_mbps = (total_bytes_in_bin / time_bin_s) / 1e6
        
        aggregate_bw.append({
            'time_s': t,
            'aggregate_bandwidth_mbps': bandwidth_mbps,
            'num_active': len(active)
        })
    
    return pd.DataFrame(aggregate_bw)


def prepare_power_data(temp_df):
    """
    Prepare power data with statistics over time.
    
    Returns time series of mean, min, max power.
    """
    # Group by time and calculate power statistics across all GPUs
    power_stats = temp_df.groupby('elapsed_s')['power'].agg(['mean', 'min', 'max'])
    
    return power_stats


def plot_bandwidth_power_timeline(execution_df, temp_df, output_path, time_bin_s=1.0):
    """
    Create dual-axis plot with aggregate bandwidth and power over time.
    """
    # Calculate aggregate bandwidth
    agg_bw_df = calculate_aggregate_bandwidth(execution_df, temp_df, time_bin_s)
    
    if len(agg_bw_df) == 0:
        print("Error: No aggregate bandwidth data!")
        return
    
    # Prepare power data
    power_stats = prepare_power_data(temp_df)
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Plot aggregate bandwidth on left axis (black color)
    color_bw = 'black'
    ax1.set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Aggregate Bandwidth (MB/s)', fontsize=12, fontweight='bold', color=color_bw)
    
    # Scatter plot of bandwidth
    ax1.scatter(agg_bw_df['time_s'], agg_bw_df['aggregate_bandwidth_mbps'],
               s=10, alpha=0.4, color=color_bw, edgecolors='none')
    
    # Calculate Huber regression for bandwidth
    if len(agg_bw_df) >= 2:
        X = agg_bw_df['time_s'].values.reshape(-1, 1)
        y = agg_bw_df['aggregate_bandwidth_mbps'].values
        huber = HuberRegressor(epsilon=1.35, max_iter=1000)
        huber.fit(X, y)
        huber_bw = huber.predict(X)
        
        ax1.plot(agg_bw_df['time_s'], huber_bw, '-', 
                color=color_bw, linewidth=2.5, alpha=0.8,
                label=f'Huber BW (Mean: {np.mean(huber_bw):.0f} MB/s)')
    
    ax1.tick_params(axis='y', labelcolor=color_bw)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    # Create second y-axis for power
    ax2 = ax1.twinx()
    color_power = 'darkgray'
    ax2.set_ylabel('Power Draw (W)', fontsize=12, fontweight='bold', color=color_power)
    
    # Plot power statistics
    time_points = power_stats.index.values
    power_mean = power_stats['mean'].values
    power_min = power_stats['min'].values
    power_max = power_stats['max'].values
    
    # Plot mean power as line
    ax2.plot(time_points, power_mean, '-', 
            color=color_power, linewidth=2.5, alpha=0.8,
            label=f'Mean Power ({np.mean(power_mean):.1f} W)')
    
    # Plot fins (fill_between) for min/max range
    ax2.fill_between(time_points, power_min, power_max,
                     color=color_power, alpha=0.2,
                     label='Power Range')
    
    ax2.tick_params(axis='y', labelcolor=color_power)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=10)
    
    # Set title
    plt.title('Aggregate Network Bandwidth and GPU Power Over Time\n' +
             f'(Bandwidth aggregated over {time_bin_s}s bins)', 
             fontsize=14, fontweight='bold', pad=20)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved bandwidth-power timeline plot to {output_path}")


def print_statistics(execution_df, agg_bw_df, temp_df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("BANDWIDTH AND POWER TIMELINE STATISTICS")
    print("="*80)
    
    print(f"\nExecution timeline:")
    print(f"  Total configurations: {len(execution_df)}")
    print(f"  First config starts: {execution_df['start_time_s'].min():.2f}s")
    print(f"  Last config ends: {execution_df['end_time_s'].max():.2f}s")
    print(f"  Total benchmark duration (from temp data): {temp_df['elapsed_s'].max():.2f}s")
    print(f"  ({temp_df['elapsed_s'].max() / 60:.2f} minutes)")
    
    print(f"\nAggregate bandwidth:")
    print(f"  Mean: {agg_bw_df['aggregate_bandwidth_mbps'].mean():.2f} MB/s")
    print(f"  Median: {agg_bw_df['aggregate_bandwidth_mbps'].median():.2f} MB/s")
    print(f"  Std: {agg_bw_df['aggregate_bandwidth_mbps'].std():.2f} MB/s")
    print(f"  Min: {agg_bw_df['aggregate_bandwidth_mbps'].min():.2f} MB/s")
    print(f"  Max: {agg_bw_df['aggregate_bandwidth_mbps'].max():.2f} MB/s")
    
    print(f"\nPower statistics:")
    power_stats = temp_df.groupby('elapsed_s')['power'].agg(['mean', 'min', 'max'])
    print(f"  Overall mean power: {power_stats['mean'].mean():.2f} W")
    print(f"  Power range: {power_stats['min'].min():.2f} - {power_stats['max'].max():.2f} W")
    
    print(f"\nConcurrency:")
    print(f"  Max concurrent transfers: {agg_bw_df['num_active'].max()}")
    print(f"  Mean concurrent transfers: {agg_bw_df['num_active'].mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot aggregate bandwidth and power timeline'
    )
    parser.add_argument('json_path', type=str, help='Path to trace_analysis.json')
    parser.add_argument('--time-bin', type=float, default=1.0,
                       help='Time bin size in seconds (default: 1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for plot (default: plots/bandwidth_power_timeline.png)')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Check for temperature data
    temp_csv_path = json_path.parent / 'temperatures.csv'
    if not temp_csv_path.exists():
        print(f"Error: temperatures.csv not found: {temp_csv_path}")
        print("This analysis requires temperature/power data.")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = json_path.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'bandwidth_power_timeline.png'
    
    print("="*80)
    print("AGGREGATE BANDWIDTH AND POWER TIMELINE ANALYSIS")
    print("="*80)
    
    print(f"\nLoading data from: {json_path}")
    trace_data = load_trace_json(json_path)
    
    print("Extracting execution timeline...")
    execution_df = extract_execution_timeline(trace_data)
    
    if len(execution_df) == 0:
        print("Error: No execution data extracted!")
        return 1
    
    print(f"✓ Extracted {len(execution_df)} configurations")
    
    print(f"\nLoading temperature/power data from: {temp_csv_path}")
    temp_df = load_temperature_data(temp_csv_path)
    print(f"✓ Loaded {len(temp_df)} temperature/power readings")
    
    print(f"\nCalculating aggregate bandwidth (time bins: {args.time_bin}s)...")
    agg_bw_df = calculate_aggregate_bandwidth(execution_df, temp_df, args.time_bin)
    print(f"✓ Created {len(agg_bw_df)} time bins")
    
    # Print statistics
    print_statistics(execution_df, agg_bw_df, temp_df)
    
    # Create plot
    print(f"\nGenerating plot...")
    plot_bandwidth_power_timeline(execution_df, temp_df, output_path, args.time_bin)
    
    # Save detailed data
    csv_path = output_path.parent / 'bandwidth_power_timeline_data.csv'
    agg_bw_df.to_csv(csv_path, index=False)
    print(f"✓ Saved timeline data to {csv_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

