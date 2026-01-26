#!/usr/bin/env python3
"""
Script to plot temperature and power draw over time from temperature CSV files.

Creates a dual-axis plot with:
- Left Y-axis: Power draw per GPU (colored lines using magma colormap)
- Right Y-axis: Mean temperature with error bars showing range (dashed black line)
- X-axis: Elapsed time in seconds

Usage:
    python plot_temperature_power.py <path_to_temperatures.csv>
    
The plot will be saved to the 'plots' subfolder in the same directory as the CSV file.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys


def load_temperature_data(csv_path):
    """Load and parse the temperature CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['elapsed_s', 'gpu_id', 'temperature', 'power']
        
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            print(f"Found columns: {list(df.columns)}")
            sys.exit(1)
            
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)


def prepare_plot_data(df):
    """
    Prepare data for plotting.
    
    Returns:
        time_points: Unique sorted elapsed time values
        power_by_gpu: Dictionary mapping gpu_id to power values at each time point
        temp_mean: Mean temperature at each time point
        temp_min: Minimum temperature at each time point
        temp_max: Maximum temperature at each time point
    """
    # Get unique time points (sorted)
    time_points = sorted(df['elapsed_s'].unique())
    
    # Get unique GPU IDs
    gpu_ids = sorted(df['gpu_id'].unique())
    
    # Prepare power data for each GPU
    power_by_gpu = {}
    for gpu_id in gpu_ids:
        gpu_df = df[df['gpu_id'] == gpu_id].sort_values('elapsed_s')
        power_by_gpu[gpu_id] = gpu_df['power'].values
    
    # Calculate temperature statistics at each time point
    temp_stats = df.groupby('elapsed_s')['temperature'].agg(['mean', 'min', 'max'])
    temp_mean = temp_stats['mean'].values
    temp_min = temp_stats['min'].values
    temp_max = temp_stats['max'].values
    
    return time_points, power_by_gpu, temp_mean, temp_min, temp_max


def create_plot(time_points, power_by_gpu, temp_mean, temp_min, temp_max, output_path):
    """
    Create the dual-axis temperature and power plot.
    
    Args:
        time_points: Array of elapsed time values
        power_by_gpu: Dictionary mapping gpu_id to power values
        temp_mean: Mean temperature values
        temp_min: Minimum temperature values
        temp_max: Maximum temperature values
        output_path: Path where the plot will be saved
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot power on left y-axis
    gpu_ids = sorted(power_by_gpu.keys())
    n_gpus = len(gpu_ids)
    
    # Use magma colormap
    colors = plt.cm.magma(np.linspace(0.2, 0.9, n_gpus))
    
    for idx, gpu_id in enumerate(gpu_ids):
        ax1.plot(time_points, power_by_gpu[gpu_id], 
                label=f'GPU {gpu_id}', 
                color=colors[idx], 
                linewidth=2,
                alpha=0.8)
    
    ax1.set_xlabel('Elapsed Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Power Draw (W)', fontsize=12, fontweight='bold', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.9)
    
    # Create secondary y-axis for temperature
    ax2 = ax1.twinx()
    
    # Calculate error bars (show range from min to max)
    error_lower = temp_mean - temp_min
    error_upper = temp_max - temp_mean
    
    # Plot mean temperature with error bars
    ax2.errorbar(time_points, temp_mean, 
                yerr=[error_lower, error_upper],
                fmt='k--',  # Black dashed line
                linewidth=2,
                capsize=3,
                capthick=1,
                elinewidth=1,
                alpha=0.7,
                label='Mean Temperature (±range)',
                markersize=0)
    
    ax2.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Set title
    plt.title('GPU Power Draw and Temperature Over Time', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Adjust layout to prevent label cutoff
    fig.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot temperature and power data from a temperatures CSV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_temperature_power.py traces/run_20260121_164227/temperatures.csv
    
This will create a plot at:
    traces/run_20260121_164227/plots/temperature_power.png
        """
    )
    parser.add_argument('csv_path', type=str, 
                       help='Path to the temperatures.csv file')
    parser.add_argument('--output-name', type=str, default='temperature_power.png',
                       help='Name of the output plot file (default: temperature_power.png)')
    
    args = parser.parse_args()
    
    # Validate input file
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    if not csv_path.is_file():
        print(f"Error: Not a file: {csv_path}")
        sys.exit(1)
    
    # Determine output directory (plots subfolder in the same directory as CSV)
    run_dir = csv_path.parent
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    output_path = plots_dir / args.output_name
    
    print(f"Loading temperature data from: {csv_path}")
    df = load_temperature_data(csv_path)
    
    print(f"Processing {len(df)} records...")
    print(f"  - Time range: {df['elapsed_s'].min():.2f}s to {df['elapsed_s'].max():.2f}s")
    print(f"  - GPUs found: {sorted(df['gpu_id'].unique())}")
    print(f"  - Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")
    print(f"  - Power range: {df['power'].min():.2f}W to {df['power'].max():.2f}W")
    
    time_points, power_by_gpu, temp_mean, temp_min, temp_max = prepare_plot_data(df)
    
    print(f"Creating plot...")
    create_plot(time_points, power_by_gpu, temp_mean, temp_min, temp_max, output_path)
    
    print(f"Done!")


if __name__ == '__main__':
    main()

