#!/usr/bin/env python3
"""
Plot the ratio of NCCL kernel time to CPU duration vs message size.

Shows how much of the CPU-measured time is actual GPU kernel execution.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_trace_json(json_path):
    """Load the trace analysis JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_ratio_data(data):
    """
    Extract kernel/CPU ratio data from the JSON.
    
    Returns: List of (size, ratio, src, dst, repetition) tuples
    """
    results = []
    
    for config_key, config_data in data.items():
        meta = config_data['metadata']
        src = meta['src']
        dst = meta['dst']
        size = meta['size']
        
        cpu_durations = config_data['cpu_duration']
        
        # Find the NCCL kernel operation
        kernel_name = None
        for op_name in config_data['gpu_duration'].keys():
            if 'ncclDevKernel_SendRecv' in op_name:
                kernel_name = op_name
                break
        
        if not kernel_name:
            continue
        
        kernel_durations = config_data['gpu_duration'][kernel_name]
        
        # For each repetition with data
        for rep_idx, (cpu_dur, kernel_dur) in enumerate(zip(cpu_durations, kernel_durations)):
            if cpu_dur is not None and kernel_dur is not None and cpu_dur > 0:
                ratio = kernel_dur / cpu_dur
                results.append((size, ratio, src, dst, rep_idx))
    
    return results


def create_link_id(src, dst):
    """Create a unique identifier for a src-dst link."""
    return f"{src}→{dst}"


def plot_kernel_cpu_ratio(data, output_path):
    """Create the plot."""
    # Extract data
    ratio_data = extract_ratio_data(data)
    
    if not ratio_data:
        print("No data with both kernel and CPU durations found!")
        return
    
    print(f"Extracted {len(ratio_data)} data points")
    
    # Organize by link
    links = {}
    for size, ratio, src, dst, rep in ratio_data:
        link_id = create_link_id(src, dst)
        if link_id not in links:
            links[link_id] = {'sizes': [], 'ratios': [], 'src': src, 'dst': dst}
        links[link_id]['sizes'].append(size)
        links[link_id]['ratios'].append(ratio)
    
    print(f"Found {len(links)} unique links")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get colors from magma colormap
    n_links = len(links)
    colors = plt.cm.magma(np.linspace(0.1, 0.9, n_links))
    
    # Plot each link
    for (link_id, link_data), color in zip(sorted(links.items()), colors):
        sizes = np.array(link_data['sizes'])
        ratios = np.array(link_data['ratios'])
        
        # Sort by size for cleaner plotting
        sort_idx = np.argsort(sizes)
        sizes = sizes[sort_idx]
        ratios = ratios[sort_idx]
        
        ax.scatter(sizes, ratios, c=[color], label=link_id, 
                  alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Add horizontal line at y=1
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
               label='Kernel = CPU time', zorder=0)
    
    # Set log scale for both axes
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Message Size (bytes)', fontsize=12)
    ax.set_ylabel('Ratio: NCCL Kernel Time / CPU Duration (log scale)', fontsize=12)
    ax.set_title('GPU Kernel vs CPU Duration Ratio by Message Size', fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Legend
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    # Format x-axis to show powers of 2
    from matplotlib.ticker import FuncFormatter
    def size_formatter(x, pos):
        if x < 1024:
            return f'{int(x)}B'
        elif x < 1024**2:
            return f'{int(x/1024)}KB'
        elif x < 1024**3:
            return f'{int(x/(1024**2))}MB'
        else:
            return f'{int(x/(1024**3))}GB'
    
    ax.xaxis.set_major_formatter(FuncFormatter(size_formatter))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved successfully")
    
    # Show statistics
    print("\nStatistics:")
    all_ratios = [ratio for _, ratio, _, _, _ in ratio_data]
    print(f"  Min ratio: {min(all_ratios):.4f}")
    print(f"  Max ratio: {max(all_ratios):.4f}")
    print(f"  Mean ratio: {np.mean(all_ratios):.4f}")
    print(f"  Median ratio: {np.median(all_ratios):.4f}")
    
    # Count points above/below 1
    above_1 = sum(1 for r in all_ratios if r > 1)
    below_1 = sum(1 for r in all_ratios if r < 1)
    print(f"\n  Points where kernel > CPU: {above_1} ({100*above_1/len(all_ratios):.1f}%)")
    print(f"  Points where kernel < CPU: {below_1} ({100*below_1/len(all_ratios):.1f}%)")


def plot_bar_chart(data, output_path):
    """Create bar chart showing cumulative average across all permutations with worst-case coloring."""
    # Extract data
    ratio_data = extract_ratio_data(data)
    
    if not ratio_data:
        print("No data for bar chart!")
        return
    
    # Organize by size (aggregating across all links)
    size_data = {}
    for size, ratio, src, dst, rep in ratio_data:
        if size not in size_data:
            size_data[size] = []
        size_data[size].append(ratio)
    
    # Calculate statistics for each size
    stats = []
    for size, ratios in size_data.items():
        mean_ratio = np.mean(ratios)
        min_ratio = np.min(ratios)
        max_ratio = np.max(ratios)
        
        # Calculate worst case deviation from 1
        deviation_from_min = abs(min_ratio - 1.0)
        deviation_from_max = abs(max_ratio - 1.0)
        worst_case_deviation = max(deviation_from_min, deviation_from_max)
        
        stats.append({
            'size': size,
            'mean': mean_ratio,
            'min': min_ratio,
            'max': max_ratio,
            'worst_case_deviation': worst_case_deviation
        })
    
    # Sort by size
    stats = sorted(stats, key=lambda x: x['size'])
    
    print(f"\nBar chart: {len(stats)} unique message sizes")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data for plotting
    sizes = [s['size'] for s in stats]
    means = [s['mean'] for s in stats]
    mins = [s['min'] for s in stats]
    maxs = [s['max'] for s in stats]
    worst_deviations = [s['worst_case_deviation'] for s in stats]
    
    # Normalize worst case deviations for colormap (log scale)
    # Add small epsilon to avoid log(0)
    log_deviations = np.log10(np.array(worst_deviations) + 1e-10)
    
    # Normalize to [0, 1] for colormap
    if log_deviations.max() > log_deviations.min():
        normalized_deviations = (log_deviations - log_deviations.min()) / (log_deviations.max() - log_deviations.min())
    else:
        normalized_deviations = np.ones_like(log_deviations) * 0.5
    
    # Get colors from magma colormap
    colors = plt.cm.magma(normalized_deviations)
    
    # Plot bars
    x_positions = np.arange(len(sizes))
    bars = ax.bar(x_positions, means, 
                  color=colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    
    # Add error bars (min to max range)
    errors_lower = [mean - min_val for mean, min_val in zip(means, mins)]
    errors_upper = [max_val - mean for mean, max_val in zip(means, maxs)]
    
    ax.errorbar(x_positions, means, 
               yerr=[errors_lower, errors_upper],
               fmt='none', ecolor='black', capsize=4, 
               linewidth=1.5, alpha=0.7)
    
    # Add horizontal line at y=1
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2.5, 
               label='Kernel = CPU time', zorder=0, alpha=0.8)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel('Message Size', fontsize=13)
    ax.set_ylabel('Average Ratio (log scale)', fontsize=13)
    ax.set_title('GPU Kernel / CPU Duration Ratio (Averaged Across All Links)\nError bars show min-max range, color shows worst-case deviation from 1', 
                fontsize=14, fontweight='bold')
    
    # Set x-ticks
    ax.set_xticks(x_positions)
    
    # Format x-axis labels
    from matplotlib.ticker import FuncFormatter
    def size_formatter_idx(x, pos):
        idx = int(x)
        if 0 <= idx < len(sizes):
            size = sizes[idx]
            if size < 1024:
                return f'{int(size)}B'
            elif size < 1024**2:
                return f'{int(size/1024)}K'
            elif size < 1024**3:
                return f'{int(size/(1024**2))}M'
            else:
                return f'{int(size/(1024**3))}G'
        return ''
    
    ax.xaxis.set_major_formatter(FuncFormatter(size_formatter_idx))
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both', axis='y')
    
    # Add colorbar to show deviation scale
    sm = plt.cm.ScalarMappable(cmap='magma', 
                               norm=plt.Normalize(vmin=log_deviations.min(), 
                                                 vmax=log_deviations.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Worst-case deviation from 1 (log₁₀)', fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    print(f"Saving bar chart to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Bar chart saved successfully")
    
    # Print some statistics
    print(f"\n  Average ratio statistics:")
    print(f"    Mean across all: {np.mean(means):.2f}")
    print(f"    Median: {np.median(means):.2f}")
    print(f"    Sizes with mean > 1: {sum(1 for m in means if m > 1)}/{len(means)}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot NCCL kernel / CPU duration ratio vs message size'
    )
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to trace_analysis.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output plot path (default: same directory as JSON with .png extension)'
    )
    parser.add_argument(
        '--bar-chart',
        action='store_true',
        help='Also generate bar chart with error bars'
    )
    
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.parent / 'kernel_cpu_ratio.png'
    
    # Load data
    print(f"Loading data from: {json_path}")
    data = load_trace_json(json_path)
    print(f"Loaded {len(data)} configurations")
    
    # Create scatter plot
    plot_kernel_cpu_ratio(data, output_path)
    
    # Create bar chart if requested
    if args.bar_chart:
        bar_output_path = output_path.parent / 'kernel_cpu_ratio_bars.png'
        plot_bar_chart(data, bar_output_path)
    
    return 0


if __name__ == '__main__':
    exit(main())

