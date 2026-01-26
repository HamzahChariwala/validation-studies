"""
Visualize aten::mm performance across batch sizes for each precision.
Shows normalised duration per bit for different batch configurations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


def load_database(db_path: str) -> dict:
    """Load the matmul database."""
    with open(db_path, 'r') as f:
        return json.load(f)


def get_bits_per_element(dtype: str) -> int:
    """Get number of bits per element for a given dtype."""
    bits_map = {
        'fp32': 32,
        'fp16': 16,
        'bf16': 16,
        'int8': 8,
    }
    return bits_map.get(dtype, 32)


def extract_data_for_plot(database: dict) -> dict:
    """
    Extract aten::mm and aten::bmm data grouped by dtype, then by batch_size.
    Normalise durations by output tensor data size.
    
    Returns:
        Dict mapping dtype -> batch_size -> {sizes, durations}
    """
    data_by_dtype = {}
    
    for key, entry in database.items():
        # Include both aten::mm (non-batched) and aten::bmm (batched)
        if entry['op_name'] not in ['aten::mm', 'aten::bmm']:
            continue
        
        config = entry['config']
        
        # Skip transposed operations
        if config['transpose_a'] or config['transpose_b']:
            continue
        
        # Get parameters
        m = config['m']
        n = config['n']
        size = m  # For square matrices
        dtype = config['dtype']
        batch_size = config['batch_size']
        
        # Calculate output tensor data size in bits
        # For batched: batch_size × M × N × bits_per_element
        output_elements = batch_size * m * n
        bits_per_element = get_bits_per_element(dtype)
        output_data_bits = output_elements * bits_per_element
        
        # Normalise duration by output data size
        duration = entry['statistics']['mean']
        normalised_duration = duration / output_data_bits
        
        # Organize by dtype and batch_size
        if dtype not in data_by_dtype:
            data_by_dtype[dtype] = {}
        
        if batch_size not in data_by_dtype[dtype]:
            data_by_dtype[dtype][batch_size] = {'sizes': [], 'durations': []}
        
        data_by_dtype[dtype][batch_size]['sizes'].append(size)
        data_by_dtype[dtype][batch_size]['durations'].append(normalised_duration)
    
    # Convert to numpy arrays and sort by size
    for dtype in data_by_dtype:
        for batch_size in data_by_dtype[dtype]:
            sizes = np.array(data_by_dtype[dtype][batch_size]['sizes'])
            durations = np.array(data_by_dtype[dtype][batch_size]['durations'])
            
            # Sort by size
            sort_idx = np.argsort(sizes)
            data_by_dtype[dtype][batch_size] = {
                'sizes': sizes[sort_idx],
                'durations': durations[sort_idx]
            }
    
    return data_by_dtype


def create_batch_plot(ax, data_by_batch, dtype, title, ylim=None):
    """
    Create line plot showing different batch sizes.
    
    Args:
        ax: Matplotlib axis object
        data_by_batch: Dict mapping batch_size to {sizes, durations}
        dtype: Data type being plotted
        title: Plot title
        ylim: Optional tuple of (ymin, ymax) to set y-axis limits
    """
    # Define colors from magma colormap for each batch size
    magma = plt.colormaps['magma']
    batch_sizes = sorted(data_by_batch.keys())
    n_batches = len(batch_sizes)
    
    # Create color mapping
    colors = {}
    for i, batch_size in enumerate(batch_sizes):
        colors[batch_size] = magma(0.2 + 0.6 * i / max(n_batches - 1, 1))
    
    # Plot each batch size
    for batch_size in batch_sizes:
        data = data_by_batch[batch_size]
        sizes = data['sizes']
        durations = data['durations']
        
        label = f'B={batch_size}' if batch_size > 1 else 'No batch'
        
        # Plot line with markers
        ax.plot(
            sizes, durations,
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=colors[batch_size],
            label=label,
            alpha=0.9
        )
    
    # Labels and title
    ax.set_xlabel('Matrix Dimension (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration per Output Bit (μs/bit)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Log scale for x-axis
    ax.set_xscale('log')
    
    # Set y-axis scale and limits
    if ylim:
        y_min, y_max = ylim
        if y_max / y_min > 100:
            ax.set_yscale('log')
        ax.set_ylim(ylim)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.95, edgecolor='gray')
    
    # Set x-axis ticks to actual matrix sizes
    if len(data_by_batch) > 0:
        first_batch = list(data_by_batch.values())[0]
        sizes = first_batch['sizes']
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{int(s)}' for s in sizes])
    
    return ax


def main():
    """Main execution function."""
    # Load database
    db_path = Path(__file__).parent / "matmul_database.json"
    print(f"Loading database from {db_path}")
    database = load_database(db_path)
    
    # Extract data grouped by dtype and batch_size
    print("Extracting data for aten::mm and aten::bmm across batch sizes...")
    data = extract_data_for_plot(database)
    print(f"Found data for dtypes: {list(data.keys())}")
    for dtype in data:
        print(f"  {dtype}: batch sizes {sorted(data[dtype].keys())}")
    
    # Find global min and max across all dtypes for consistent y-axis scaling
    all_durations = []
    for dtype in data.values():
        for batch_data in dtype.values():
            all_durations.extend(batch_data['durations'])
    
    if len(all_durations) > 0:
        global_min = min(all_durations)
        global_max = max(all_durations)
        # Add some padding
        y_range = global_max - global_min
        ylim = (global_min * 0.5, global_max * 1.5)
        print(f"Global y-axis range: {global_min:.6f} - {global_max:.6f} μs/bit")
    else:
        ylim = None
    
    # Create figure with three subplots (one per dtype)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot each dtype with the same y-axis scale
    dtype_order = ['fp32', 'fp16', 'bf16']
    for ax, dtype in zip(axes, dtype_order):
        if dtype in data:
            print(f"Creating plot for {dtype.upper()}...")
            bits = get_bits_per_element(dtype)
            create_batch_plot(
                ax, data[dtype], dtype,
                f'aten::mm - {dtype.upper()} ({bits} bits/element)',
                ylim=ylim
            )
        else:
            print(f"No data for {dtype}")
    
    # Overall title
    fig.suptitle(
        'Batch Size Effect on Matrix Multiplication Efficiency',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )
    
    # Add note about normalisation
    fig.text(
        0.5, 0.01,
        'Normalised by output tensor size: batch_size × M × N × bits_per_element (no transpose)',
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "matmul_batch_effect_plot.png"
    print(f"\nSaving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
    
    # Show summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (BATCH EFFECT)")
    print("="*80)
    
    for dtype in dtype_order:
        if dtype not in data:
            continue
        print(f"\n{dtype.upper()}:")
        for batch_size in sorted(data[dtype].keys()):
            durations = data[dtype][batch_size]['durations']
            sizes = data[dtype][batch_size]['sizes']
            label = f"Batch {batch_size}" if batch_size > 1 else "No batch"
            print(f"  {label}:")
            print(f"    Matrix sizes: {sizes.tolist()}")
            print(f"    Duration/bit range: {durations.min():.6f} - {durations.max():.6f} μs/bit")
            print(f"    Mean duration/bit: {durations.mean():.6f} μs/bit")
    
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()

