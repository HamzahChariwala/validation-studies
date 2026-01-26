"""
Visualize matmul performance normalised by output tensor data size.
Creates line plots showing duration per bit of output data.
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


def extract_data_for_plot(database: dict, op_name: str) -> dict:
    """
    Extract data for plotting from database, grouped by precision.
    Normalise durations by output tensor data size (M×N × bits_per_element).
    
    Args:
        database: The matmul database
        op_name: Operation name to filter by (e.g., 'aten::mm', 'aten::matmul')
    
    Returns:
        Dict mapping dtype to (sizes, normalised_durations, normalised_stds) arrays
    """
    data_by_dtype = {}
    
    for key, entry in database.items():
        # Filter by operation name and batch size
        if entry['op_name'] != op_name:
            continue
        
        config = entry['config']
        if config['batch_size'] != 1:
            continue
        
        # Skip transposed operations for cleaner comparison
        if config['transpose_a'] or config['transpose_b']:
            continue
        
        # Get matrix dimensions (output is M×N)
        m = config['m']
        n = config['n']
        size = m  # For square matrices
        dtype = config['dtype']
        
        # Calculate output tensor data size in bits
        output_elements = m * n
        bits_per_element = get_bits_per_element(dtype)
        output_data_bits = output_elements * bits_per_element
        
        # Normalise duration by output data size (microseconds per bit)
        duration = entry['statistics']['mean']
        std_dev = entry['statistics']['std']
        
        normalised_duration = duration / output_data_bits
        normalised_std = std_dev / output_data_bits
        
        if dtype not in data_by_dtype:
            data_by_dtype[dtype] = {'sizes': [], 'durations': [], 'stds': []}
        
        data_by_dtype[dtype]['sizes'].append(size)
        data_by_dtype[dtype]['durations'].append(normalised_duration)
        data_by_dtype[dtype]['stds'].append(normalised_std)
    
    # Convert to numpy arrays and sort by size
    for dtype in data_by_dtype:
        sizes = np.array(data_by_dtype[dtype]['sizes'])
        durations = np.array(data_by_dtype[dtype]['durations'])
        stds = np.array(data_by_dtype[dtype]['stds'])
        
        # Sort by size
        sort_idx = np.argsort(sizes)
        data_by_dtype[dtype] = {
            'sizes': sizes[sort_idx],
            'durations': durations[sort_idx],
            'stds': stds[sort_idx]
        }
    
    return data_by_dtype


def create_line_plot(ax, data_by_dtype, title):
    """
    Create line plots for different precisions.
    
    Args:
        ax: Matplotlib axis object
        data_by_dtype: Dict mapping dtype to {sizes, durations, stds}
        title: Plot title
    """
    # Define colors from magma colormap for each precision
    magma = plt.colormaps['magma']
    colors = {
        'fp32': magma(0.2),   # Dark purple
        'fp16': magma(0.5),   # Orange
        'bf16': magma(0.8),   # Yellow
    }
    
    # Plot each precision
    for dtype in sorted(data_by_dtype.keys()):
        data = data_by_dtype[dtype]
        sizes = data['sizes']
        durations = data['durations']
        stds = data['stds']
        
        # Plot line with markers
        ax.plot(
            sizes, durations,
            marker='o',
            markersize=8,
            linewidth=2.5,
            color=colors.get(dtype, 'gray'),
            label=dtype.upper(),
            alpha=0.9
        )
    
    # Labels and title
    ax.set_xlabel('Matrix Dimension (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Duration per Output Bit (μs/bit)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Log scale for x-axis
    ax.set_xscale('log')
    # Linear or log scale for y-axis depending on range
    y_range = []
    for data in data_by_dtype.values():
        y_range.extend(data['durations'])
    if len(y_range) > 0:
        y_min, y_max = min(y_range), max(y_range)
        if y_max / y_min > 100:
            ax.set_yscale('log')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    
    # Legend
    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='gray')
    
    # Set x-axis ticks to actual matrix sizes
    if len(data_by_dtype) > 0:
        first_dtype = list(data_by_dtype.keys())[0]
        sizes = data_by_dtype[first_dtype]['sizes']
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{int(s)}' for s in sizes])
    
    return ax


def main():
    """Main execution function."""
    # Load database
    db_path = Path(__file__).parent / "matmul_database.json"
    print(f"Loading database from {db_path}")
    database = load_database(db_path)
    
    # Extract data for both operation types, grouped by precision
    print("Extracting data for aten::mm...")
    mm_data = extract_data_for_plot(database, 'aten::mm')
    print(f"  Found data for precisions: {list(mm_data.keys())}")
    
    print("Extracting data for aten::matmul...")
    matmul_data = extract_data_for_plot(database, 'aten::matmul')
    print(f"  Found data for precisions: {list(matmul_data.keys())}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot aten::mm
    print("Creating aten::mm plot...")
    create_line_plot(
        ax1, mm_data,
        'aten::mm Performance (normalised by output data)'
    )
    
    # Plot aten::matmul
    print("Creating aten::matmul plot...")
    create_line_plot(
        ax2, matmul_data,
        'aten::matmul Performance (normalised by output data)'
    )
    
    # Overall title
    fig.suptitle(
        'Matrix Multiplication Efficiency: Duration per Output Bit',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    # Add note about normalisation
    fig.text(
        0.5, 0.01,
        'Normalised by output tensor size: M×N × bits_per_element (batch_size=1, no transpose)',
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "matmul_normalised_plot.png"
    print(f"\nSaving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
    
    # Show summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (NORMALISED)")
    print("="*80)
    
    for op_name, data in [('aten::mm', mm_data), ('aten::matmul', matmul_data)]:
        print(f"\n{op_name}:")
        for dtype in sorted(data.keys()):
            durations = data[dtype]['durations']
            sizes = data[dtype]['sizes']
            bits = get_bits_per_element(dtype)
            print(f"  {dtype.upper()} ({bits} bits/element):")
            print(f"    Data points: {len(durations)}")
            print(f"    Matrix sizes: {sizes.tolist()}")
            print(f"    Duration/bit range: {durations.min():.6f} - {durations.max():.6f} μs/bit")
            print(f"    Mean duration/bit: {durations.mean():.6f} μs/bit")
            print(f"    Efficiency trend: {'improving' if durations[-1] < durations[0] else 'degrading'} with size")
    
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()

