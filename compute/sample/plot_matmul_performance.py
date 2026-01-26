"""
Visualize matmul performance across different tensor dimensions.
Creates 2D scatterplots with contours for aten::mm and aten::matmul operations.
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


def extract_data_for_plot(database: dict, op_name: str) -> dict:
    """
    Extract data for plotting from database, grouped by precision.
    
    Args:
        database: The matmul database
        op_name: Operation name to filter by (e.g., 'aten::mm', 'aten::matmul')
    
    Returns:
        Dict mapping dtype to (sizes, durations, std_devs) arrays
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
        
        # Get matrix size (square matrices, so M=N=K)
        size = config['m']
        dtype = config['dtype']
        duration = entry['statistics']['mean']
        std_dev = entry['statistics']['std']
        
        if dtype not in data_by_dtype:
            data_by_dtype[dtype] = {'sizes': [], 'durations': [], 'stds': []}
        
        data_by_dtype[dtype]['sizes'].append(size)
        data_by_dtype[dtype]['durations'].append(duration)
        data_by_dtype[dtype]['stds'].append(std_dev)
    
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
    ax.set_ylabel('Mean Duration (μs)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Log scale for both axes (performance scales non-linearly with size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.5, which='minor')
    
    # Legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='gray')
    
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
        'aten::mm Performance (batch_size=1, no transpose)'
    )
    
    # Plot aten::matmul
    print("Creating aten::matmul plot...")
    create_line_plot(
        ax2, matmul_data,
        'aten::matmul Performance (batch_size=1, no transpose)'
    )
    
    # Overall title
    fig.suptitle(
        'Matrix Multiplication Performance: Size and Precision Scaling',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "matmul_performance_plot.png"
    print(f"\nSaving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
    
    # Show summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for op_name, data in [('aten::mm', mm_data), ('aten::matmul', matmul_data)]:
        print(f"\n{op_name}:")
        for dtype in sorted(data.keys()):
            durations = data[dtype]['durations']
            sizes = data[dtype]['sizes']
            print(f"  {dtype.upper()}:")
            print(f"    Data points: {len(durations)}")
            print(f"    Matrix sizes: {sizes.tolist()}")
            print(f"    Duration range: {durations.min():.2f} - {durations.max():.2f} μs")
            print(f"    Mean duration: {durations.mean():.2f} μs")
    
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()

