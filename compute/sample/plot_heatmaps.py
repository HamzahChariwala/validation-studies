"""
Create heatmaps showing matmul performance across size and batch dimensions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
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


def extract_heatmap_data(database: dict) -> dict:
    """
    Extract data for heatmaps: size × batch × dtype.
    Normalised by output tensor size.
    
    Returns:
        Dict mapping dtype -> (sizes, batches, values_2d_array)
    """
    # Collect all data points
    data_points = {}
    
    for key, entry in database.items():
        # Only aten::mm and aten::bmm, no transpose
        if entry['op_name'] not in ['aten::mm', 'aten::bmm']:
            continue
        
        config = entry['config']
        if config['transpose_a'] or config['transpose_b']:
            continue
        
        # Get parameters
        size = config['m']
        batch = config['batch_size']
        dtype = config['dtype']
        
        # Normalise by output data size
        output_elements = batch * size * size
        bits_per_element = get_bits_per_element(dtype)
        output_data_bits = output_elements * bits_per_element
        
        duration = entry['statistics']['mean']
        normalised_duration = duration / output_data_bits
        
        if dtype not in data_points:
            data_points[dtype] = {}
        if size not in data_points[dtype]:
            data_points[dtype][size] = {}
        
        data_points[dtype][size][batch] = normalised_duration
    
    # Convert to structured format for heatmap
    heatmap_data = {}
    
    for dtype in data_points:
        sizes = sorted(data_points[dtype].keys())
        # Get all batch sizes from the data
        all_batches = set()
        for size_data in data_points[dtype].values():
            all_batches.update(size_data.keys())
        batches = sorted(all_batches)
        
        # Create 2D array
        values = np.zeros((len(batches), len(sizes)))
        for i, batch in enumerate(batches):
            for j, size in enumerate(sizes):
                if batch in data_points[dtype][size]:
                    values[i, j] = data_points[dtype][size][batch]
                else:
                    values[i, j] = np.nan
        
        heatmap_data[dtype] = {
            'sizes': sizes,
            'batches': batches,
            'values': values
        }
    
    return heatmap_data


def create_size_batch_heatmaps(heatmap_data: dict) -> None:
    """Create separate Size × Batch heatmaps for each precision."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    dtype_order = ['fp32', 'fp16', 'bf16']
    
    # Find global min/max for consistent color scale
    all_values = []
    for data in heatmap_data.values():
        all_values.extend(data['values'].flatten())
    all_values = [v for v in all_values if not np.isnan(v)]
    vmin, vmax = min(all_values), max(all_values)
    
    for ax, dtype in zip(axes, dtype_order):
        if dtype not in heatmap_data:
            continue
        
        data = heatmap_data[dtype]
        sizes = data['sizes']
        batches = data['batches']
        values = data['values']
        
        # Create heatmap with reversed y-axis (batch increases upward) and log color scale
        im = ax.imshow(
            values,
            cmap='magma_r',  # Reversed so darker = better (lower duration)
            aspect='auto',
            norm=LogNorm(vmin=vmin, vmax=vmax),  # Logarithmic color scale
            origin='lower'  # Reverse y-axis so batch increases upward
        )
        
        # Labels - display batch sizes with log spacing visually represented
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_yticks(np.arange(len(batches)))
        ax.set_xticklabels(sizes)
        ax.set_yticklabels([str(b) for b in batches])  # Display actual batch values
        
        ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Batch Size (log scale)', fontsize=12, fontweight='bold')
        
        bits = get_bits_per_element(dtype)
        ax.set_title(f'{dtype.upper()} ({bits} bits/element)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(len(batches)):
            for j in range(len(sizes)):
                if not np.isnan(values[i, j]):
                    text = ax.text(j, i, f'{values[i, j]:.5f}',
                                 ha="center", va="center", color="white", fontsize=8)
    
    # Title
    fig.suptitle(
        'Matrix Multiplication Efficiency: Size × Batch Heatmaps',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    # Adjust layout to make room for colorbar at the bottom
    plt.subplots_adjust(bottom=0.2)
    
    # Colorbar - positioned at the bottom with explicit positioning
    fig.colorbar(im, ax=axes, label='Duration per Output Bit (μs/bit)', 
                 orientation='horizontal', pad=0.15, aspect=40, shrink=0.8)
    
    # Save
    output_path = Path(__file__).parent / "heatmap_size_batch.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Size × Batch heatmap to: {output_path}")
    plt.close()


def create_composite_heatmap(heatmap_data: dict) -> None:
    """Create composite Size × Precision × Batch grid."""
    dtype_order = ['fp32', 'fp16', 'bf16']
    
    # Get dimensions
    all_sizes = set()
    all_batches = set()
    for data in heatmap_data.values():
        all_sizes.update(data['sizes'])
        all_batches.update(data['batches'])
    
    sizes = sorted(all_sizes)
    batches = sorted(all_batches)
    n_dtypes = len(dtype_order)
    
    # Create large composite array: (batches × dtypes) × sizes
    composite_height = len(batches) * n_dtypes
    composite_values = np.zeros((composite_height, len(sizes)))
    
    # Fill composite array
    for dtype_idx, dtype in enumerate(dtype_order):
        if dtype not in heatmap_data:
            continue
        
        data = heatmap_data[dtype]
        
        for batch_idx, batch in enumerate(batches):
            row = batch_idx * n_dtypes + dtype_idx
            
            if batch_idx < len(data['batches']) and data['batches'][batch_idx] == batch:
                for size_idx, size in enumerate(sizes):
                    if size_idx < len(data['sizes']) and data['sizes'][size_idx] == size:
                        composite_values[row, size_idx] = data['values'][batch_idx, size_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Find global min/max
    valid_values = composite_values[~np.isnan(composite_values)]
    vmin, vmax = valid_values.min(), valid_values.max()
    
    # Create heatmap
    im = ax.imshow(
        composite_values,
        cmap='magma_r',
        aspect='auto',
        vmin=vmin,
        vmax=vmax
    )
    
    # X-axis (sizes)
    ax.set_xticks(np.arange(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=12, fontweight='bold')
    
    # Y-axis (batch × dtype)
    y_labels = []
    y_ticks = []
    for batch_idx, batch in enumerate(batches):
        for dtype_idx, dtype in enumerate(dtype_order):
            y_ticks.append(batch_idx * n_dtypes + dtype_idx)
            batch_label = f'B={batch}' if batch > 1 else 'None'
            y_labels.append(f'{batch_label}, {dtype.upper()}')
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_ylabel('Batch Size, Precision', fontsize=12, fontweight='bold')
    
    # Add horizontal lines between batch groups
    for batch_idx in range(1, len(batches)):
        ax.axhline(y=batch_idx * n_dtypes - 0.5, color='white', linewidth=2)
    
    # Title
    ax.set_title(
        'Composite Heatmap: Size × Precision × Batch',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Duration per Output Bit (μs/bit)')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(__file__).parent / "heatmap_composite.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved composite heatmap to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    # Load database
    db_path = Path(__file__).parent / "matmul_database.json"
    print(f"Loading database from {db_path}")
    database = load_database(db_path)
    
    # Extract data
    print("Extracting heatmap data...")
    heatmap_data = extract_heatmap_data(database)
    
    print(f"Found data for dtypes: {list(heatmap_data.keys())}")
    for dtype, data in heatmap_data.items():
        print(f"  {dtype}: {len(data['sizes'])} sizes × {len(data['batches'])} batches")
    
    # Create heatmaps
    print("\nCreating Size × Batch heatmaps...")
    create_size_batch_heatmaps(heatmap_data)
    
    print("\nCreating composite heatmap...")
    create_composite_heatmap(heatmap_data)
    
    print("\nAll heatmaps created successfully!")


if __name__ == "__main__":
    main()

