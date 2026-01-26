"""
Visualize transpose overhead as ratios compared to no-transpose baseline.
Shows relative slowdown for different transpose configurations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_database(db_path: str) -> dict:
    """Load the matmul database."""
    with open(db_path, 'r') as f:
        return json.load(f)


def extract_transpose_data(database: dict) -> dict:
    """
    Extract aten::mm data grouped by dtype and transpose configuration.
    
    Returns:
        Dict mapping dtype -> size -> transpose_config -> duration
    """
    data_by_dtype = {}
    
    for key, entry in database.items():
        # Only aten::mm, batch_size=1
        if entry['op_name'] != 'aten::mm':
            continue
        
        config = entry['config']
        if config['batch_size'] != 1:
            continue
        
        # Get parameters
        size = config['m']  # Square matrices
        dtype = config['dtype']
        transpose_a = config['transpose_a']
        transpose_b = config['transpose_b']
        duration = entry['statistics']['mean']
        
        # Create transpose config key
        if not transpose_a and not transpose_b:
            trans_key = 'none'
        elif transpose_a and not transpose_b:
            trans_key = 'At'
        elif not transpose_a and transpose_b:
            trans_key = 'Bt'
        else:  # both
            trans_key = 'AtBt'
        
        # Organize data
        if dtype not in data_by_dtype:
            data_by_dtype[dtype] = {}
        if size not in data_by_dtype[dtype]:
            data_by_dtype[dtype][size] = {}
        
        data_by_dtype[dtype][size][trans_key] = duration
    
    return data_by_dtype


def compute_ratios(data_by_dtype: dict) -> dict:
    """
    Compute ratios relative to no-transpose baseline.
    
    Returns:
        Dict mapping dtype -> size -> transpose_config -> ratio
    """
    ratios = {}
    
    for dtype in data_by_dtype:
        ratios[dtype] = {}
        for size in data_by_dtype[dtype]:
            if 'none' not in data_by_dtype[dtype][size]:
                continue  # Skip if no baseline
            
            baseline = data_by_dtype[dtype][size]['none']
            ratios[dtype][size] = {}
            
            for trans_key in ['At', 'Bt', 'AtBt']:
                if trans_key in data_by_dtype[dtype][size]:
                    ratio = data_by_dtype[dtype][size][trans_key] / baseline
                    ratios[dtype][size][trans_key] = ratio
    
    return ratios


def create_bar_plot(ax, ratios_by_size, dtype, title, y_max=8):
    """
    Create grouped bar chart showing transpose overhead ratios.
    
    Args:
        ax: Matplotlib axis object
        ratios_by_size: Dict mapping size -> transpose_config -> ratio
        dtype: Data type being plotted
        title: Plot title
        y_max: Maximum y-axis value (bars exceeding this get labeled)
    """
    sizes = sorted(ratios_by_size.keys())
    trans_configs = ['At', 'Bt', 'AtBt']
    trans_labels = ['A.T @ B', 'A @ B.T', 'A.T @ B.T']
    
    # Colors from magma colormap
    magma = plt.colormaps['magma']
    colors = [magma(0.3), magma(0.55), magma(0.8)]
    
    # Bar positioning
    x = np.arange(len(sizes))
    width = 0.25
    
    # Plot bars for each transpose configuration
    for i, (trans_key, label, color) in enumerate(zip(trans_configs, trans_labels, colors)):
        ratios = []
        for size in sizes:
            if trans_key in ratios_by_size[size]:
                ratios.append(ratios_by_size[size][trans_key])
            else:
                ratios.append(np.nan)
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, ratios, width, label=label, color=color, alpha=0.9)
        
        # Add labels for bars that exceed y_max
        for j, (bar, ratio) in enumerate(zip(bars, ratios)):
            if not np.isnan(ratio) and ratio > y_max:
                ax.text(bar.get_x() + bar.get_width() / 2, y_max * 0.95,
                       f'{ratio:.1f}',
                       ha='center', va='top', fontsize=9, fontweight='bold',
                       color='white', bbox=dict(boxstyle='round,pad=0.3', 
                                               facecolor=color, alpha=0.8))
    
    # Baseline line at y=1
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.6, 
               label='Baseline (no transpose)')
    
    # Labels and title
    ax.set_xlabel('Matrix Dimension (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Duration (ratio to baseline)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set y-limit
    ax.set_ylim(0, y_max)
    
    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, edgecolor='gray')
    
    return ax


def main():
    """Main execution function."""
    # Load database
    db_path = Path(__file__).parent / "matmul_database.json"
    print(f"Loading database from {db_path}")
    database = load_database(db_path)
    
    # Extract transpose data
    print("Extracting transpose configuration data...")
    data = extract_transpose_data(database)
    print(f"Found data for dtypes: {list(data.keys())}")
    
    # Compute ratios
    print("Computing ratios relative to no-transpose baseline...")
    ratios = compute_ratios(data)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot each dtype
    dtype_order = ['fp32', 'fp16', 'bf16']
    for ax, dtype in zip(axes, dtype_order):
        if dtype in ratios and len(ratios[dtype]) > 0:
            print(f"Creating plot for {dtype.upper()}...")
            create_bar_plot(
                ax, ratios[dtype], dtype,
                f'Transpose Overhead - {dtype.upper()}',
                y_max=8
            )
        else:
            print(f"No data for {dtype}")
    
    # Overall title
    fig.suptitle(
        'Matrix Transpose Overhead (relative to A @ B baseline)',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )
    
    # Add note
    fig.text(
        0.5, 0.01,
        'Ratio = 1.0 means same performance as baseline (batch_size=1, aten::mm only). Bars exceeding y=8 are labeled with their value.',
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent / "transpose_overhead_plot.png"
    print(f"\nSaving plot to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("Plot saved successfully!")
    
    # Show summary statistics
    print("\n" + "="*80)
    print("TRANSPOSE OVERHEAD SUMMARY")
    print("="*80)
    
    for dtype in dtype_order:
        if dtype not in ratios:
            continue
        print(f"\n{dtype.upper()}:")
        for size in sorted(ratios[dtype].keys()):
            print(f"  Size {size}×{size}:")
            for trans_key in ['At', 'Bt', 'AtBt']:
                if trans_key in ratios[dtype][size]:
                    ratio = ratios[dtype][size][trans_key]
                    overhead = (ratio - 1) * 100
                    label = {'At': 'A.T @ B', 'Bt': 'A @ B.T', 'AtBt': 'A.T @ B.T'}[trans_key]
                    print(f"    {label:15s}: {ratio:.3f}× ({overhead:+.1f}% vs baseline)")
    
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()

