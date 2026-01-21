# Point-to-Point Analysis Tools

Scripts for analyzing point-to-point communication profiling data.

## Quick Start

### Analyze Execution Log

```bash
# From the point-to-point directory
cd networking/intra-node/point-to-point

# Analyze a specific run
python3 analysis/analyze_execution_log.py traces/run_20260121_164227

# Specify custom execution log name
python3 analysis/analyze_execution_log.py traces/run_20260121_164227 --execution-log execution_log.csv

# Change outlier threshold (default: 10000 μs)
python3 analysis/analyze_execution_log.py traces/run_20260121_164227 --outlier-threshold 5000
```

## Generated Plots

The script creates a `plots/` subdirectory within the run directory with:

1. **`duration_vs_size_log.png`** - Duration vs message size (log₂ x-axis)
   - Separate line for each GPU pair (src → dst)
   - Shows overall communication pattern across message sizes
   - Title shows count of outliers removed (if filtering applied)

2. **`duration_vs_size_linear_regression.png`** - Linear regression analysis
   - Duration vs message size (linear x-axis)
   - Fitted regression lines for each GPU pair
   - Reports R², slope (m), and intercept (c) for each pair
   - Title shows count of outliers removed (if filtering applied)

3. **`latency_heatmap.png`** - Communication latency matrix
   - Y-intercept from regression as latency proxy (computed from filtered data)
   - Shows base communication overhead (independent of message size)
   - Lower values = better latency
   - Title shows count of outliers removed (if filtering applied)

4. **`bandwidth_heatmap.png`** - Communication bandwidth matrix
   - Derived from regression slope (1/slope, computed from filtered data)
   - Shows effective bandwidth in GB/s
   - Higher values = better bandwidth
   - Title shows count of outliers removed (if filtering applied)

5. **`individual_pairs/gpu_X_to_Y.png`** - Per-GPU-pair detailed plots
   - Left: Log-scale scatterplot with filtered data points
   - Right: Linear-scale with gray dashed regression line
   - **Points colored by log(residual)**: Uses logarithmic scale to emphasize small deviations
     - Darker colors (magma) indicate points closer to the fit
     - Brighter colors indicate larger deviations from the fit
   - Includes colorbars showing normalized log(residual) magnitude
   - Shows count of removed outliers per GPU pair
   - Reports maximum residual value
   - Useful for diagnosing poor R² values and identifying non-linear behavior

6. **`regression_summary.csv`** - Detailed regression statistics
   - All regression parameters for each GPU pair (computed from filtered data)
   - Latency and bandwidth estimates

## Command-Line Options

```bash
python3 analysis/analyze_execution_log.py RUN_DIR [OPTIONS]

Options:
  --execution-log FILE    Name of CSV file (default: execution_log.csv)
  --outlier-threshold N   Duration threshold in μs for outlier removal (default: 10000)
```

### Outlier Filtering

**All plots and analyses** automatically filter out measurements with duration > 10,000 μs (configurable via `--outlier-threshold`). This helps:
- Remove erroneous measurements
- Improve regression fit quality (R² values)
- Better visualize the true communication pattern
- Ensure consistency across all visualizations

The filtering is applied globally during data loading, so:
- Log and linear plots show only filtered data
- Regression analysis uses only filtered data
- Heatmaps reflect filtered regression results
- Individual plots show filtered data with removal counts

## Understanding the Metrics

### Latency (Y-intercept)
- **Units:** microseconds (μs)
- **What it means:** Base communication overhead
- **Interpretation:** Time to send a zero-size message (fixed overhead)

### Bandwidth (1/Slope)
- **Units:** GB/s
- **What it means:** Effective data transfer rate
- **Interpretation:** How fast large messages are transferred
- **Note:** This is an approximation based on linear fit

### R² Value
- **Range:** 0 to 1
- **What it means:** Quality of linear fit
- **Good values:** > 0.95 indicates linear relationship holds well
- **Low values:** < 0.90 suggests non-linear behavior (e.g., message size thresholds)

## Example Workflow

```bash
# Run profiling
bash launch_p2p.sh --gpus 4 --seed 42

# Analyze results (finds latest run automatically)
RUN_DIR=$(ls -dt traces/run_* | head -1)
python3 analysis/analyze_execution_log.py $RUN_DIR

# View plots
eog $RUN_DIR/plots/*.png  # On Linux with Eye of GNOME
# or
open $RUN_DIR/plots/*.png  # On macOS
```

## Requirements

Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install with:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

