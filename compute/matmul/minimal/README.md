# Compute Roofline Profiler - Minimal

Production-ready GPU compute benchmarking tool that extracts roofline model parameters (peak FLOPS and memory bandwidth) for use in performance simulators.

## Overview

This profiler measures matrix multiplication performance across a range of sizes, precisions, and batch sizes to fit continuous roofline models. Similar to the networking profiler, it provides the core functionality needed to characterize GPU compute behavior.

## Quick Start

### Multi-GPU Profiling (Recommended)

```bash
cd /home/azureuser/validation-studies
torchrun --nproc_per_node=4 compute/matmul/minimal/profile_minimal.py
```

### Single GPU

```bash
cd /home/azureuser/validation-studies
python3 compute/matmul/minimal/profile_minimal.py --no-lock-clocks
```

## Configuration

The profiler tests:
- **31 matrix sizes**: Mix of square (128³ to 4096³) and non-square matrices
- **3 precisions**: fp32, fp16, bf16 (fp8 if H100+)
- **4 batch sizes**: 1, 4, 8, 16
- **5 repetitions** per configuration

Total: **1,860 measurements** per run (~5-10 minutes on 4 GPUs)

### Customization

Edit `compute_config.py` in this directory to modify:
- `MATRIX_SIZES`: Matrix dimensions to test
- `PRECISIONS`: Data types to profile
- `BATCH_SIZES`: Batch sizes to test
- `REPEAT_COUNT`: Repetitions per configuration

## Output

Results are saved to `compute_roofline_results_YYYYMMDD_HHMMSS/`:

```
compute_roofline_results_20260209_171345/
├── RESULTS_SUMMARY.txt              # Human-readable summary
├── results.json                     # Complete structured results
├── roofline_parameters.csv          # Overall models per precision
├── roofline_parameters_detailed.csv # Models per (precision, batch_size)
├── measurements.csv                 # Raw kernel timing data
└── gpu_info.txt                     # GPU hardware info
```

## Roofline Models

The profiler fits **3 regression methods** for each configuration:
- **L2**: Ordinary least squares (most stable)
- **Huber**: Robust to outliers
- **IRLS**: Iteratively reweighted least squares (no hyperparameters)

Models are provided:
- **Per (precision, batch_size)**: 12 groups (3 precisions × 4 batches)
- **Per precision (overall)**: 3 groups (aggregated across batches)

## Options

```bash
--no-lock-clocks        Disable GPU clock locking (default: enabled, requires sudo)
--output-dir DIR        Custom output directory
--repeat-count N        Number of repetitions per config (default: 5)
```

## Requirements

- PyTorch with CUDA support
- numpy, scipy, scikit-learn
- sudo access (optional, for GPU clock locking)
- Multiple GPUs (optional, but recommended for speed)

## Architecture

```
compute/matmul/minimal/
├── profile_minimal.py    # Main profiler script
├── compute_config.py     # Configuration (matrix sizes, precisions, etc.)
└── README.md            # This file

compute/matmul/analysis/
├── fit_roofline.py          # Base roofline fitting (single method)
├── fit_roofline_multi.py    # Multi-method fitting (L2, Huber, IRLS)
└── build_database.py        # Database construction tools
```

## Workflow

1. **Profile**: Run `profile_minimal.py` to collect measurements
2. **Analyze**: Results include fitted roofline parameters
3. **Select**: Choose best method (L2/Huber/IRLS) based on R² and convergence
4. **Export**: Use parameters from CSV files in your simulator

## Example Results

```
OVERALL ROOFLINE MODELS (AGGREGATED BY PRECISION) - ALL 3 METHODS
--------------------------------------------------------------------------------

Precision    Method   Bandwidth        Peak FLOPS       R²         Status
--------------------------------------------------------------------------------
fp16         l2          923.45 GB/s     19.23 TFLOPS  0.998234  OK
fp16         huber       920.12 GB/s     19.31 TFLOPS  0.997845  OK
fp16         irls        921.78 GB/s     19.27 TFLOPS  0.998012  OK

fp32         l2          850.23 GB/s     19.45 TFLOPS  0.996543  OK
fp32         huber       847.91 GB/s     19.52 TFLOPS  0.995821  OK
fp32         irls        849.12 GB/s     19.48 TFLOPS  0.996234  OK
...
```

## Design Notes

This minimal profiler follows the same architecture as the networking profiler:
- Located in a `minimal/` subdirectory
- Self-contained with local configuration
- Produces consistent output format
- Supports multi-GPU distributed execution
- Provides multiple model options for robustness

## Related Tools

- **Networking Profiler**: `networking/intra-node/point-to-point/minimal/profile_minimal.py`
- **Analysis Tools**: `compute/matmul/analysis/`

## Contact

For issues or questions, see the main project documentation.

