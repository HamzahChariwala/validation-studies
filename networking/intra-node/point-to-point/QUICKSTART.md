# Quick Start Guide

## Prerequisites

```bash
# Check you have CUDA and multiple GPUs
nvidia-smi

# Verify PyTorch can see GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Run Experiment

```bash
cd networking/intra-node/point-to-point

# Basic run (auto-detects all GPUs)
bash launch_p2p.sh

# Specify number of GPUs
bash launch_p2p.sh --gpus 4

# Quick test without warmup (much faster, ~2-3 minutes)
bash launch_p2p.sh --gpus 2 --no-warmup

# Custom random seed for different config ordering
bash launch_p2p.sh --seed 12345

# Custom number of repetitions (fewer for faster testing)
bash launch_p2p.sh --repeat 5

# With clock locking for consistency (requires sudo)
bash launch_p2p.sh --lock-clocks

# Combined options
bash launch_p2p.sh --gpus 4 --seed 999 --repeat 20 --no-warmup
```

## CLI Arguments

All available options:
- `--gpus N` - Number of GPUs to use (default: all available)
- `--seed N` - Random seed for config ordering (default: 42)
- `--repeat N` - Number of repetitions per config (default: 10)
- `--no-warmup` - Skip thermal warmup and warmup iterations (faster testing)
- `--lock-clocks` - Lock GPU clocks to max frequency (requires sudo)
- `--help` - Show help message

## Check Configuration

```bash
# Preview what will be tested
python p2p_config.py
```

## Output

Results saved to `./traces/run_YYYYMMDD_HHMMSS/`:
- `run_info.txt` - Run configuration summary
- `rank_N/` - Directory with individual GPU trace files (one per operation)
- `rank_N_CPU_trace.json` - CPU traces
- `execution_log.csv` - Execution order and timings
- `temperatures.csv` - Temperature/power/clock logs
- `metadata_rank_N.yml` - System metadata

Each run creates a timestamped subdirectory, so multiple runs never overwrite each other.

## Expected Runtime

Runtime scales with `--repeat` value (default: 10 repetitions per config).

For 4 GPUs (372 configurations):
- 372 configs × 10 repetitions = 3,720 tests
- **Full run:** ~20-30 minutes (including thermal warmup, NCCL warmup, drift measurements)
- **With --no-warmup:** ~5-10 minutes (skips thermal and NCCL warmup)
- **Quick test (--repeat 5 --no-warmup):** ~2-3 minutes

For 2 GPUs (62 configurations):
- 62 configs × 10 repetitions = 620 tests
- **Full run:** ~10-15 minutes
- **With --no-warmup:** ~2-3 minutes
- **Quick test (--repeat 5 --no-warmup):** ~1-2 minutes

## Troubleshooting

**CUDA not available:**
```bash
# Check CUDA installation
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

**Processes already running:**
```bash
# Kill GPU processes
nvidia-smi | grep python | awk '{print $5}' | xargs kill -9
```

**Permission denied for clock locking:**
```bash
# Run without clock locking, or
sudo bash launch_p2p.sh --lock-clocks
```

## Next Steps

See `README.md` for full documentation on experimental design, analysis, and interpretation.

