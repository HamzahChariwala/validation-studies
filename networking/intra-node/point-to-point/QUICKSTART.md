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

# With clock locking for consistency (requires sudo)
bash launch_p2p.sh --lock-clocks

# Specify number of GPUs
bash launch_p2p.sh --gpus 4
```

## Check Configuration

```bash
# Preview what will be tested
python p2p_config.py
```

## Output

Results saved to `./traces/`:
- `rank_N.pt.trace.json` - GPU traces with config annotations
- `rank_N_CPU_trace.json` - CPU traces
- `execution_log.csv` - Execution order and timings
- `temperatures.csv` - Temperature/power/clock logs
- `metadata_rank_N.yml` - System metadata

## Expected Runtime

For 4 GPUs:
- 372 configurations × 10 repetitions = 3,720 tests
- ~20-30 minutes total (including warmup, drift measurements, etc.)

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

