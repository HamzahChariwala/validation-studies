# Networking Utilities

Reusable utilities for distributed GPU communication profiling experiments.

## Modules

### Core Distributed
- **`distributed_utils.py`** - NCCL setup, barriers, rank management
- **`profiling_utils.py`** - PyTorch profiler + ExecutionTrace integration
- **`sync_utils.py`** - Timestamp synchronization, clock drift measurement

### System & Hardware
- **`metadata_utils.py`** - GPU properties, topology, system state capture
- **`launch_utils.py`** - Environment validation, clock management, thermal warmup
- **`monitoring_utils.py`** - Background temperature/power/clock logging

## Quick Reference

```python
from networking.utils import (
    # Distributed
    init_distributed,
    cleanup_distributed,
    barrier_all,
    
    # Profiling
    ProfilerContext,
    warmup_phase,
    
    # Monitoring
    TemperatureLogger,
    thermal_warmup,
    
    # Synchronization
    measure_clock_drift,
    create_sync_barrier,
    
    # Metadata
    capture_full_metadata,
    save_metadata,
)
```

## Example Usage

```python
def main():
    # Initialize distributed
    rank, world_size = init_distributed()
    
    # Start temperature monitoring
    if rank == 0:
        temp_logger = TemperatureLogger(gpu_ids=list(range(world_size)))
        temp_logger.start()
    
    # Warm up GPUs
    thermal_warmup(device=f'cuda:{rank}', duration=60, target_temp_min=50)
    
    # Measure clock drift
    drift = measure_clock_drift(rank, duration_seconds=30)
    
    # Profile experiment
    with ProfilerContext(rank=rank, output_dir='./traces') as prof:
        for config in configs:
            run_experiment(config)
        prof.step()
    
    # Stop monitoring
    if rank == 0:
        temp_logger.stop()
        temp_logger.save('./traces/temperatures.csv')
    
    # Save metadata
    metadata = capture_full_metadata(rank)
    save_metadata(metadata, f'./traces/metadata_rank_{rank}.yml')
    
    cleanup_distributed()
```

## Documentation

See experiment READMEs for detailed usage:
- `networking/intra-node/point-to-point/README.md`
