"""
Networking utilities for validation experiments.

This module provides reusable components for:
- Distributed setup and teardown (NCCL, process groups)
- Profiling infrastructure (PyTorch profiler + ET observer)
- Metadata capture (GPU properties, topology, system state)
- Timestamp synchronization (barrier-based alignment, drift measurement)
- Launch helpers (environment setup, pre-flight checks)
"""

from .distributed_utils import (
    init_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    barrier_all,
    print_once,
)

from .profiling_utils import (
    ProfilerContext,
    warmup_phase,
    configure_profiler,
)

from .metadata_utils import (
    capture_gpu_metadata,
    capture_topology,
    capture_nccl_info,
    save_metadata,
    capture_full_metadata,
)

from .sync_utils import (
    SyncPoint,
    measure_clock_drift,
    create_sync_barrier,
)

from .launch_utils import (
    validate_environment,
    lock_gpu_clocks,
    thermal_warmup,
)

from .monitoring_utils import (
    TemperatureLogger,
    quick_warmup,
    check_temperature,
)

__all__ = [
    # Distributed
    'init_distributed',
    'cleanup_distributed',
    'get_rank',
    'get_world_size',
    'barrier_all',
    'print_once',
    # Profiling
    'ProfilerContext',
    'warmup_phase',
    'configure_profiler',
    # Metadata
    'capture_gpu_metadata',
    'capture_topology',
    'capture_nccl_info',
    'save_metadata',
    'capture_full_metadata',
    # Synchronization
    'SyncPoint',
    'measure_clock_drift',
    'create_sync_barrier',
    # Launch
    'validate_environment',
    'lock_gpu_clocks',
    'thermal_warmup',
    # Monitoring
    'TemperatureLogger',
    'quick_warmup',
    'check_temperature',
]

