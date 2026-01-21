"""
Timestamp synchronization utilities for multi-GPU experiments.

Handles clock alignment and drift measurement across GPUs with independent clocks.
"""

import time
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SyncPoint:
    """
    A synchronization point capturing both CPU and GPU timestamps.
    
    Used to align GPU clocks across multiple processes by correlating
    GPU events with CPU timestamps at a common barrier.
    """
    rank: int
    cpu_timestamp_ns: int  # CPU time in nanoseconds (from time.perf_counter_ns)
    gpu_event: torch.cuda.Event  # GPU event for elapsed time measurement
    description: str = "sync"
    
    def elapsed_ms_from(self, other: 'SyncPoint') -> float:
        """
        Calculate elapsed time between two sync points (GPU time).
        
        Args:
            other: Earlier sync point
        
        Returns:
            Elapsed time in milliseconds (GPU time)
        """
        if not self.gpu_event.query():
            self.gpu_event.synchronize()
        if not other.gpu_event.query():
            other.gpu_event.synchronize()
        
        return other.gpu_event.elapsed_time(self.gpu_event)
    
    def cpu_elapsed_ms_from(self, other: 'SyncPoint') -> float:
        """
        Calculate elapsed time between two sync points (CPU time).
        
        Args:
            other: Earlier sync point
        
        Returns:
            Elapsed time in milliseconds (CPU time)
        """
        return (self.cpu_timestamp_ns - other.cpu_timestamp_ns) / 1e6


def create_sync_barrier(rank: int, description: str = "sync") -> SyncPoint:
    """
    Create a synchronization point at a distributed barrier.
    
    All processes record CPU and GPU timestamps, then synchronize via barrier.
    This provides a common reference point for timestamp alignment.
    
    Args:
        rank: Process rank
        description: Description of this sync point
    
    Returns:
        SyncPoint with CPU and GPU timestamps
    """
    # Record CPU timestamp
    cpu_time = time.perf_counter_ns()
    
    # Record GPU event
    gpu_event = torch.cuda.Event(enable_timing=True)
    gpu_event.record()
    
    # Barrier: all ranks synchronize here
    if dist.is_initialized():
        dist.barrier()
    
    # Ensure GPU event is recorded
    gpu_event.synchronize()
    
    return SyncPoint(
        rank=rank,
        cpu_timestamp_ns=cpu_time,
        gpu_event=gpu_event,
        description=description
    )


def measure_clock_drift(
    rank: int,
    duration_seconds: float = 30.0,
    description: str = "drift_measurement"
) -> Dict[str, float]:
    """
    Measure clock drift between GPU and CPU.
    
    Records start and end timestamps, waits for duration, then computes
    the drift ratio: (GPU elapsed) / (CPU elapsed).
    
    If drift ratio ≈ 1.0, clocks run at same rate.
    If drift ratio ≠ 1.0, GPU clock runs faster/slower than CPU.
    
    Args:
        rank: Process rank
        duration_seconds: How long to measure (default: 30s)
                         30-60s is sufficient for main experiments
                         Use 300s+ only for precision characterization
        description: Description for this measurement
    
    Returns:
        Dictionary with drift metrics
    """
    # Start sync point
    start_sync = create_sync_barrier(rank, f"{description}_start")
    
    # Wait for duration
    time.sleep(duration_seconds)
    
    # End sync point
    end_sync = create_sync_barrier(rank, f"{description}_end")
    
    # Calculate elapsed times
    gpu_elapsed_ms = end_sync.elapsed_ms_from(start_sync)
    cpu_elapsed_ms = end_sync.cpu_elapsed_ms_from(start_sync)
    
    # Compute drift ratio
    drift_ratio = gpu_elapsed_ms / cpu_elapsed_ms if cpu_elapsed_ms > 0 else 1.0
    drift_percent = (drift_ratio - 1.0) * 100
    
    return {
        'rank': rank,
        'description': description,
        'duration_requested_s': duration_seconds,
        'gpu_elapsed_ms': gpu_elapsed_ms,
        'cpu_elapsed_ms': cpu_elapsed_ms,
        'drift_ratio': drift_ratio,
        'drift_percent': drift_percent,
    }


def characterize_drift_multi_window(
    rank: int,
    num_windows: int = 10,
    window_duration_s: float = 30.0
) -> List[Dict[str, float]]:
    """
    Measure drift over multiple time windows.
    
    Helps identify if drift is stable or changing over time.
    
    Args:
        rank: Process rank
        num_windows: Number of measurement windows
        window_duration_s: Duration of each window
    
    Returns:
        List of drift measurements (one per window)
    """
    measurements = []
    
    for i in range(num_windows):
        drift_data = measure_clock_drift(
            rank=rank,
            duration_seconds=window_duration_s,
            description=f"drift_window_{i}"
        )
        measurements.append(drift_data)
        
        # Print progress from rank 0
        if rank == 0:
            print(f"  Drift window {i+1}/{num_windows}: "
                  f"ratio={drift_data['drift_ratio']:.10f}, "
                  f"drift={drift_data['drift_percent']:.6f}%")
    
    return measurements


def compute_alignment_offset(
    sync_point: SyncPoint,
    reference_cpu_time_ns: int
) -> float:
    """
    Compute offset to align GPU timestamps to a reference CPU time.
    
    This offset can be added to GPU trace timestamps to align them
    with a common CPU time reference across all ranks.
    
    Args:
        sync_point: SyncPoint from this rank
        reference_cpu_time_ns: Reference CPU timestamp (e.g., from rank 0)
    
    Returns:
        Offset in milliseconds to add to GPU timestamps
    """
    # Offset = (reference CPU time) - (local CPU time)
    # When added to GPU trace times (which are relative to local CPU),
    # this aligns them to the reference
    offset_ns = reference_cpu_time_ns - sync_point.cpu_timestamp_ns
    offset_ms = offset_ns / 1e6
    return offset_ms


def gather_sync_points(
    local_sync: SyncPoint,
    rank: int,
    world_size: int
) -> Optional[List[SyncPoint]]:
    """
    Gather synchronization points from all ranks to rank 0.
    
    Note: This is a simplified version. In practice, you'd serialize
    the CPU timestamps and gather them, then reconstruct SyncPoints.
    
    Args:
        local_sync: This rank's sync point
        rank: Process rank
        world_size: Total number of ranks
    
    Returns:
        List of all sync points (only on rank 0, None elsewhere)
    """
    if not dist.is_initialized():
        return [local_sync]
    
    # Gather CPU timestamps
    cpu_times = [torch.tensor(0, dtype=torch.int64) for _ in range(world_size)]
    local_time = torch.tensor(local_sync.cpu_timestamp_ns, dtype=torch.int64)
    
    if rank == 0:
        dist.gather(local_time, cpu_times, dst=0)
        # Convert to list of timestamps
        return [t.item() for t in cpu_times]
    else:
        dist.gather(local_time, dst=0)
        return None

