"""
Distributed computing utilities for multi-GPU/multi-node networking tests.

Provides common setup/teardown for torch.distributed and NCCL operations.
Designed for reuse across different communication pattern tests.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def init_distributed(
    backend: str = 'nccl',
    init_method: str = 'env://',
    timeout_minutes: int = 30
) -> tuple[int, int]:
    """
    Initialize distributed process group.
    
    Designed to work with torchrun launcher, which sets required env vars:
    - RANK: Global rank of this process
    - WORLD_SIZE: Total number of processes
    - MASTER_ADDR: Address of rank 0
    - MASTER_PORT: Port for communication
    
    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method (default: 'env://' for torchrun)
        timeout_minutes: Timeout for operations
    
    Returns:
        Tuple of (rank, world_size)
    
    Raises:
        RuntimeError: If required environment variables not set
    """
    # Validate environment
    required_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    missing = [var for var in required_vars if var not in os.environ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {missing}\n"
            f"Are you using torchrun to launch?"
        )
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout * (timeout_minutes / 30)
    )
    
    # Set CUDA device for this rank
    if backend == 'nccl' and torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
    
    return rank, world_size


def cleanup_distributed():
    """
    Clean up distributed process group.
    
    Safe to call even if process group not initialized.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get rank of current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier_all():
    """
    Synchronize all processes.
    
    All processes wait until all have reached this point.
    """
    if dist.is_initialized():
        dist.barrier()


def get_backend() -> Optional[str]:
    """Get current distributed backend."""
    if dist.is_initialized():
        return dist.get_backend()
    return None


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def print_once(*args, **kwargs):
    """Print only from rank 0 (avoids duplicate prints)."""
    if is_main_process():
        print(*args, **kwargs)

