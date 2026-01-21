"""
Profiling utilities for PyTorch profiler and Chakra ExecutionTrace.

Provides reusable profiling context managers and configuration helpers.
Consistent with profiling approach used in compute/matmul experiments.
"""

import os
import torch
from torch.profiler import ExecutionTraceObserver, ProfilerActivity
from typing import Optional, List, Callable
from contextlib import contextmanager


class ProfilerContext:
    """
    Context manager for PyTorch profiler with ExecutionTrace observer.
    
    Combines GPU profiler and CPU trace collection in a single convenient interface.
    Handles proper setup/teardown and file naming.
    
    Example:
        with ProfilerContext(rank=0, output_dir='./traces') as prof:
            for step in range(num_steps):
                run_experiment()
                prof.step()
    """
    
    def __init__(
        self,
        rank: int,
        output_dir: str,
        trace_name: str = 'trace',
        activities: Optional[List[ProfilerActivity]] = None,
        schedule: Optional[Callable] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
    ):
        """
        Initialize profiler context.
        
        Args:
            rank: Process rank (for multi-process experiments)
            output_dir: Directory to save traces
            trace_name: Base name for trace files
            activities: List of profiler activities (default: CPU + CUDA)
            schedule: Profiler schedule (default: no schedule)
            record_shapes: Record tensor shapes
            profile_memory: Profile memory allocations
            with_stack: Record Python stack traces
        """
        self.rank = rank
        self.output_dir = output_dir
        self.trace_name = trace_name
        
        # Default activities: CPU + CUDA if available
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        self.activities = activities
        
        self.schedule = schedule
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        
        self.profiler = None
        self.et_observer = None
    
    def __enter__(self):
        """Start profiling."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup ExecutionTrace observer for CPU trace
        cpu_trace_path = os.path.join(
            self.output_dir, 
            f"rank_{self.rank}_CPU_trace.json"
        )
        self.et_observer = ExecutionTraceObserver()
        self.et_observer.register_callback(cpu_trace_path)
        self.et_observer.start()
        
        # Setup GPU profiler
        gpu_trace_dir = os.path.join(self.output_dir, f"rank_{self.rank}")
        
        self.profiler = torch.profiler.profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(gpu_trace_dir),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )
        
        self.profiler.__enter__()
        return self.profiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and clean up."""
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
        
        if self.et_observer is not None:
            self.et_observer.stop()
            self.et_observer.unregister_callback()
        
        return False


def configure_profiler(
    wait_steps: int = 0,
    warmup_steps: int = 0,
    active_steps: int = 1,
    repeat_count: int = 1
) -> Callable:
    """
    Create a standard profiler schedule.
    
    Args:
        wait_steps: Steps to wait before profiling
        warmup_steps: Warmup steps (not profiled)
        active_steps: Steps to actively profile
        repeat_count: Number of times to repeat the cycle
    
    Returns:
        Schedule function for torch.profiler.profile()
    """
    return torch.profiler.schedule(
        wait=wait_steps,
        warmup=warmup_steps,
        active=active_steps,
        repeat=repeat_count
    )


def warmup_phase(
    experiment_fn: Callable,
    configs: List,
    warmup_iterations: int = 5,
    verbose: bool = True
):
    """
    Run warmup phase before profiling.
    
    Ensures:
    - CUDA kernels are compiled (JIT)
    - Communication libraries initialized (NCCL)
    - Memory buffers allocated
    - GPU caches warmed
    
    Args:
        experiment_fn: Function to run for each config
        configs: List of configurations to warmup
        warmup_iterations: Number of iterations per config
        verbose: Print progress
    """
    if verbose:
        print(f"Warmup phase: {warmup_iterations} iterations × {len(configs)} configs...")
    
    for i, config in enumerate(configs):
        if verbose and i % 10 == 0:
            print(f"  Warmup progress: {i}/{len(configs)} configs...")
        
        for _ in range(warmup_iterations):
            experiment_fn(config)
        
        # Synchronize after each config
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    if verbose:
        print("Warmup complete.")


@contextmanager
def torch_deterministic(seed: int = 42):
    """
    Context manager for deterministic PyTorch operations.
    
    Sets random seeds for reproducibility.
    """
    # Save current state
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    
    # Set deterministic seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    try:
        yield
    finally:
        # Restore state
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

