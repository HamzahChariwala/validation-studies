"""
Profile matrix multiplications using PyTorch Profiler for Chakra trace collection.
Systematically tests different configurations (size, precision, layout, batching).
"""

import torch
import torch.profiler
from torch.profiler import ExecutionTraceObserver
import os
import argparse
from typing import Dict, Any, Tuple

from matmul_config import (
    generate_experiment_configs,
    get_config_summary,
    WARMUP_ITERATIONS,
    REPEAT_COUNT,
    WAIT_STEPS,
    ACTIVE_STEPS,
    RANDOM_SEED,
)


def dtype_str_to_torch(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'int8': torch.int8,
    }
    return dtype_map[dtype_str]


def is_dtype_supported(dtype_str: str, device: str) -> bool:
    """Check if a dtype is supported on the given device."""
    if dtype_str == 'bf16':
        if device == 'cuda':
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif device == 'cpu':
            return hasattr(torch, 'bfloat16')
        elif device == 'mps':
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return True


def create_tensors(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random NON-ZERO tensors for matmul.
    
    Uses torch.rand() + offset to ensure all values are non-zero.
    This prevents hardware sparsity optimizations from affecting results.
    Seeding is handled globally via torch.manual_seed() for reproducibility.
    
    Args:
        m: Number of rows in A
        n: Number of columns in B
        k: Inner dimension (columns of A, rows of B)
        dtype: PyTorch data type
        batch_size: Batch size (1 for non-batched 2D tensors)
        device: Device string ('cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of (A, B) tensors ready for matmul
    """
    if batch_size == 1:
        A = torch.rand(m, k, dtype=dtype, device=device) + 0.1
        B = torch.rand(k, n, dtype=dtype, device=device) + 0.1
    else:
        A = torch.rand(batch_size, m, k, dtype=dtype, device=device) + 0.1
        B = torch.rand(batch_size, k, n, dtype=dtype, device=device) + 0.1
    
    return A, B


def run_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False
) -> torch.Tensor:
    """Execute matmul with optional transposes."""
    if transpose_a:
        A = A.transpose(-2, -1)
    if transpose_b:
        B = B.transpose(-2, -1)
    
    return torch.matmul(A, B)


def run_single_config(config: Dict[str, Any]) -> None:
    """Execute matmul for a single configuration (called within profiling context)."""
    dtype = dtype_str_to_torch(config['dtype'])
    device = config['device']
    
    A, B = create_tensors(
        m=config['m'],
        n=config['n'],
        k=config['k'],
        dtype=dtype,
        batch_size=config['batch_size'],
        device=device,
    )
    
    with torch.no_grad():
        _ = run_matmul(A, B, config['transpose_a'], config['transpose_b'])


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Profile matrix multiplications with frontier LLM dimensions'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='traces_matmul',
        help='Name for the output directory and trace files (default: traces_matmul)'
    )
    args = parser.parse_args()
    
    # Set output directory based on CLI argument
    output_dir = f'./{args.output_name}'
    
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
    
    configs = generate_experiment_configs()
    print(f"Matmul Profiling: {len(configs)} configurations, {REPEAT_COUNT} repeats")
    print(f"Output: {output_dir}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter configs by device availability
    device = configs[0]['device'] if configs else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA not available but configs require it")
        return
    if device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("ERROR: MPS not available but configs require it")
        return
    
    # Warmup all configurations
    print(f"Running warmup iterations ({WARMUP_ITERATIONS} iterations × {len(configs)} configs = {WARMUP_ITERATIONS * len(configs)} total)...")
    for i, config in enumerate(configs):
        if i % 100 == 0:
            print(f"  Warmup progress: {i}/{len(configs)} configs...")
        for _ in range(WARMUP_ITERATIONS):
            run_single_config(config)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print("Warmup complete.")
    
    print(f"Starting profiling of {len(configs)} configs x {REPEAT_COUNT} repeats...")
    print(f"Total profiling steps: {(WAIT_STEPS + ACTIVE_STEPS) * REPEAT_COUNT}")
    
    # Setup Chakra ET observer with custom name
    chakra_output_path = os.path.join(output_dir, f"{args.output_name}_CPU_trace")
    et_observer = ExecutionTraceObserver()
    et_observer.register_callback(chakra_output_path + ".json")
    et_observer.start()
    
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    # Profile all configurations in a single session
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=WAIT_STEPS,
            warmup=0,
            active=ACTIVE_STEPS,
            repeat=REPEAT_COUNT
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Total steps = (wait + active) * repeat
        total_steps = (WAIT_STEPS + ACTIVE_STEPS) * REPEAT_COUNT
        for step in range(total_steps):
            print(f"  Profiling step {step+1}/{total_steps}...")
            # Run all configs in this step
            for config in configs:
                run_single_config(config)
            prof.step()
            if device == 'cuda':
                torch.cuda.synchronize()
    
    et_observer.stop()
    et_observer.unregister_callback()
    
    print(f"\nComplete! Traces saved to {output_dir}/")


if __name__ == "__main__":
    main()

