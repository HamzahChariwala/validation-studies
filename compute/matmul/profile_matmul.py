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
    
    # Check for new precision types (may not be supported in all PyTorch versions)
    if dtype_str == 'int4' and hasattr(torch, 'int4'):
        return torch.int4
    elif dtype_str == 'fp8_e4m3' and hasattr(torch, 'float8_e4m3fn'):
        return torch.float8_e4m3fn
    elif dtype_str == 'fp8_e5m2' and hasattr(torch, 'float8_e5m2'):
        return torch.float8_e5m2
    elif dtype_str in dtype_map:
        return dtype_map[dtype_str]
    else:
        raise ValueError(f"Unsupported or unavailable dtype: {dtype_str}")


def is_dtype_supported(dtype_str: str, device: str) -> bool:
    """Check if a dtype is supported on the given device for matmul operations."""
    # Check bfloat16 support
    if dtype_str == 'bf16':
        if device == 'cuda':
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        elif device == 'cpu':
            return hasattr(torch, 'bfloat16')
        elif device == 'mps':
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    # Check int8 support - torch.matmul does NOT support int8 natively
    # Int8 matmul requires special kernels (e.g., torch.int_repr + quantized ops)
    # For basic profiling, we skip int8 matmul
    elif dtype_str == 'int8':
        # Standard torch.matmul doesn't support int8, would need quantized API
        return False
    
    # Check int4 support (typically requires special hardware/libraries)
    elif dtype_str == 'int4':
        return hasattr(torch, 'int4') and device == 'cuda'
    
    # Check FP8 support (requires newer GPUs like H100)
    elif dtype_str in ['fp8_e4m3', 'fp8_e5m2']:
        if device != 'cuda':
            return False
        if not torch.cuda.is_available():
            return False
        # Check if FP8 dtypes exist in PyTorch
        if dtype_str == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
            return False
        if dtype_str == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
            return False
        # FP8 requires Hopper+ GPUs (compute capability 9.0+)
        # T4 is 7.5, A100 is 8.0, H100 is 9.0
        try:
            capability = torch.cuda.get_device_capability()
            if capability[0] < 9:
                return False
            # Additional check: try to create a small FP8 tensor
            # Some systems may report wrong capability
            try:
                dtype = torch.float8_e4m3fn if dtype_str == 'fp8_e4m3' else torch.float8_e5m2
                _ = torch.tensor([1.0], dtype=dtype, device='cuda')
                return True
            except:
                return False
        except:
            return False
    
    # Standard floating point types (fp32, fp16) are generally supported everywhere
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
    
    For floating point types: Uses torch.rand() + offset to ensure non-zero values.
    For integer types: Uses torch.randint() with range [1, 128] for non-zero values.
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
    # Check if dtype is an integer type
    is_integer = dtype in [torch.int8, torch.int16, torch.int32, torch.int64]
    if hasattr(torch, 'int4') and dtype == torch.int4:
        is_integer = True
    
    # Check if dtype is an FP8 type (these need special handling)
    is_fp8 = False
    if hasattr(torch, 'float8_e4m3fn') and dtype == torch.float8_e4m3fn:
        is_fp8 = True
    if hasattr(torch, 'float8_e5m2') and dtype == torch.float8_e5m2:
        is_fp8 = True
    
    if is_integer:
        # For integer types, use randint with non-zero range
        # Range [1, 128] provides good coverage while avoiding overflow in accumulation
        if batch_size == 1:
            A = torch.randint(1, 128, (m, k), dtype=dtype, device=device)
            B = torch.randint(1, 128, (k, n), dtype=dtype, device=device)
        else:
            A = torch.randint(1, 128, (batch_size, m, k), dtype=dtype, device=device)
            B = torch.randint(1, 128, (batch_size, k, n), dtype=dtype, device=device)
    elif is_fp8:
        # For FP8 types, create as float32 first, then convert
        # FP8 types don't support rand() directly
        if batch_size == 1:
            A = (torch.rand(m, k, dtype=torch.float32, device=device) + 0.1).to(dtype)
            B = (torch.rand(k, n, dtype=torch.float32, device=device) + 0.1).to(dtype)
        else:
            A = (torch.rand(batch_size, m, k, dtype=torch.float32, device=device) + 0.1).to(dtype)
            B = (torch.rand(batch_size, k, n, dtype=torch.float32, device=device) + 0.1).to(dtype)
    else:
        # For standard floating point types (fp32, fp16, bf16), use rand + offset
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


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that a configuration can actually create tensors and run matmul.
    Returns True if valid, False if not.
    """
    try:
        dtype = dtype_str_to_torch(config['dtype'])
        device = config['device']
        
        # Try creating small test tensors
        A_test, B_test = create_tensors(
            m=2, n=2, k=2,
            dtype=dtype,
            batch_size=1,
            device=device,
        )
        
        # Try running matmul
        with torch.no_grad():
            _ = run_matmul(A_test, B_test, config['transpose_a'], config['transpose_b'])
        
        return True
    except Exception as e:
        # Config is not actually supported despite passing initial checks
        return False


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
    
    # Filter configs by device availability and dtype support
    device = configs[0]['device'] if configs else 'cpu'
    
    # Filter out unsupported dtypes (first pass - static checks)
    original_count = len(configs)
    configs = [c for c in configs if is_dtype_supported(c['dtype'], device)]
    filtered_static = original_count - len(configs)
    
    if filtered_static > 0:
        print(f"Note: Filtered out {filtered_static} configurations (static dtype checks)")
    
    # Validate actual tensor creation (second pass - runtime checks)
    print("Validating remaining configurations with actual tensor creation...")
    validated_configs = []
    failed_dtypes = set()
    
    for config in configs:
        if validate_config(config):
            validated_configs.append(config)
        else:
            failed_dtypes.add(config['dtype'])
    
    filtered_runtime = len(configs) - len(validated_configs)
    configs = validated_configs
    
    if filtered_runtime > 0:
        print(f"Note: Filtered out {filtered_runtime} additional configurations (runtime validation failed)")
        print(f"      Failed dtypes: {sorted(failed_dtypes)}")
    
    print(f"\nMatmul Profiling: {len(configs)} valid configurations, {REPEAT_COUNT} repeats")
    print(f"Output: {output_dir}\n")
    
    if len(configs) == 0:
        print("ERROR: No valid configurations remaining after filtering!")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the actual configs that will be run for database building
    import json as json_lib
    configs_file = os.path.join(output_dir, 'configs.json')
    with open(configs_file, 'w') as f:
        json_lib.dump(configs, f, indent=2)
    print(f"Saved configuration list to: {configs_file}")
    
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

