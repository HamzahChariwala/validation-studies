"""
Configuration file for compute roofline profiling experiments.
Defines benchmark configurations to span the memory-bound to compute-bound spectrum.
"""

import itertools
from typing import List, Dict, Any
import torch


# ============================================================================
# MATRIX SIZES - Selected to span low to high arithmetic intensity
# ============================================================================
# Mix of square and non-square matrices (common in production workloads)
# Arithmetic intensity I = 2*M*N*K / [(M*K + K*N + M*N) * bytes_per_element]
# Larger matrices and more square-like shapes have higher arithmetic intensity
#
# Strategy: Cover both square and rectangular shapes across size ranges

# Square matrices (baseline, highest arithmetic intensity for given size)
# Expanded with more intermediate sizes for better model fitting
SQUARE_SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (384, 384, 384),     # Added
    (512, 512, 512),
    (768, 768, 768),     # Added
    (1024, 1024, 1024),
    (1536, 1536, 1536),  # Added
    (2048, 2048, 2048),
    (3072, 3072, 3072),  # Added
    (4096, 4096, 4096),
]

# Common production non-square shapes (attention, FFN, embeddings)
# Format: (M, N, K) for C[M,N] = A[M,K] @ B[K,N]
# Expanded to cover more arithmetic intensity points
NON_SQUARE_SIZES = [
    # Very small / inference shapes (low arithmetic intensity)
    (64, 2048, 512),     # Tiny batch
    (128, 4096, 1024),   # Small batch, hidden projection
    (256, 2048, 512),    # Small batch, narrow projection
    
    # Small-medium batch shapes
    (512, 4096, 4096),   # Medium batch, FFN intermediate
    (512, 2048, 1024),   # Medium batch, smaller hidden
    (768, 3072, 1024),   # Medium batch, intermediate size
    (1024, 2048, 4096),  # Attention-like shape
    (1024, 4096, 2048),  # Rectangular projection
    
    # Medium batch × various hidden sizes
    (2048, 1024, 512),   # Wide and shallow
    (2048, 4096, 1024),  # QKV projection
    (2048, 2048, 512),   # Medium square-ish, shallow K
    (2048, 8192, 2048),  # FFN up-projection
    
    # Large batch shapes
    (4096, 1024, 4096),  # FFN down-projection
    (4096, 4096, 2048),  # Large batch projection
    (4096, 2048, 1024),  # Large batch, narrow output
    (4096, 8192, 4096),  # Very large projection
    
    # Very large batch / sequence
    (8192, 2048, 2048),  # Very large batch/sequence
    (8192, 4096, 1024),  # Very large batch, narrow
    (8192, 1024, 512),   # Very large batch, very narrow
    
    # Extreme aspect ratios (additional coverage)
    (128, 8192, 4096),   # Tiny batch, huge hidden
    (256, 8192, 2048),   # Small batch, huge projection
]

# Combine for full test suite
MATRIX_SIZES = SQUARE_SIZES + NON_SQUARE_SIZES

# Precisions to profile (each has different peak FLOPS)
PRECISIONS = [
    'fp32',      # 32-bit floating point - baseline
    'fp16',      # 16-bit floating point - 2x faster than fp32
    'bf16',      # bfloat16 - similar to fp16, requires Ampere+
    # FP8 types commented out - only work on H100+
    # 'fp8_e4m3',  # 8-bit float E4M3 - requires Hopper (compute 9.0+)
    # 'fp8_e5m2',  # 8-bit float E5M2 - requires Hopper (compute 9.0+)
]

# Batch sizes to test (batching can affect performance characteristics)
# Test single (inference), small, medium, and large batch sizes
BATCH_SIZES = [1, 4, 8, 16]

# Auto-detect available device
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ============================================================================
# PROFILING SETTINGS
# ============================================================================
WARMUP_ITERATIONS = 2   # Warmup iterations per config
REPEAT_COUNT = 5        # Measurements per config
RANDOM_SEED = 42

# Default output directory
OUTPUT_DIR = './compute_roofline_results'


def get_bytes_per_element(dtype_str: str) -> int:
    """Get number of bytes per element for a given dtype string."""
    bytes_map = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
        'int4': 0.5,  # 4 bits = 0.5 bytes
        'fp8_e4m3': 1,
        'fp8_e5m2': 1,
    }
    return bytes_map.get(dtype_str, 4)


def compute_matmul_metrics(m: int, n: int, k: int, dtype_str: str, batch_size: int = 1) -> Dict[str, float]:
    """
    Compute theoretical FLOPS, bytes transferred, and arithmetic intensity for a matmul.
    
    For C = A @ B where A[M,K], B[K,N]:
    - FLOPS = 2*M*N*K (each output element requires K multiply-adds)
    - Bytes = (M*K + K*N + M*N) * bytes_per_element (read A, read B, write C)
    - Arithmetic Intensity = FLOPS / Bytes
    
    Args:
        m: Rows of A
        n: Columns of B  
        k: Columns of A / Rows of B
        dtype_str: Data type string ('fp32', 'fp16', etc.)
        batch_size: Batch size (multiplies FLOPS and bytes)
    
    Returns:
        Dictionary with 'flops', 'bytes', 'arithmetic_intensity'
    """
    bytes_per_elem = get_bytes_per_element(dtype_str)
    
    # FLOPS for single matmul
    flops_single = 2 * m * n * k
    
    # Bytes transferred for single matmul (simplified model)
    # Read A: m*k elements, Read B: k*n elements, Write C: m*n elements
    bytes_single = (m * k + k * n + m * n) * bytes_per_elem
    
    # Account for batching
    flops = flops_single * batch_size
    bytes_transferred = bytes_single * batch_size
    
    # Arithmetic intensity (ops per byte)
    arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0.0
    
    return {
        'flops': float(flops),
        'bytes': float(bytes_transferred),
        'arithmetic_intensity': float(arithmetic_intensity)
    }


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
    elif dtype_str == 'int8':
        return False
    
    # Check int4 support
    elif dtype_str == 'int4':
        return hasattr(torch, 'int4') and device == 'cuda'
    
    # Check FP8 support (requires Hopper GPUs, compute capability 9.0+)
    elif dtype_str in ['fp8_e4m3', 'fp8_e5m2']:
        if device != 'cuda' or not torch.cuda.is_available():
            return False
        
        # Check if FP8 dtypes exist in PyTorch
        if dtype_str == 'fp8_e4m3' and not hasattr(torch, 'float8_e4m3fn'):
            return False
        if dtype_str == 'fp8_e5m2' and not hasattr(torch, 'float8_e5m2'):
            return False
        
        # FP8 requires Hopper+ GPUs (compute capability 9.0+)
        try:
            capability = torch.cuda.get_device_capability()
            if capability[0] < 9:
                return False
            
            # Try to create a small FP8 tensor
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


def generate_benchmark_configs(verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Generate all benchmark configurations for roofline profiling.
    
    Args:
        verbose: If True, print information about skipped dtypes
    
    Returns:
        List of config dicts with: config_id, m, n, k, dtype, batch_size, device,
        and precomputed metrics (flops, bytes, arithmetic_intensity)
    """
    configs = []
    config_id = 1
    skipped_dtypes = set()
    
    for (m, n, k), dtype, batch_size in itertools.product(
        MATRIX_SIZES, PRECISIONS, BATCH_SIZES
    ):
        # Check dtype support with graceful error handling
        try:
            if not is_dtype_supported(dtype, DEVICE):
                skipped_dtypes.add(dtype)
                continue
        except Exception as e:
            if verbose:
                print(f"  Warning: Error checking dtype {dtype}: {e}")
            skipped_dtypes.add(dtype)
            continue
        
        # Compute theoretical metrics
        try:
            metrics = compute_matmul_metrics(m, n, k, dtype, batch_size)
        except Exception as e:
            if verbose:
                print(f"  Warning: Error computing metrics for {dtype}: {e}")
            continue
        
        config = {
            'config_id': config_id,
            'm': m,
            'n': n,
            'k': k,
            'dtype': dtype,
            'batch_size': batch_size,
            'device': DEVICE,
            # Precomputed theoretical values
            'flops': metrics['flops'],
            'bytes': metrics['bytes'],
            'arithmetic_intensity': metrics['arithmetic_intensity'],
        }
        configs.append(config)
        config_id += 1
    
    if verbose and skipped_dtypes:
        print(f"\n  Note: Skipped unsupported dtypes on {DEVICE}: {sorted(skipped_dtypes)}")
    
    return configs


def get_config_summary(config: Dict[str, Any]) -> str:
    """Generate a human-readable summary string for a configuration."""
    batch_str = f"B{config['batch_size']}x" if config['batch_size'] > 1 else ""
    
    return (f"Config {config['config_id']:04d}: "
            f"{batch_str}[{config['m']}x{config['k']}] @ "
            f"[{config['k']}x{config['n']}] "
            f"({config['dtype']}) "
            f"I={config['arithmetic_intensity']:.1f} ops/byte")


if __name__ == "__main__":
    # Test the configuration generator
    configs = generate_benchmark_configs()
    
    print("=" * 80)
    print(f"COMPUTE ROOFLINE PROFILING CONFIGURATION")
    print("=" * 80)
    print(f"Total configurations: {len(configs)}")
    print(f"Device: {DEVICE}")
    
    print(f"\nMatrix sizes: {len(MATRIX_SIZES)}")
    print(f"Precisions: {PRECISIONS}")
    print(f"Batch sizes: {BATCH_SIZES}")
    
    # Group by precision to show arithmetic intensity range per precision
    print(f"\nArithmetic intensity ranges by precision:")
    by_precision = {}
    for config in configs:
        dtype = config['dtype']
        if dtype not in by_precision:
            by_precision[dtype] = []
        by_precision[dtype].append(config['arithmetic_intensity'])
    
    for dtype in sorted(by_precision.keys()):
        intensities = by_precision[dtype]
        print(f"  {dtype:8s}: {min(intensities):6.1f} to {max(intensities):8.1f} ops/byte "
              f"({len(intensities)} configs)")
    
    print(f"\nFirst 10 example configurations:\n")
    for config in configs[:10]:
        print(get_config_summary(config))
    
    if len(configs) > 10:
        print(f"\n... and {len(configs) - 10} more configurations")
    
    print("=" * 80)

