"""
Configuration file for matmul profiling experiments.
Defines parameter space and grid search generator.
"""

import itertools
from typing import List, Dict, Any


# ============================================================================
# FRONTIER LLM DIMENSIONS - All Permutations
# ============================================================================
# Extract all unique dimension values from LLaMA 70B, LLaMA 405B, and GPT-4,
# then generate every permutation (M, N, K) from these values.
#
# Source dimensions (before filtering):
# - LLaMA 70B: 512, 1024, 2048, 4096, 8192, 24576, 28672
# - LLaMA 405B: 1024, 2048, 4096, 16384, 49152, 53248 (>50k excluded)
# - GPT-4 small: 1024, 2048, 12288, 36864, 49152
# - GPT-4 large: 1024, 2048, 18432, 55296 (>50k), 73728 (>50k) - both excluded
#
# Unique values ≤50,000: 512, 1024, 2048, 4096, 8192, 12288, 16384, 18432,
#                         24576, 28672, 36864, 49152
# ============================================================================

# All unique dimension values from frontier LLMs (excluding values > 50,000)
FRONTIER_DIMS = [
    4096,   # Extended context sequence length
    8192,   # LLaMA 70B hidden dimension
    12288,  # GPT-4 hidden dimension
    16384,  # LLaMA 405B hidden dimension
]

# Generate all permutations: every (M, N, K) combination
# This creates len(FRONTIER_DIMS)³ = 12³ = 1,728 matrix size combinations
FRONTIER_LLM_SIZES = [
    (m, n, k) 
    for m in FRONTIER_DIMS 
    for n in FRONTIER_DIMS 
    for k in FRONTIER_DIMS
]

PRECISIONS = ['fp32', 'fp16', 'bf16']

MEMORY_LAYOUTS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]

# Small batch sizes to avoid memory issues with large permutation grid
# Testing: 1 (single inference), 4, 8, 16 (small batch inference)
BATCH_SIZES = [1, 4, 8, 16]

# Auto-detect available device
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ============================================================================
# CONFIGURATION FLAGS - Control which dimensions to test
# ============================================================================
# Boolean flags to control search space
ENABLE_SIZE_SEARCH = True
ENABLE_PRECISION_SEARCH = True
ENABLE_LAYOUT_SEARCH = True
ENABLE_BATCH_SEARCH = True

# Profiling settings
WARMUP_ITERATIONS = 1  # Single warmup per config to compile kernels
REPEAT_COUNT = 5
WAIT_STEPS = 0  # No wait needed since we already did warmup
ACTIVE_STEPS = 1
RANDOM_SEED = 42

OUTPUT_DIR = './traces_matmul'

def is_valid_matmul_config(m: int, n: int, k: int, transpose_a: bool, transpose_b: bool) -> bool:
    """
    Check if a matmul configuration is valid based on dimension compatibility.
    
    Matrix multiplication: A @ B = C where A[M, K], B[K, N], C[M, N]
    - No transpose: A[M, K] @ B[K, N] → inner dims K, K → always valid
    - Transpose A:  A[M, K]^T @ B[K, N] = A[K, M] @ B[K, N] → inner dims M, K → need M == K
    - Transpose B:  A[M, K] @ B[K, N]^T = A[M, K] @ B[N, K] → inner dims K, N → need K == N
    - Both:         A[M, K]^T @ B[K, N]^T = A[K, M] @ B[N, K] → inner dims M, N → need M == N
    
    Args:
        m: Rows of A
        n: Columns of B
        k: Columns of A / Rows of B (inner dimension)
        transpose_a: Whether A is transposed
        transpose_b: Whether B is transposed
    
    Returns:
        True if the configuration produces a valid matmul operation
    """
    if not transpose_a and not transpose_b:
        # A[M, K] @ B[K, N] - inner dims K and K, always valid
        return True
    elif transpose_a and not transpose_b:
        # A^T[K, M] @ B[K, N] - inner dims M and K, need M == K
        return m == k
    elif not transpose_a and transpose_b:
        # A[M, K] @ B^T[N, K] - inner dims K and N, need K == N
        return k == n
    else:  # both transposes
        # A^T[K, M] @ B^T[N, K] - inner dims M and N, need M == N
        return m == n


def generate_experiment_configs() -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations based on enabled flags.
    
    Filters out invalid transpose configurations where dimensions don't align.
    Square matrices (M = N = K) support all transpose combinations.
    
    Returns:
        List of config dicts with: m, n, k, dtype, transpose_a, transpose_b,
        batch_size, device, config_id
    """
    # Use frontier LLM sizes
    matrix_sizes = FRONTIER_LLM_SIZES if ENABLE_SIZE_SEARCH else [FRONTIER_LLM_SIZES[0]]
    
    precisions = PRECISIONS if ENABLE_PRECISION_SEARCH else [PRECISIONS[0]]
    layouts = MEMORY_LAYOUTS if ENABLE_LAYOUT_SEARCH else [MEMORY_LAYOUTS[0]]
    batch_sizes = BATCH_SIZES if ENABLE_BATCH_SEARCH else [BATCH_SIZES[0]]
    
    # Generate Cartesian product of all enabled parameters
    configs = []
    config_id = 1
    skipped_count = 0
    
    for (m, n, k), dtype, (transpose_a, transpose_b), batch_size in itertools.product(
        matrix_sizes, precisions, layouts, batch_sizes
    ):
        # Validate that this transpose configuration is mathematically valid
        if not is_valid_matmul_config(m, n, k, transpose_a, transpose_b):
            skipped_count += 1
            continue
        
        config = {
            'config_id': config_id,
            'm': m,
            'n': n,
            'k': k,
            'dtype': dtype,
            'transpose_a': transpose_a,
            'transpose_b': transpose_b,
            'batch_size': batch_size,
            'device': DEVICE,
        }
        configs.append(config)
        config_id += 1
    
    if skipped_count > 0:
        print(f"Note: Skipped {skipped_count} invalid transpose configurations")
    
    return configs


def get_config_summary(config: Dict[str, Any]) -> str:
    """Generate a human-readable summary string for a configuration."""
    trans_a = "T" if config['transpose_a'] else ""
    trans_b = "T" if config['transpose_b'] else ""
    batch_str = f"B{config['batch_size']}" if config['batch_size'] > 1 else ""
    
    return (f"Config {config['config_id']:04d}: "
            f"{batch_str}{'x' if batch_str else ''}"
            f"[{config['m']}x{config['k']}]{trans_a} @ "
            f"[{config['k']}x{config['n']}]{trans_b} "
            f"({config['dtype']}) on {config['device']}")


if __name__ == "__main__":
    # Test the configuration generator
    configs = generate_experiment_configs()
    
    # Count square matrices
    square_count = sum(1 for c in configs if c['m'] == c['n'] == c['k'])
    square_unique = len(set((c['m'], c['n'], c['k']) for c in configs if c['m'] == c['n'] == c['k']))
    
    print("=" * 80)
    print(f"PROFILING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Total valid configurations: {len(configs)}")
    print(f"Device: {DEVICE}")
    
    print(f"\nDimension generation:")
    print(f"  - Unique dimension values: {len(FRONTIER_DIMS)}")
    print(f"  - Dimension values: {FRONTIER_DIMS}")
    print(f"  - Matrix size permutations: {len(FRONTIER_LLM_SIZES)} (all M×N×K combinations)")
    print(f"  - Square matrices (M=N=K): {square_unique} unique sizes, {square_count} configs")
    print(f"    (Square matrices support all 4 transpose configurations)")
    
    print(f"\nPrecisions: {PRECISIONS if ENABLE_PRECISION_SEARCH else [PRECISIONS[0]]}")
    print(f"Batch sizes: {BATCH_SIZES if ENABLE_BATCH_SEARCH else [BATCH_SIZES[0]]}")
    print(f"Memory layouts: {len(MEMORY_LAYOUTS) if ENABLE_LAYOUT_SEARCH else 1} variants")
    print(f"  Note: Invalid transpose configs filtered out automatically")
    
    # Calculate theoretical max vs actual
    theoretical_max = (len(FRONTIER_LLM_SIZES) * 
                      (len(PRECISIONS) if ENABLE_PRECISION_SEARCH else 1) * 
                      (len(MEMORY_LAYOUTS) if ENABLE_LAYOUT_SEARCH else 1) * 
                      (len(BATCH_SIZES) if ENABLE_BATCH_SEARCH else 1))
    filtered_out = theoretical_max - len(configs)
    
    print(f"\nFiltering stats:")
    print(f"  - Theoretical maximum: {theoretical_max:,}")
    print(f"  - Valid configurations: {len(configs):,}")
    print(f"  - Filtered out (invalid transposes): {filtered_out:,} ({100*filtered_out/theoretical_max:.1f}%)")
    
    print("=" * 80)
    print(f"\nFirst 10 example configurations:\n")
    
    # Print first 10 configs as examples
    for config in configs[:10]:
        print(get_config_summary(config))
    
    if len(configs) > 10:
        print(f"\n... and {len(configs) - 10} more configurations")

