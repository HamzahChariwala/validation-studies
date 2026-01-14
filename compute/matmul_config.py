"""
Configuration file for matmul profiling experiments.
Defines parameter space and grid search generator.
"""

import itertools
from typing import List, Dict, Any


# Matrix size options: (M, N, K) for A[M×K] @ B[K×N] = C[M×N]
MATRIX_SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
]

RECTANGULAR_SIZES = [
    (1024, 512, 2048),
    (2048, 1024, 512),
    (512, 2048, 1024),
]

PRECISIONS = ['fp32', 'fp16', 'bf16']

MEMORY_LAYOUTS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]

BATCH_SIZES = [1, 8, 32, 128]

# Auto-detect available device
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# Boolean flags to control search space
ENABLE_SIZE_SEARCH = True
ENABLE_PRECISION_SEARCH = True
ENABLE_LAYOUT_SEARCH = True
ENABLE_BATCH_SEARCH = True
ENABLE_RECTANGULAR = False

# Profiling settings
WARMUP_ITERATIONS = 3
REPEAT_COUNT = 5
WAIT_STEPS = 1
ACTIVE_STEPS = 1
RANDOM_SEED = 42

OUTPUT_DIR = './traces_matmul'

def generate_experiment_configs() -> List[Dict[str, Any]]:
    """
    Generate all experiment configurations based on enabled flags.
    
    Returns:
        List of config dicts with: m, n, k, dtype, transpose_a, transpose_b,
        batch_size, device, config_id
    """
    # Determine active parameter values based on flags
    sizes = []
    if ENABLE_SIZE_SEARCH:
        sizes.extend(MATRIX_SIZES)
        if ENABLE_RECTANGULAR:
            sizes.extend(RECTANGULAR_SIZES)
    else:
        sizes = [MATRIX_SIZES[0]]
    
    precisions = PRECISIONS if ENABLE_PRECISION_SEARCH else [PRECISIONS[0]]
    layouts = MEMORY_LAYOUTS if ENABLE_LAYOUT_SEARCH else [MEMORY_LAYOUTS[0]]
    batch_sizes = BATCH_SIZES if ENABLE_BATCH_SEARCH else [BATCH_SIZES[0]]
    
    # Generate Cartesian product of all enabled parameters
    configs = []
    config_id = 1
    
    for (m, n, k), dtype, (transpose_a, transpose_b), batch_size in itertools.product(
        sizes, precisions, layouts, batch_sizes
    ):
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
    print(f"Total configurations: {len(configs)}\n")
    
    # Print first 10 configs as examples
    for config in configs[:10]:
        print(get_config_summary(config))
    
    if len(configs) > 10:
        print(f"... and {len(configs) - 10} more configurations")

