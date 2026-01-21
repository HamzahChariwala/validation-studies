"""
Configuration for point-to-point communication profiling experiments.
Defines message sizes, GPU pairs, and profiling parameters.
"""

import numpy as np
from typing import List, Dict, Any


# ============================================================================
# MESSAGE SIZES - Spanning latency-bound to bandwidth-bound regimes
# ============================================================================

# Small messages: 1B to 1KB (latency-bound)
SMALL_SIZES = [2**i for i in range(0, 11)]  # 1, 2, 4, ..., 512, 1024

# Medium messages: 2KB to 512KB (transition region)
MEDIUM_SIZES = [2**i for i in range(11, 20)]  # 2048, 4096, ..., 262144, 524288

# Large messages: 1MB to 1GB (bandwidth-bound)
LARGE_SIZES = [2**i for i in range(20, 31)]  # 1048576, ..., 536870912, 1073741824

# All message sizes (31 total, powers of 2)
MESSAGE_SIZES = SMALL_SIZES + MEDIUM_SIZES + LARGE_SIZES

# Min and max message sizes for linear sampling mode
MIN_MESSAGE_SIZE = 1  # 1 byte
MAX_MESSAGE_SIZE = 1073741824  # 1 GB


def generate_linear_message_sizes(num_sizes: int = 50) -> List[int]:
    """
    Generate uniformly-spaced message sizes in linear space.
    
    This provides better data for linear regression compared to log-spaced sizes.
    
    Args:
        num_sizes: Number of message sizes to generate
        
    Returns:
        List of message sizes in bytes, sorted in ascending order
    """
    sizes = np.linspace(MIN_MESSAGE_SIZE, MAX_MESSAGE_SIZE, num_sizes, dtype=np.int64)
    # Ensure sizes are unique and sorted
    sizes = sorted(set(sizes.tolist()))
    return sizes


# ============================================================================
# PROFILING PARAMETERS
# ============================================================================

# Warmup iterations per configuration (to initialize NCCL, compile kernels)
WARMUP_ITERATIONS = 5

# Number of repetitions for each configuration (for statistical significance)
REPEAT_COUNT = 10

# Random seed for reproducible pseudo-random ordering
RANDOM_SEED = 42

# Buffer data type (float32 = 4 bytes per element)
import torch
BUFFER_DTYPE = torch.float32

# ============================================================================
# THERMAL MANAGEMENT
# ============================================================================

# Initial thermal warmup
THERMAL_WARMUP_DURATION = 60  # seconds
THERMAL_TARGET_TEMP = 50      # Celsius

# Temperature monitoring
TEMP_MONITOR_INTERVAL = 1.0   # seconds

# Optional: periodic re-warming (set to None to disable)
PERIODIC_REWARM_INTERVAL = None  # Check every N configs (e.g., 20)
PERIODIC_REWARM_THRESHOLD = 45   # Re-warm if temp drops below this

# ============================================================================
# CLOCK DRIFT MEASUREMENT
# ============================================================================

# Duration for drift measurements (30s is sufficient for main experiment)
DRIFT_MEASUREMENT_DURATION = 30  # seconds

# ============================================================================
# OUTPUT PATHS
# ============================================================================

OUTPUT_DIR = './traces'

# ============================================================================
# GPU PAIR GENERATION
# ============================================================================

def generate_all_configs(world_size: int, message_sizes: List[int]) -> List[Dict[str, Any]]:
    """
    Generate all (src, dst, size) configurations.
    
    Tests all directed GPU pairs (both directions) with all message sizes.
    
    Args:
        world_size: Number of GPUs
        message_sizes: List of message sizes in bytes
    
    Returns:
        List of configuration dictionaries with keys:
        - config_id: Unique identifier
        - src: Source GPU rank
        - dst: Destination GPU rank
        - size: Message size in bytes
    """
    configs = []
    config_id = 0
    
    # All directed pairs (tests both directions)
    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                continue  # Skip self-communication
            
            # All message sizes
            for size in message_sizes:
                configs.append({
                    'config_id': config_id,
                    'src': src,
                    'dst': dst,
                    'size': size,
                })
                config_id += 1
    
    return configs


def get_config_summary(config: Dict[str, Any]) -> str:
    """Generate human-readable summary for a configuration."""
    size_str = format_size(config['size'])
    return (f"Config {config['config_id']:04d}: "
            f"GPU {config['src']} → GPU {config['dst']}, "
            f"{size_str}")


def format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024**2:
        return f"{size_bytes // 1024}KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes // (1024**2)}MB"
    else:
        return f"{size_bytes // (1024**3)}GB"


if __name__ == "__main__":
    # Test configuration generation
    print("=" * 80)
    print("POINT-TO-POINT PROFILING CONFIGURATION")
    print("=" * 80)
    
    # Example: 4 GPUs
    test_world_size = 4
    configs = generate_all_configs(test_world_size, MESSAGE_SIZES)
    
    print(f"\nGPUs: {test_world_size}")
    print(f"Message sizes: {len(MESSAGE_SIZES)} ({MESSAGE_SIZES[0]} to {MESSAGE_SIZES[-1]} bytes)")
    print(f"Total configurations: {len(configs)}")
    print(f"  = {test_world_size * (test_world_size - 1)} GPU pairs × {len(MESSAGE_SIZES)} sizes")
    
    print(f"\nProfiling parameters:")
    print(f"  Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"  Repetitions: {REPEAT_COUNT}")
    print(f"  Random seed: {RANDOM_SEED}")
    
    print(f"\nThermal management:")
    print(f"  Initial warmup: {THERMAL_WARMUP_DURATION}s to {THERMAL_TARGET_TEMP}°C")
    print(f"  Monitoring interval: {TEMP_MONITOR_INTERVAL}s")
    
    print(f"\nExpected runtime:")
    total_tests = len(configs) * REPEAT_COUNT
    estimated_time_per_test = 0.001  # ~1ms per small message, more for large
    estimated_total = total_tests * estimated_time_per_test + THERMAL_WARMUP_DURATION + 2 * DRIFT_MEASUREMENT_DURATION
    print(f"  Total tests: {total_tests}")
    print(f"  Estimated time: ~{estimated_total / 60:.1f} minutes (rough estimate)")
    
    print(f"\nFirst 5 example configurations:")
    for config in configs[:5]:
        print(f"  {get_config_summary(config)}")
    
    print(f"\n... and {len(configs) - 5} more configurations")
    print("=" * 80)

