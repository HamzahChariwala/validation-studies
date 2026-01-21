"""
Launch and environment utilities for distributed experiments.

Provides helpers for pre-flight checks, GPU clock management,
thermal warmup, and environment validation.
"""

import os
import subprocess
import time
import torch
import torch.distributed
from typing import Dict, List, Optional, Tuple


def validate_environment() -> Tuple[bool, List[str]]:
    """
    Validate that the environment is ready for distributed experiments.
    
    Checks:
    - CUDA available
    - Multiple GPUs present
    - NCCL available
    - Required environment variables set (if using torchrun)
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check CUDA
    if not torch.cuda.is_available():
        issues.append("CUDA not available")
    
    # Check multiple GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus < 2:
        issues.append(f"Need at least 2 GPUs, found {num_gpus}")
    
    # Check NCCL
    try:
        if not hasattr(torch.distributed, 'is_nccl_available') or \
           not torch.distributed.is_nccl_available():
            issues.append("NCCL not available")
    except (ImportError, AttributeError):
        issues.append("torch.distributed not available")
    
    # Check nvidia-smi
    try:
        subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        issues.append("nvidia-smi not available")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def lock_gpu_clocks(
    gpu_ids: List[int],
    clock_mhz: Optional[int] = None,
    use_max: bool = True
) -> bool:
    """
    Lock GPU clocks to a fixed frequency.
    
    Requires sudo access. If clock_mhz not specified, locks to max clock.
    
    Args:
        gpu_ids: List of GPU IDs to lock
        clock_mhz: Clock frequency in MHz (None = use max)
        use_max: If True and clock_mhz is None, lock to max frequency
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Enable persistence mode
        subprocess.run(
            ['sudo', 'nvidia-smi', '-pm', '1'],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        for gpu_id in gpu_ids:
            if clock_mhz is None and use_max:
                # Query max clock
                cmd = f"nvidia-smi -i {gpu_id} --query-gpu=clocks.max.graphics --format=csv,noheader,nounits"
                result = subprocess.check_output(cmd.split())
                clock_mhz = int(float(result.decode().strip()))
            
            # Lock clock
            subprocess.run(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-lgc', str(clock_mhz)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        print(f"✓ Locked GPU clocks to {clock_mhz} MHz for GPUs {gpu_ids}")
        return True
    
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        print(f"✗ Failed to lock GPU clocks: {e}")
        print("  Note: Requires sudo access. Continuing without locked clocks.")
        return False


def unlock_gpu_clocks(gpu_ids: List[int]) -> bool:
    """
    Unlock GPU clocks (reset to auto).
    
    Args:
        gpu_ids: List of GPU IDs to unlock
    
    Returns:
        True if successful, False otherwise
    """
    try:
        for gpu_id in gpu_ids:
            subprocess.run(
                ['sudo', 'nvidia-smi', '-i', str(gpu_id), '-rgc'],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        print(f"✓ Unlocked GPU clocks for GPUs {gpu_ids}")
        return True
    
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
        print(f"✗ Failed to unlock GPU clocks: {e}")
        return False


def thermal_warmup(
    device: torch.device,
    duration_seconds: float = 60.0,
    target_temp_min: int = 50
) -> Dict[str, float]:
    """
    Warm up GPU to stable thermal state.
    
    Runs dummy compute workload until target temperature reached or
    duration elapsed.
    
    Args:
        device: GPU device to warm up
        duration_seconds: Maximum warmup duration
        target_temp_min: Minimum target temperature (Celsius)
    
    Returns:
        Dictionary with warmup metrics
    """
    print(f"Thermal warmup on {device} for up to {duration_seconds}s...")
    
    # Get GPU ID from device
    if isinstance(device, torch.device):
        gpu_id = device.index if device.index is not None else 0
    else:
        gpu_id = 0
    
    # Create dummy workload
    size = 8192
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < duration_seconds:
        # Compute-intensive operation
        _ = torch.matmul(A, B)
        iterations += 1
        
        # Check temperature every 10 iterations
        if iterations % 10 == 0:
            torch.cuda.synchronize(device)
            try:
                cmd = f"nvidia-smi -i {gpu_id} --query-gpu=temperature.gpu --format=csv,noheader,nounits"
                result = subprocess.check_output(cmd.split())
                temp = int(result.decode().strip())
                
                if temp >= target_temp_min:
                    print(f"  Target temperature reached: {temp}°C")
                    break
            except (subprocess.CalledProcessError, ValueError):
                pass
    
    torch.cuda.synchronize(device)
    elapsed = time.time() - start_time
    
    # Get final temperature
    try:
        cmd = f"nvidia-smi -i {gpu_id} --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split())
        final_temp = int(result.decode().strip())
    except (subprocess.CalledProcessError, ValueError):
        final_temp = -1
    
    print(f"  Warmup complete: {elapsed:.1f}s, {iterations} iterations, {final_temp}°C")
    
    return {
        'device': str(device),
        'gpu_id': gpu_id,
        'duration_s': elapsed,
        'iterations': iterations,
        'final_temp_c': final_temp,
        'target_temp_c': target_temp_min,
    }


def check_gpu_idle() -> List[int]:
    """
    Check which GPUs are idle (no running processes).
    
    Returns:
        List of idle GPU IDs
    """
    try:
        # Query compute processes
        cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
        result = subprocess.check_output(cmd.split())
        
        # If no output, all GPUs idle
        if not result.decode().strip():
            return list(range(torch.cuda.device_count()))
        
        # Otherwise, check per-GPU
        idle_gpus = []
        for gpu_id in range(torch.cuda.device_count()):
            cmd = f"nvidia-smi -i {gpu_id} --query-compute-apps=pid --format=csv,noheader"
            result = subprocess.check_output(cmd.split())
            if not result.decode().strip():
                idle_gpus.append(gpu_id)
        
        return idle_gpus
    
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If can't check, assume all idle
        return list(range(torch.cuda.device_count()))


def print_environment_summary():
    """Print summary of execution environment."""
    print("=" * 80)
    print("ENVIRONMENT SUMMARY")
    print("=" * 80)
    
    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List GPUs
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # NCCL
    try:
        if hasattr(torch.cuda.nccl, 'version'):
            nccl_ver = torch.cuda.nccl.version()
            print(f"NCCL version: {nccl_ver[0]}.{nccl_ver[1]}.{nccl_ver[2]}")
    except AttributeError:
        print("NCCL version: N/A")
    
    # Environment variables
    print("\nRelevant environment variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'NCCL_DEBUG', 'NCCL_IB_DISABLE']
    for var in env_vars:
        value = os.environ.get(var, 'not set')
        print(f"  {var}: {value}")
    
    print("=" * 80)

