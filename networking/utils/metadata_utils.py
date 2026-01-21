"""
Metadata capture utilities for hardware and system state.

Collects GPU properties, topology, temperatures, clock speeds, and other
system information relevant to performance analysis.
"""

import os
import subprocess
import torch
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


def query_nvidia_smi(gpu_id: int, query: str) -> str:
    """
    Query nvidia-smi for a specific metric.
    
    Args:
        gpu_id: GPU device ID
        query: nvidia-smi query string (e.g., 'temperature.gpu')
    
    Returns:
        Query result as string
    """
    try:
        cmd = f"nvidia-smi -i {gpu_id} --query-gpu={query} --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        return result.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"


def capture_gpu_metadata(rank: int, gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Capture comprehensive GPU metadata for a specific rank.
    
    Args:
        rank: Process rank
        gpu_id: GPU device ID (defaults to rank if single-node)
    
    Returns:
        Dictionary of metadata
    """
    if gpu_id is None:
        gpu_id = rank
    
    metadata = {
        'rank': rank,
        'gpu_id': gpu_id,
        'timestamp': datetime.now().isoformat(),
    }
    
    # PyTorch GPU properties
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        props = torch.cuda.get_device_properties(gpu_id)
        metadata.update({
            'gpu_name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'total_memory_gb': props.total_memory / (1024**3),
            'multiprocessor_count': props.multi_processor_count,
        })
    
    # nvidia-smi queries
    metadata.update({
        'driver_version': query_nvidia_smi(gpu_id, 'driver_version'),
        'clock_graphics_mhz': query_nvidia_smi(gpu_id, 'clocks.current.graphics'),
        'clock_sm_mhz': query_nvidia_smi(gpu_id, 'clocks.current.sm'),
        'clock_memory_mhz': query_nvidia_smi(gpu_id, 'clocks.current.memory'),
        'clock_graphics_max_mhz': query_nvidia_smi(gpu_id, 'clocks.max.graphics'),
        'clock_sm_max_mhz': query_nvidia_smi(gpu_id, 'clocks.max.sm'),
        'clock_memory_max_mhz': query_nvidia_smi(gpu_id, 'clocks.max.memory'),
        'temperature_c': query_nvidia_smi(gpu_id, 'temperature.gpu'),
        'temperature_max_c': query_nvidia_smi(gpu_id, 'temperature.gpu.tlimit'),
        'power_draw_w': query_nvidia_smi(gpu_id, 'power.draw'),
        'power_limit_w': query_nvidia_smi(gpu_id, 'power.limit'),
        'utilization_gpu_percent': query_nvidia_smi(gpu_id, 'utilization.gpu'),
        'utilization_memory_percent': query_nvidia_smi(gpu_id, 'utilization.memory'),
        'memory_used_mb': query_nvidia_smi(gpu_id, 'memory.used'),
        'memory_total_mb': query_nvidia_smi(gpu_id, 'memory.total'),
    })
    
    # CUDA/PyTorch versions
    metadata.update({
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
    })
    
    return metadata


def capture_topology() -> Dict[str, Any]:
    """
    Capture GPU topology information (NVLink, PCIe).
    
    Returns:
        Dictionary with topology information
    """
    topology = {}
    
    try:
        # nvidia-smi topo matrix
        cmd = "nvidia-smi topo -m"
        result = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        topology['topo_matrix'] = result.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        topology['topo_matrix'] = "N/A"
    
    try:
        # nvlink status (if available)
        cmd = "nvidia-smi nvlink --status"
        result = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        topology['nvlink_status'] = result.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        topology['nvlink_status'] = "N/A"
    
    return topology


def capture_nccl_info() -> Dict[str, Any]:
    """
    Capture NCCL configuration and version.
    
    Returns:
        Dictionary with NCCL information
    """
    nccl_info = {}
    
    # NCCL version (if available)
    try:
        # Try to get NCCL version from torch
        if hasattr(torch.cuda.nccl, 'version'):
            version = torch.cuda.nccl.version()
            nccl_info['nccl_version'] = f"{version[0]}.{version[1]}.{version[2]}"
        else:
            nccl_info['nccl_version'] = "N/A"
    except AttributeError:
        nccl_info['nccl_version'] = "N/A"
    
    # NCCL environment variables
    nccl_env_vars = [
        'NCCL_DEBUG',
        'NCCL_DEBUG_SUBSYS',
        'NCCL_IB_DISABLE',
        'NCCL_P2P_DISABLE',
        'NCCL_SHM_DISABLE',
        'NCCL_SOCKET_IFNAME',
        'NCCL_ALGO',
        'NCCL_PROTO',
    ]
    
    nccl_info['environment'] = {
        var: os.environ.get(var, 'not set')
        for var in nccl_env_vars
    }
    
    return nccl_info


def save_metadata(metadata: Dict[str, Any], output_path: str):
    """
    Save metadata to YAML file.
    
    Args:
        metadata: Dictionary of metadata
        output_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def capture_full_metadata(rank: int, gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Capture all metadata (GPU, topology, NCCL) in one call.
    
    Args:
        rank: Process rank
        gpu_id: GPU device ID (defaults to rank)
    
    Returns:
        Complete metadata dictionary
    """
    metadata = {
        'gpu': capture_gpu_metadata(rank, gpu_id),
    }
    
    # Only capture topology once (from rank 0)
    if rank == 0:
        metadata['topology'] = capture_topology()
        metadata['nccl'] = capture_nccl_info()
    
    return metadata

