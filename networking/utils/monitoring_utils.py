"""
Monitoring utilities for continuous system state tracking.

Provides background logging for GPU temperature, power, clocks, and other
system metrics during experiments.
"""

import threading
import time
import subprocess
import os
from typing import List, Optional, Dict, Any
from datetime import datetime


class TemperatureLogger:
    """
    Background thread to continuously log GPU temperatures and system state.
    
    Samples GPU metrics at regular intervals without blocking the main experiment.
    Useful for detecting thermal throttling, performance variance, and system issues.
    
    Example:
        logger = TemperatureLogger(gpu_ids=[0, 1, 2, 3], interval=1.0)
        logger.start()
        
        # ... run your experiment ...
        
        logger.stop()
        logger.save('./traces/temperatures.csv')
        stats = logger.get_stats()
        print(f"GPU 0: {stats['gpu_0']['temp_mean']:.1f}°C")
    """
    
    def __init__(
        self,
        gpu_ids: List[int],
        interval: float = 1.0,
        metrics: Optional[List[str]] = None
    ):
        """
        Initialize temperature logger.
        
        Args:
            gpu_ids: List of GPU IDs to monitor
            interval: Sampling interval in seconds (default: 1.0)
            metrics: List of metrics to log (default: temp, power, clock)
                     Options: 'temperature', 'power', 'clock_graphics', 
                             'clock_sm', 'clock_memory', 'utilization'
        """
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.running = False
        self.thread = None
        self.data = []
        self.start_time = None
        
        # Default metrics to log
        if metrics is None:
            self.metrics = ['temperature', 'power', 'clock_graphics']
        else:
            self.metrics = metrics
        
        # Metric to nvidia-smi query mapping
        self.query_map = {
            'temperature': 'temperature.gpu',
            'power': 'power.draw',
            'clock_graphics': 'clocks.current.graphics',
            'clock_sm': 'clocks.current.sm',
            'clock_memory': 'clocks.current.memory',
            'utilization': 'utilization.gpu',
            'memory_used': 'memory.used',
        }
    
    def start(self):
        """Start background logging thread."""
        if self.running:
            print("Warning: Temperature logger already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._log_loop, daemon=True)
        self.thread.start()
        
        print(f"Temperature logger started:")
        print(f"  GPUs: {self.gpu_ids}")
        print(f"  Interval: {self.interval}s")
        print(f"  Metrics: {self.metrics}")
    
    def stop(self):
        """Stop background logging thread."""
        if not self.running:
            print("Warning: Temperature logger not running")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        
        print(f"Temperature logger stopped: {len(self.data)} samples collected")
    
    def _query_gpu(self, gpu_id: int, metric: str) -> Optional[float]:
        """
        Query a single metric for a GPU.
        
        Args:
            gpu_id: GPU device ID
            metric: Metric name (from self.query_map)
        
        Returns:
            Metric value as float, or None if query failed
        """
        if metric not in self.query_map:
            return None
        
        query = self.query_map[metric]
        
        try:
            cmd = f"nvidia-smi -i {gpu_id} --query-gpu={query} --format=csv,noheader,nounits"
            result = subprocess.check_output(
                cmd.split(),
                stderr=subprocess.DEVNULL,
                timeout=1.0
            )
            value = float(result.decode().strip())
            return value
        except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
            return None
    
    def _log_loop(self):
        """Background loop that continuously samples GPU metrics."""
        while self.running:
            timestamp = time.time()
            elapsed_s = timestamp - self.start_time
            
            for gpu_id in self.gpu_ids:
                sample = {
                    'timestamp_abs': timestamp,
                    'elapsed_s': elapsed_s,
                    'gpu_id': gpu_id,
                }
                
                # Query all requested metrics
                for metric in self.metrics:
                    value = self._query_gpu(gpu_id, metric)
                    sample[metric] = value
                
                self.data.append(sample)
            
            # Sleep for interval
            time.sleep(self.interval)
    
    def save(self, path: str):
        """
        Save collected data to CSV file.
        
        Args:
            path: Output file path (e.g., './traces/temperatures.csv')
        """
        if not self.data:
            print("Warning: No data to save")
            return
        
        import csv
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.data[0].keys())
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"Temperature data saved to {path} ({len(self.data)} samples)")
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics for each GPU.
        
        Returns:
            Dictionary with statistics per GPU:
            {
                'gpu_0': {'temp_mean': 65.2, 'temp_std': 1.5, ...},
                'gpu_1': {'temp_mean': 63.8, 'temp_std': 1.2, ...},
            }
        """
        if not self.data:
            return {}
        
        stats = {}
        
        # Group by GPU
        for gpu_id in self.gpu_ids:
            gpu_data = [s for s in self.data if s['gpu_id'] == gpu_id]
            
            if not gpu_data:
                continue
            
            gpu_stats = {}
            
            # Compute stats for each metric
            for metric in self.metrics:
                values = [s[metric] for s in gpu_data if s[metric] is not None]
                
                if values:
                    gpu_stats[f'{metric}_mean'] = sum(values) / len(values)
                    gpu_stats[f'{metric}_std'] = (
                        sum((x - gpu_stats[f'{metric}_mean'])**2 for x in values) / len(values)
                    ) ** 0.5
                    gpu_stats[f'{metric}_min'] = min(values)
                    gpu_stats[f'{metric}_max'] = max(values)
                    gpu_stats[f'{metric}_range'] = max(values) - min(values)
            
            stats[f'gpu_{gpu_id}'] = gpu_stats
        
        return stats
    
    def print_summary(self):
        """Print summary statistics to console."""
        stats = self.get_stats()
        
        if not stats:
            print("No data available")
            return
        
        print("\n" + "="*70)
        print("TEMPERATURE LOGGER SUMMARY")
        print("="*70)
        
        for gpu_key, gpu_stats in stats.items():
            print(f"\n{gpu_key.upper()}:")
            
            if 'temperature_mean' in gpu_stats:
                print(f"  Temperature: {gpu_stats['temperature_mean']:.1f}°C "
                      f"± {gpu_stats['temperature_std']:.1f}°C "
                      f"(range: {gpu_stats['temperature_min']:.0f}-"
                      f"{gpu_stats['temperature_max']:.0f}°C)")
            
            if 'power_mean' in gpu_stats:
                print(f"  Power:       {gpu_stats['power_mean']:.1f}W "
                      f"± {gpu_stats['power_std']:.1f}W")
            
            if 'clock_graphics_mean' in gpu_stats:
                print(f"  Clock:       {gpu_stats['clock_graphics_mean']:.0f} MHz "
                      f"(range: {gpu_stats['clock_graphics_min']:.0f}-"
                      f"{gpu_stats['clock_graphics_max']:.0f} MHz)")
        
        print("="*70 + "\n")
    
    def check_throttling(self, temp_threshold: float = 80.0) -> List[Dict[str, Any]]:
        """
        Check if any GPU exceeded temperature threshold.
        
        Args:
            temp_threshold: Temperature threshold in Celsius
        
        Returns:
            List of throttling events with timestamps and GPUs
        """
        throttling_events = []
        
        for sample in self.data:
            if 'temperature' in sample and sample['temperature'] is not None:
                if sample['temperature'] > temp_threshold:
                    throttling_events.append({
                        'timestamp': sample['timestamp_abs'],
                        'elapsed_s': sample['elapsed_s'],
                        'gpu_id': sample['gpu_id'],
                        'temperature': sample['temperature'],
                    })
        
        return throttling_events


def quick_warmup(device, duration_seconds: float = 10, target_temp_min: int = 45):
    """
    Quick GPU warmup to maintain temperature.
    
    Lighter than full thermal_warmup, used between experiment phases to
    prevent GPU from cooling down too much during idle periods.
    
    Args:
        device: CUDA device
        duration_seconds: Maximum warmup duration (default: 10s)
        target_temp_min: Minimum target temperature (default: 45°C)
    """
    import torch
    
    gpu_id = torch.cuda.current_device() if device == 'cuda' else int(str(device).split(':')[1])
    
    # Lighter compute workload than full warmup
    size = 4096  # Smaller than full warmup
    A = torch.randn(size, size, device=device, dtype=torch.float32)
    B = torch.randn(size, size, device=device, dtype=torch.float32)
    
    start_time = time.time()
    current_temp = 0
    
    while time.time() - start_time < duration_seconds:
        _ = torch.matmul(A, B)
        
        # Check temperature
        try:
            cmd = f"nvidia-smi -i {gpu_id} --query-gpu=temperature.gpu --format=csv,noheader,nounits"
            result = subprocess.check_output(cmd.split())
            current_temp = int(result.decode().strip())
            
            if current_temp >= target_temp_min:
                break
        except (subprocess.CalledProcessError, ValueError):
            pass
    
    torch.cuda.synchronize(device)


def check_temperature(gpu_id: int) -> Optional[int]:
    """
    Quick temperature check for a GPU.
    
    Args:
        gpu_id: GPU device ID
    
    Returns:
        Current temperature in Celsius, or None if query failed
    """
    try:
        cmd = f"nvidia-smi -i {gpu_id} --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        result = subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL)
        return int(result.decode().strip())
    except (subprocess.CalledProcessError, ValueError):
        return None

