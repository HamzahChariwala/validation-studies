#!/usr/bin/env python3
"""
Compare the smooth approximation vs actual roofline model.
"""

import numpy as np
import matplotlib.pyplot as plt

# T4 GPU specifications
MEMORY_BANDWIDTH_GBS = 320  # GB/s
PEAK_COMPUTE_FP32_TFLOPS = 8.141  # TFLOPS
PEAK_COMPUTE_FP16_TFLOPS = 65.13  # TFLOPS

# Convert to TFLOP/s per FLOPs/Byte
memory_bandwidth_tflops_per_ai = MEMORY_BANDWIDTH_GBS / 1000  # 0.32


def actual_roofline(ai, peak_compute, memory_bandwidth):
    """
    Actual roofline model with sharp corner.
    
    Throughput = min(memory_bandwidth * AI, peak_compute)
    """
    return np.minimum(memory_bandwidth * ai, peak_compute)


def smooth_roofline(ai, peak_compute, memory_bandwidth):
    """
    Smooth approximation using harmonic mean.
    
    Throughput = 1 / (1/(memory_bandwidth * AI) + 1/peak_compute)
    
    This is equivalent to:
    Throughput = (memory_bandwidth * AI * peak_compute) / (memory_bandwidth * AI + peak_compute)
    """
    return 1 / (1 / (memory_bandwidth * ai) + 1 / peak_compute)


def plot_comparison():
    """Create comparison plots of actual vs smooth roofline."""
    
    # Generate range of arithmetic intensities
    ai_values = np.linspace(1, 1000, 1000)
    
    # Calculate both models for FP32
    actual_fp32 = actual_roofline(ai_values, PEAK_COMPUTE_FP32_TFLOPS, memory_bandwidth_tflops_per_ai)
    smooth_fp32 = smooth_roofline(ai_values, PEAK_COMPUTE_FP32_TFLOPS, memory_bandwidth_tflops_per_ai)
    
    # Calculate both models for FP16
    actual_fp16 = actual_roofline(ai_values, PEAK_COMPUTE_FP16_TFLOPS, memory_bandwidth_tflops_per_ai)
    smooth_fp16 = smooth_roofline(ai_values, PEAK_COMPUTE_FP16_TFLOPS, memory_bandwidth_tflops_per_ai)
    
    # Calculate the ridge points (where memory-bound meets compute-bound)
    ridge_fp32 = PEAK_COMPUTE_FP32_TFLOPS / memory_bandwidth_tflops_per_ai
    ridge_fp16 = PEAK_COMPUTE_FP16_TFLOPS / memory_bandwidth_tflops_per_ai
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # --- FP32 Linear Scale ---
    ax = axes[0, 0]
    ax.plot(ai_values, actual_fp32, 'b-', linewidth=2, label='Actual Roofline (min)')
    ax.plot(ai_values, smooth_fp32, 'r--', linewidth=2, label='Smooth Approximation (harmonic)')
    ax.axvline(ridge_fp32, color='gray', linestyle=':', alpha=0.5, label=f'Ridge Point: {ridge_fp32:.2f}')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11)
    ax.set_ylabel('Throughput (TFLOP/s)', fontsize=11)
    ax.set_title('FP32 Roofline - Linear Scale', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    # --- FP32 Log Scale ---
    ax = axes[0, 1]
    ax.plot(ai_values, actual_fp32, 'b-', linewidth=2, label='Actual Roofline (min)')
    ax.plot(ai_values, smooth_fp32, 'r--', linewidth=2, label='Smooth Approximation (harmonic)')
    ax.axvline(ridge_fp32, color='gray', linestyle=':', alpha=0.5, label=f'Ridge Point: {ridge_fp32:.2f}')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11)
    ax.set_ylabel('Throughput (TFLOP/s)', fontsize=11)
    ax.set_title('FP32 Roofline - Log Scale', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # --- FP16 Linear Scale ---
    ax = axes[1, 0]
    ax.plot(ai_values, actual_fp16, 'b-', linewidth=2, label='Actual Roofline (min)')
    ax.plot(ai_values, smooth_fp16, 'r--', linewidth=2, label='Smooth Approximation (harmonic)')
    ax.axvline(ridge_fp16, color='gray', linestyle=':', alpha=0.5, label=f'Ridge Point: {ridge_fp16:.2f}')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11)
    ax.set_ylabel('Throughput (TFLOP/s)', fontsize=11)
    ax.set_title('FP16 Roofline - Linear Scale', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 400)
    
    # --- FP16 Log Scale ---
    ax = axes[1, 1]
    ax.plot(ai_values, actual_fp16, 'b-', linewidth=2, label='Actual Roofline (min)')
    ax.plot(ai_values, smooth_fp16, 'r--', linewidth=2, label='Smooth Approximation (harmonic)')
    ax.axvline(ridge_fp16, color='gray', linestyle=':', alpha=0.5, label=f'Ridge Point: {ridge_fp16:.2f}')
    ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=11)
    ax.set_ylabel('Throughput (TFLOP/s)', fontsize=11)
    ax.set_title('FP16 Roofline - Log Scale', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('roofline_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to: roofline_comparison.png")
    
    # Calculate and print differences at key points
    print("\n" + "="*70)
    print("DIFFERENCE ANALYSIS")
    print("="*70)
    
    # Check differences at various points
    test_ais = [10, 25.44, 50, 100, 203.53, 300, 500]
    
    print("\nFP32 (Peak: {:.3f} TFLOP/s, Ridge: {:.2f} FLOPs/Byte):".format(
        PEAK_COMPUTE_FP32_TFLOPS, ridge_fp32))
    print(f"{'AI':>10s} {'Actual':>12s} {'Smooth':>12s} {'Diff':>12s} {'Diff %':>10s}")
    print("-" * 70)
    for ai in test_ais:
        actual = actual_roofline(ai, PEAK_COMPUTE_FP32_TFLOPS, memory_bandwidth_tflops_per_ai)
        smooth = smooth_roofline(ai, PEAK_COMPUTE_FP32_TFLOPS, memory_bandwidth_tflops_per_ai)
        diff = smooth - actual
        diff_pct = (diff / actual) * 100 if actual > 0 else 0
        print(f"{ai:10.2f} {actual:12.4f} {smooth:12.4f} {diff:12.4f} {diff_pct:9.2f}%")
    
    print("\nFP16 (Peak: {:.3f} TFLOP/s, Ridge: {:.2f} FLOPs/Byte):".format(
        PEAK_COMPUTE_FP16_TFLOPS, ridge_fp16))
    print(f"{'AI':>10s} {'Actual':>12s} {'Smooth':>12s} {'Diff':>12s} {'Diff %':>10s}")
    print("-" * 70)
    for ai in test_ais:
        actual = actual_roofline(ai, PEAK_COMPUTE_FP16_TFLOPS, memory_bandwidth_tflops_per_ai)
        smooth = smooth_roofline(ai, PEAK_COMPUTE_FP16_TFLOPS, memory_bandwidth_tflops_per_ai)
        diff = smooth - actual
        diff_pct = (diff / actual) * 100 if actual > 0 else 0
        print(f"{ai:10.2f} {actual:12.4f} {smooth:12.4f} {diff:12.4f} {diff_pct:9.2f}%")
    
    print("\n" + "="*70)
    print("\nKEY OBSERVATIONS:")
    print("- The smooth approximation is always <= actual roofline (conservative)")
    print("- Maximum difference occurs near the ridge point")
    print("- Far from ridge point, both models converge")
    print("- The smooth model is differentiable everywhere (good for optimization)")
    print("="*70 + "\n")


if __name__ == '__main__':
    plot_comparison()

