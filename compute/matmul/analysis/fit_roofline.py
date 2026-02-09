#!/usr/bin/env python3
"""
Roofline Model Fitting - Multi-Method

Fits continuous roofline models to performance data using the approximation:
    T(I) = [(B*I)^(-k) + P^(-k)]^(-1/k)

where:
    - T = throughput (ops/sec)
    - I = arithmetic intensity (ops/byte)
    - B = effective bandwidth (bytes/sec)
    - P = effective compute performance (ops/sec)
    - k = sharpness parameter (controls the "knee" steepness)

Provides multiple loss functions for robust optimization, similar to the networking profiler:

Methods:
1. L2 (Ordinary Least Squares) - Most stable convergence
2. Huber - Robust to outliers with hyperparameter δ
3. IRLS (Iteratively Reweighted Least Squares) - Robust without hyperparameter

Usage:
    from compute.matmul.analysis.fit_roofline_multi import fit_roofline_all_methods
    
    # Prepare data
    I = arithmetic_intensity_array  # ops/byte
    T = throughput_array            # ops/sec
    
    # Fit with all 3 methods
    results = fit_roofline_all_methods(I, T, k_fixed=100.0)
    
    # Compare results
    for method_name, result in results.items():
        if result['success']:
            print(f"{method_name}: BW={result['B']/1e9:.2f} GB/s, R²={result['metrics']['r_squared']:.4f}")
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from scipy.optimize import minimize


# ============================================================================
# Core Roofline Model Functions
# ============================================================================

def roofline_model(I: np.ndarray, B: float, P: float, k: float) -> np.ndarray:
    """
    Compute throughput using continuous roofline approximation.
    
    T(I) = [(B*I)^(-k) + P^(-k)]^(-1/k)
    
    This is a smooth approximation of the traditional roofline model:
        T(I) = min(B*I, P)
    
    When k → ∞:
        - If B*I << P: T → B*I (memory-bound)
        - If P << B*I: T → P (compute-bound)
    
    Args:
        I: Arithmetic intensity (ops/byte) - can be scalar or array
        B: Effective bandwidth (bytes/sec)
        P: Effective compute performance (ops/sec)
        k: Sharpness parameter (higher = sharper knee, k→∞ approaches traditional roofline)
    
    Returns:
        T: Throughput (ops/sec) - same shape as I
    """
    I = np.asarray(I)
    
    # Handle edge cases
    if B <= 0 or P <= 0 or k <= 0:
        raise ValueError(f"Parameters must be positive: B={B}, P={P}, k={k}")
    
    # Compute T(I) = [(B*I)^(-k) + P^(-k)]^(-1/k)
    # For numerical stability with large k, compute in log space
    BI = B * I
    
    if k > 50:
        # Use log-space computation for large k to avoid overflow
        # T = exp(-1/k * log((B*I)^(-k) + P^(-k)))
        # = exp(-1/k * log(exp(-k*log(B*I)) + exp(-k*log(P))))
        log_BI = np.log(BI)
        log_P = np.log(P)
        
        # Compute log-sum-exp trick: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
        a = -k * log_BI
        b = -k * log_P
        max_ab = np.maximum(a, b)
        log_sum = max_ab + np.log(1.0 + np.exp(-np.abs(a - b)))
        
        T = np.exp(-log_sum / k)
    else:
        # Direct computation for moderate k
        term1 = np.power(BI, -k)
        term2 = np.power(P, -k)
        T = np.power(term1 + term2, -1.0/k)
    
    return T


def compute_metrics(I: np.ndarray, T: np.ndarray, T_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive goodness-of-fit metrics.
    
    Args:
        I: Arithmetic intensity array (ops/byte)
        T: Measured throughput array (ops/sec)
        T_pred: Predicted throughput array (ops/sec)
    
    Returns:
        Dictionary with metrics:
        - r_squared: Coefficient of determination (1 = perfect fit)
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - mape: Mean absolute percentage error
        - max_residual: Maximum absolute residual
        - residuals: Array of residuals (T - T_pred)
    """
    residuals = T - T_pred
    
    # R² (coefficient of determination)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((T - np.mean(T))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # RMSE (root mean squared error)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # MAE (mean absolute error)
    mae = np.mean(np.abs(residuals))
    
    # MAPE (mean absolute percentage error) - handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs(residuals / T)) * 100
        if not np.isfinite(mape):
            mape = np.inf
    
    # Maximum absolute residual
    max_residual = np.max(np.abs(residuals))
    
    return {
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'max_residual': float(max_residual),
        'residuals': residuals
    }


# ============================================================================
# Loss Functions
# ============================================================================


def compute_l2_loss(params: np.ndarray, I: np.ndarray, T: np.ndarray, 
                    fixed_k: Optional[float] = None) -> float:
    """
    Compute ordinary least squares (L2) loss.
    
    Loss = Σ (T - T_pred)²
    
    Most stable convergence, sensitive to outliers.
    """
    if fixed_k is not None:
        B, P = params
        k = fixed_k
    else:
        B, P, k = params
    
    if B <= 0 or P <= 0 or k <= 0:
        return 1e10
    
    try:
        T_pred = roofline_model(I, B, P, k)
        residuals = T - T_pred
        return np.sum(residuals**2)
    except (ValueError, FloatingPointError, OverflowError):
        return 1e10


def compute_huber_loss(params: np.ndarray, I: np.ndarray, T: np.ndarray, 
                       delta: float, fixed_k: Optional[float] = None) -> float:
    """
    Compute Huber loss (already in fit_roofline.py but replicated here for completeness).
    
    ρ(r) = 0.5 * r²           if |r| ≤ δ
           δ(|r| - 0.5δ)      if |r| > δ
    
    Robust to outliers, quadratic for small residuals, linear for large.
    """
    if fixed_k is not None:
        B, P = params
        k = fixed_k
    else:
        B, P, k = params
    
    if B <= 0 or P <= 0 or k <= 0:
        return 1e10
    
    try:
        T_pred = roofline_model(I, B, P, k)
        residuals = T - T_pred
        
        abs_residuals = np.abs(residuals)
        huber_loss = np.where(
            abs_residuals <= delta,
            0.5 * residuals**2,
            delta * (abs_residuals - 0.5 * delta)
        )
        
        return np.sum(huber_loss)
    except (ValueError, FloatingPointError, OverflowError):
        return 1e10




def fit_roofline_irls(
    I: np.ndarray,
    T: np.ndarray,
    k_fixed: Optional[float] = None,
    initial_params: Optional[Dict[str, float]] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Fit roofline using IRLS (Iteratively Reweighted Least Squares).
    
    Robust to outliers without needing hyperparameter tuning (unlike Huber).
    Uses Huber weighting function but iteratively reweights.
    
    Args:
        I: Arithmetic intensity array (ops/byte)
        T: Measured throughput array (ops/sec)
        k_fixed: If provided, fix k and only optimize B and P
        initial_params: Optional dict with 'B', 'P', 'k' initial guesses
        max_iter: Maximum IRLS iterations (default 50)
        tol: Convergence tolerance (default 1e-6)
        bounds: Optional dict with parameter bounds
    
    Returns:
        Dictionary with fitted parameters and metrics
    """
    # Validate inputs
    I = np.asarray(I).flatten()
    T = np.asarray(T).flatten()
    
    if len(I) != len(T):
        raise ValueError(f"I and T must have same length: {len(I)} vs {len(T)}")
    
    if len(I) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(I)}")
    
    if np.any(I <= 0) or np.any(T <= 0):
        raise ValueError("I and T must be positive")
    
    # Set up parameter bounds
    if bounds is None:
        bounds_dict = {
            'B': (1e6, 1e15),   # Bandwidth: 1 MB/s to 1 PB/s
            'P': (1e9, 1e18),   # Compute: 1 GFLOP/s to 1 EFLOP/s
            'k': (1.0, 100.0)   # Sharpness: 1 to 100
        }
    else:
        bounds_dict = bounds
    
    # Initial fit with L2 loss
    if initial_params is None:
        # Simple initial guess
        B_init = np.median(T / I)  # Estimate from bandwidth-bound region
        P_init = np.median(T)      # Estimate from compute-bound region
        k_init = 100.0
        
        B_init = np.clip(B_init, bounds_dict['B'][0], bounds_dict['B'][1])
        P_init = np.clip(P_init, bounds_dict['P'][0], bounds_dict['P'][1])
        
        initial_params = {'B': B_init, 'P': P_init, 'k': k_init}
    
    # Initialize parameters
    if k_fixed is not None:
        params = np.array([initial_params['B'], initial_params['P']])
        param_bounds = [bounds_dict['B'], bounds_dict['P']]
        k = k_fixed
    else:
        params = np.array([initial_params['B'], initial_params['P'], initial_params['k']])
        param_bounds = [bounds_dict['B'], bounds_dict['P'], bounds_dict['k']]
        k = initial_params['k']
    
    # IRLS iterations
    for iteration in range(max_iter):
        # Current prediction
        if k_fixed is not None:
            B, P = params
            k_current = k_fixed
        else:
            B, P, k_current = params
        
        try:
            T_pred = roofline_model(I, B, P, k_current)
        except:
            return {
                'B': float(B),
                'P': float(P),
                'k': float(k_current),
                'metrics': {},
                'success': False,
                'message': 'Failed to evaluate roofline model',
                'iterations': iteration,
            }
        
        # Calculate residuals and weights
        residuals = T - T_pred
        
        # Robust scale estimate (MAD)
        mad = np.median(np.abs(residuals))
        scale = 1.4826 * mad  # Convert MAD to equivalent std dev
        
        if scale < 1e-10:
            scale = 1.0  # Avoid division by zero
        
        # Huber weighting function
        k_huber = 1.345 * scale
        weights = np.ones_like(residuals)
        outlier_mask = np.abs(residuals) > k_huber
        weights[outlier_mask] = k_huber / np.abs(residuals[outlier_mask])
        
        # Weighted least squares
        def weighted_objective(p):
            if k_fixed is not None:
                B_w, P_w = p
                k_w = k_fixed
            else:
                B_w, P_w, k_w = p
            
            if B_w <= 0 or P_w <= 0 or k_w <= 0:
                return 1e10
            
            try:
                T_pred_w = roofline_model(I, B_w, P_w, k_w)
                residuals_w = T - T_pred_w
                return np.sum(weights * residuals_w**2)
            except:
                return 1e10
        
        # Optimize with current weights
        result = minimize(
            weighted_objective,
            params,
            method='L-BFGS-B',
            bounds=param_bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        
        if not result.success:
            # Return current best estimate
            break
        
        params_new = result.x
        
        # Check convergence
        if np.allclose(params, params_new, atol=tol):
            params = params_new
            break
        
        params = params_new
    
    # Extract final parameters
    if k_fixed is not None:
        B_fit, P_fit = params
        k_fit = k_fixed
    else:
        B_fit, P_fit, k_fit = params
    
    # Compute final metrics
    try:
        T_pred = roofline_model(I, B_fit, P_fit, k_fit)
        metrics = compute_metrics(I, T, T_pred)
        
        return {
            'B': float(B_fit),
            'P': float(P_fit),
            'k': float(k_fit),
            'metrics': metrics,
            'success': True,
            'message': 'Converged',
            'iterations': iteration + 1,
        }
    except Exception as e:
        return {
            'B': float(B_fit),
            'P': float(P_fit),
            'k': float(k_fit),
            'metrics': {},
            'success': False,
            'message': f'Failed to compute metrics: {str(e)}',
            'iterations': iteration + 1,
        }


def fit_roofline_single_method(
    I: np.ndarray,
    T: np.ndarray,
    method: str = 'l2',
    k_fixed: Optional[float] = None,
    initial_params: Optional[Dict[str, float]] = None,
    huber_delta: float = 1.0,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """
    Fit roofline using a single specified method.
    
    Args:
        I: Arithmetic intensity array
        T: Throughput array
        method: 'l2', 'huber', or 'irls'
        k_fixed: Optional fixed k value
        initial_params: Optional initial parameter estimates
        huber_delta: Delta parameter for Huber loss (default 1.0)
        bounds: Optional parameter bounds
    
    Returns:
        Dictionary with fitted parameters and metrics
    """
    # Validate inputs
    I = np.asarray(I).flatten()
    T = np.asarray(T).flatten()
    
    if len(I) != len(T):
        raise ValueError(f"I and T must have same length: {len(I)} vs {len(T)}")
    
    if len(I) < 3:
        raise ValueError(f"Need at least 3 data points, got {len(I)}")
    
    # Handle IRLS separately (uses iterative approach)
    if method == 'irls':
        return fit_roofline_irls(I, T, k_fixed, initial_params, bounds=bounds)
    
    # For other methods, use scipy minimize
    
    # Estimate initial parameters if not provided
    if initial_params is None:
        # Sort by arithmetic intensity
        sort_idx = np.argsort(I)
        I_sorted = I[sort_idx]
        T_sorted = T[sort_idx]
        
        n = len(I)
        
        # Estimate B from low-I region
        n_bw = max(3, n // 3)
        B_init = np.median(T_sorted[:n_bw] / I_sorted[:n_bw])
        
        # Estimate P from high-I region
        P_init = np.median(T_sorted[-n // 3:])
        
        k_init = 100.0
        
        B_init = np.clip(B_init, 1e6, 1e15)
        P_init = np.clip(P_init, 1e9, 1e18)
        
        initial_params = {'B': B_init, 'P': P_init, 'k': k_init}
    
    # Set up parameter bounds
    if bounds is None:
        bounds_dict = {
            'B': (1e6, 1e15),
            'P': (1e9, 1e18),
            'k': (1.0, 100.0)
        }
    else:
        bounds_dict = bounds
    
    # Select loss function
    if method == 'l2':
        loss_fn = compute_l2_loss
    elif method == 'huber':
        loss_fn = lambda p, I, T, fk: compute_huber_loss(p, I, T, huber_delta, fk)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'l2', 'huber', or 'irls'")
    
    # Set up optimization
    if k_fixed is not None:
        x0 = [initial_params['B'], initial_params['P']]
        param_bounds = [bounds_dict['B'], bounds_dict['P']]
        
        def objective(params):
            return loss_fn(params, I, T, k_fixed)
    else:
        x0 = [initial_params['B'], initial_params['P'], initial_params['k']]
        param_bounds = [bounds_dict['B'], bounds_dict['P'], bounds_dict['k']]
        
        def objective(params):
            return loss_fn(params, I, T, None)
    
    # Run optimization
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=param_bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-9}
    )
    
    # Extract fitted parameters
    if k_fixed is not None:
        B_fit, P_fit = result.x
        k_fit = k_fixed
    else:
        B_fit, P_fit, k_fit = result.x
    
    # Compute predicted values and metrics
    try:
        T_pred = roofline_model(I, B_fit, P_fit, k_fit)
        metrics = compute_metrics(I, T, T_pred)
        
        return {
            'B': float(B_fit),
            'P': float(P_fit),
            'k': float(k_fit),
            'metrics': metrics,
            'loss': float(result.fun),
            'success': bool(result.success),
            'message': result.message,
            'nit': int(result.nit) if hasattr(result, 'nit') else None,
            'nfev': int(result.nfev) if hasattr(result, 'nfev') else None
        }
    except Exception as e:
        return {
            'B': float(B_fit),
            'P': float(P_fit),
            'k': float(k_fit),
            'metrics': {},
            'success': False,
            'message': f"Failed to compute metrics: {str(e)}",
        }


def fit_roofline_all_methods(
    I: np.ndarray,
    T: np.ndarray,
    k_fixed: Optional[float] = None,
    huber_delta: float = 1.0,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Fit roofline using all available methods and return results for comparison.
    
    Similar to the networking profiler's approach - apply multiple methods
    and let the user choose based on convergence and performance.
    
    Args:
        I: Arithmetic intensity array (ops/byte)
        T: Throughput array (ops/sec)
        k_fixed: Optional fixed k value (default None)
        huber_delta: Delta for Huber loss (default 1.0)
        bounds: Optional parameter bounds
    
    Returns:
        Dictionary mapping method name to results:
        {
            'l2': {...},
            'huber': {...},
            'irls': {...}
        }
    
    Each result contains:
        - B: Bandwidth (bytes/sec)
        - P: Peak FLOPS (ops/sec)
        - k: Sharpness parameter
        - metrics: Dict with r_squared, rmse, mae, etc.
        - success: Boolean
        - message: Status message
    """
    methods = ['l2', 'huber', 'irls']
    results = {}
    
    for method in methods:
        try:
            result = fit_roofline_single_method(
                I, T,
                method=method,
                k_fixed=k_fixed,
                huber_delta=huber_delta,
                bounds=bounds
            )
            results[method] = result
        except Exception as e:
            results[method] = {
                'B': np.nan,
                'P': np.nan,
                'k': k_fixed if k_fixed is not None else np.nan,
                'metrics': {},
                'success': False,
                'message': f"Exception: {str(e)}"
            }
    
    return results


if __name__ == "__main__":
    """Test all methods on synthetic data."""
    print("=" * 80)
    print("Testing All Roofline Fitting Methods")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    I_true = np.logspace(-2, 3, 100)
    B_true = 1e11   # 100 GB/s
    P_true = 1e13   # 10 TFLOP/s
    k_true = 100.0
    
    T_true = roofline_model(I_true, B_true, P_true, k_true)
    
    # Add noise
    T_noisy = T_true * (1 + 0.05 * np.random.randn(len(T_true)))
    
    # Add some outliers
    n_outliers = 10
    outlier_idx = np.random.choice(len(T_noisy), n_outliers, replace=False)
    T_noisy[outlier_idx] *= (1 + 0.3 * np.random.randn(n_outliers))
    
    print(f"\nGenerated {len(I_true)} samples with 5% noise + {n_outliers} outliers")
    print(f"True parameters: B={B_true:.2e} ({B_true/1e9:.1f} GB/s), "
          f"P={P_true:.2e} ({P_true/1e12:.1f} TFLOPS), k={k_true:.1f}")
    
    # Fit with all 3 methods
    print("\nFitting with all 3 methods (L2, Huber, IRLS)...")
    results = fit_roofline_all_methods(I_true, T_noisy, k_fixed=100.0)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON OF ALL 3 METHODS")
    print("=" * 80)
    print(f"\n{'Method':<10} {'Success':<10} {'BW (GB/s)':<12} {'Peak (TFLOPS)':<15} {'R²':<10} {'BW Error':<12} {'Peak Error':<12}")
    print("-" * 80)
    
    for method, result in results.items():
        if result['success']:
            bw_gbps = result['B'] / 1e9
            peak_tflops = result['P'] / 1e12
            r2 = result['metrics']['r_squared']
            bw_error = abs(result['B'] - B_true) / B_true * 100
            peak_error = abs(result['P'] - P_true) / P_true * 100
            
            print(f"{method:<10} {'✓':<10} {bw_gbps:>10.2f}  {peak_tflops:>13.2f}  "
                  f"{r2:>8.6f}  {bw_error:>10.1f}%  {peak_error:>10.1f}%")
        else:
            print(f"{method:<10} {'✗':<10} FAILED - {result.get('message', 'unknown')}")
    
    print("\n" + "=" * 80)

