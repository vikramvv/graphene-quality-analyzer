import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple


def baseline_correction(wavelength: np.ndarray, 
                       intensity: np.ndarray,
                       poly_order: int = 2,
                       method: str = 'polynomial') -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform baseline correction on Raman spectrum.
    
    Args:
        wavelength: Wavelength/Raman shift array
        intensity: Intensity array
        poly_order: Order of polynomial for fitting
        method: 'polynomial' or 'als' (asymmetric least squares)
        
    Returns:
        Tuple of (corrected_intensity, baseline)
    """
    if method == 'polynomial':
        return _polynomial_baseline(wavelength, intensity, poly_order)
    elif method == 'als':
        return _als_baseline(intensity)
    else:
        raise ValueError(f"Unknown method: {method}")


def _polynomial_baseline(wavelength: np.ndarray, 
                        intensity: np.ndarray,
                        poly_order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Polynomial baseline correction."""
    # Fit polynomial to the data
    coeffs = np.polyfit(wavelength, intensity, poly_order)
    baseline = np.polyval(coeffs, wavelength)
    
    # Ensure baseline doesn't exceed the data
    baseline = np.minimum(baseline, intensity)
    
    # Subtract baseline
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)  # No negative values
    
    return corrected, baseline


def _als_baseline(intensity: np.ndarray, 
                  lam: float = 1e6, 
                  p: float = 0.01,
                  niter: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asymmetric Least Squares baseline correction.
    
    Args:
        intensity: Intensity array
        lam: Smoothness parameter (larger = smoother)
        p: Asymmetry parameter (0 < p < 1, typically 0.001-0.1)
        niter: Number of iterations
        
    Returns:
        Tuple of (corrected_intensity, baseline)
    """
    L = len(intensity)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    
    for _ in range(niter):
        W.setdiag(w)
        Z = W + D
        baseline = spsolve(Z, w * intensity)
        w = p * (intensity > baseline) + (1 - p) * (intensity < baseline)
    
    corrected = intensity - baseline
    corrected = np.maximum(corrected, 0)
    
    return corrected, baseline


def smooth_spectrum(intensity: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to spectrum.
    
    Args:
        intensity: Intensity array
        window_size: Size of smoothing window (must be odd)
        
    Returns:
        Smoothed intensity array
    """
    from scipy.signal import savgol_filter
    
    if window_size % 2 == 0:
        window_size += 1
    
    return savgol_filter(intensity, window_size, 2)