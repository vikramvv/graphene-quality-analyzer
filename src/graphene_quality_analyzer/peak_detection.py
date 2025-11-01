import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple


# Expected peak positions (cm-1) with wider windows
EXPECTED_PEAKS = {
    'D': (1250, 1450),   # D peak range (±100 from 1350)
    'G': (1480, 1680),   # G peak range (±100 from 1580)
    '2D': (2500, 2900)   # 2D peak range (±200 from 2700)
}


def detect_peaks(wavelength: np.ndarray,
                intensity: np.ndarray,
                prominence: float = 0.1,
                distance: int = 100) -> np.ndarray:
    """
    Detect peaks in the spectrum.
    
    Args:
        wavelength: Wavelength/Raman shift array
        intensity: Intensity array (baseline corrected)
        prominence: Minimum prominence for peak detection (relative to max intensity)
        distance: Minimum distance between peaks in number of points
        
    Returns:
        Array of peak indices
    """
    # Normalize intensity for prominence calculation
    normalized_intensity = intensity / intensity.max()
    
    # Find peaks
    peaks, properties = find_peaks(
        normalized_intensity,
        prominence=prominence,
        distance=distance
    )
    
    return peaks


def refine_peak_regions(wavelength: np.ndarray, 
                       peak_indices: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """
    Identify which peaks correspond to D, G, and 2D peaks.
    
    Args:
        wavelength: Wavelength/Raman shift array
        peak_indices: Indices of detected peaks
        
    Returns:
        Dictionary mapping peak names to (start_idx, end_idx) for fitting window
    """
    peak_regions = {}
    peak_positions = wavelength[peak_indices]
    
    for peak_name, (min_pos, max_pos) in EXPECTED_PEAKS.items():
        # Find peaks in the expected range
        in_range = (peak_positions >= min_pos) & (peak_positions <= max_pos)
        
        if np.any(in_range):
            # Get the strongest peak in this range
            candidates = peak_indices[in_range]
            peak_idx = candidates[0]  # Take first (should be strongest after filtering)
            
            # Define fitting window around the peak
            window_size = 100  # points around peak
            start_idx = max(0, peak_idx - window_size)
            end_idx = min(len(wavelength), peak_idx + window_size)
            
            peak_regions[peak_name] = (start_idx, end_idx, peak_idx)
        else:
            peak_regions[peak_name] = None
    
    return peak_regions


def estimate_peak_width(wavelength: np.ndarray,
                       intensity: np.ndarray,
                       peak_idx: int) -> float:
    """
    Estimate FWHM of a peak.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        peak_idx: Index of peak maximum
        
    Returns:
        Estimated FWHM in wavelength units
    """
    peak_intensity = intensity[peak_idx]
    half_max = peak_intensity / 2
    
    # Find left half-maximum
    left_idx = peak_idx
    while left_idx > 0 and intensity[left_idx] > half_max:
        left_idx -= 1
    
    # Find right half-maximum
    right_idx = peak_idx
    while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
        right_idx += 1
    
    # Calculate FWHM
    if left_idx < peak_idx < right_idx:
        fwhm = wavelength[right_idx] - wavelength[left_idx]
    else:
        # Fallback to typical value
        fwhm = 30.0
    
    return fwhm


def auto_adjust_detection_params(intensity: np.ndarray) -> Dict[str, float]:
    """
    Automatically adjust peak detection parameters based on spectrum characteristics.
    
    Args:
        intensity: Intensity array
        
    Returns:
        Dictionary of suggested parameters
    """
    # Calculate signal-to-noise ratio
    noise_std = np.std(intensity[:100])  # Estimate from beginning
    signal_max = np.max(intensity)
    snr = signal_max / noise_std if noise_std > 0 else 100
    
    # Adjust prominence based on SNR
    if snr > 50:
        prominence = 0.05
    elif snr > 20:
        prominence = 0.1
    else:
        prominence = 0.15
    
    return {
        'prominence': prominence,
        'distance': 100
    }