import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional


# Expected peak positions (cm-1) with wider windows
EXPECTED_PEAKS = {
    'D': (1250, 1450),   # D peak range (±100 from 1350)
    'G': (1480, 1680),   # G peak range (±100 from 1580)
    '2D': (2500, 2900)   # 2D peak range (±200 from 2700)
}


# --- helpers ---

def _cm_to_samples(dx_cm1: float, wavelength: np.ndarray) -> int:
    """Convert cm-1 distance to number of samples."""
    step = float(np.median(np.diff(wavelength)))
    if step == 0 or not np.isfinite(step):
        step = 1.0
    return max(1, int(round(abs(dx_cm1) / abs(step))))


def _robust_noise(y: np.ndarray, frac_bg: float = 0.30):
    """Return (sigma, bg_median) estimated from the lowest-intensity fraction."""
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return 1.0, 0.0
    k = max(50, int(frac_bg * n))
    idx = np.argpartition(y, k)[:k]
    bg = y[idx]
    med = float(np.median(bg))
    mad = float(np.median(np.abs(bg - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(bg))
    return max(sigma, 1e-12), med


# --- main API ---

def detect_peaks(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    prominence: Optional[float] = None,   # absolute prominence (intensity units); if None, compute from noise
    distance: Optional[int] = None,       # points; if None compute from min_distance_cm1
    min_distance_cm1: float = 60.0,       # physical separation between peaks
    min_prom_sigmas: float = 7.0,         # if prominence is None, use X * noise_sigma
    min_height_sigmas: float = 3.5        # minimum height above background
) -> np.ndarray:
    """
    Detect peaks with absolute thresholds tied to the noise (not normalized to global max).
    
    Args:
        wavelength: Wavelength/Raman shift array
        intensity: Intensity array (baseline corrected)
        prominence: Absolute prominence in intensity units (if None, auto-calculated)
        distance: Minimum distance between peaks in points (if None, calculated from min_distance_cm1)
        min_distance_cm1: Minimum physical separation between peaks in cm-1
        min_prom_sigmas: Number of sigmas above noise for prominence threshold
        min_height_sigmas: Number of sigmas above background for height threshold
        
    Returns:
        Array of peak indices
    """
    # Ensure x is sorted to keep distance meaningful
    order = np.argsort(wavelength)
    x = np.asarray(wavelength, float)[order]
    y = np.asarray(intensity, float)[order]

    # Robust noise & background
    sigma, bg_med = _robust_noise(y)

    # Thresholds
    prom_abs = float(prominence) if prominence is not None else (min_prom_sigmas * sigma)
    height_abs = bg_med + min_height_sigmas * sigma
    dist_pts = int(distance) if distance is not None else _cm_to_samples(min_distance_cm1, x)

    # DEBUG: print thresholds actually used
    print(f"[detect_peaks] step≈{np.median(np.diff(x)):.3f} cm^-1/pt | distance={dist_pts} pts "
          f"(~{dist_pts*np.median(np.diff(x)):.1f} cm^-1), height>={height_abs:.3f}, prominence>={prom_abs:.3f}")

    # Find peaks on absolute signal (no normalization)
    peaks, props = find_peaks(y, prominence=prom_abs, distance=dist_pts, height=height_abs)

    print(f"[detect_peaks] raw peaks found: {len(peaks)}")
    if len(peaks) > 0:
        print(f"[detect_peaks] peak positions: {x[peaks]}")
    
    return peaks


def refine_peak_regions(
    wavelength: np.ndarray,
    peak_indices: np.ndarray,
    intensity: Optional[np.ndarray] = None,
    window_half_width_cm1: float = 80.0,
    force_detection: bool = True
) -> Dict[str, Optional[Tuple[int, int, int]]]:
    """
    Map detected peaks to D/G/2D. Use strongest (by intensity) candidate inside each expected window.
    If force_detection=True, will search for local maxima in expected regions even if not in peak_indices.
    
    Args:
        wavelength: Wavelength array
        peak_indices: Indices of detected peaks
        intensity: Intensity array (for force detection)
        window_half_width_cm1: Half-width of fitting window around peak
        force_detection: If True, search for local maxima even if peak not detected
        
    Returns:
        Dictionary mapping peak names to (start_idx, end_idx, peak_idx) or None
    """
    peak_regions: Dict[str, Optional[Tuple[int, int, int]]] = {}
    if peak_indices is None:
        peak_indices = np.array([], dtype=int)

    x = np.asarray(wavelength, float)
    y = np.asarray(intensity, float) if intensity is not None else np.ones(len(x))
    
    strengths = y[peak_indices] if len(peak_indices) > 0 else np.array([])
    pos = x[peak_indices] if len(peak_indices) > 0 else np.array([])
    half_pts = _cm_to_samples(window_half_width_cm1, x)

    for name, (lo, hi) in EXPECTED_PEAKS.items():
        # First try to find in detected peaks
        in_range = (pos >= lo) & (pos <= hi)
        
        if np.any(in_range):
            cand_idx = np.where(in_range)[0]
            best = cand_idx[np.argmax(strengths[in_range])]
            pk = int(peak_indices[best])
            
            start = max(0, pk - half_pts)
            end = min(len(x), pk + half_pts)
            peak_regions[name] = (start, end, pk)
            print(f"[refine] {name} peak found in detected peaks at {x[pk]:.1f} cm⁻¹ (intensity: {y[pk]:.1f})")
            
        elif force_detection:
            # Force detection: find strongest local maximum in expected region
            mask = (x >= lo) & (x <= hi)
            if np.any(mask):
                indices = np.where(mask)[0]
                y_region = y[mask]
                
                # Find local maximum in this region
                if len(y_region) > 5:
                    # Look for the highest point
                    local_max_rel = np.argmax(y_region)
                    pk = indices[local_max_rel]
                    
                    # Check if it's actually a peak (higher than neighbors)
                    if pk > 0 and pk < len(y) - 1:
                        if y[pk] > y[pk-1] and y[pk] > y[pk+1]:
                            start = max(0, pk - half_pts)
                            end = min(len(x), pk + half_pts)
                            peak_regions[name] = (start, end, pk)
                            print(f"[refine] {name} peak FORCED at {x[pk]:.1f} cm⁻¹ (intensity: {y[pk]:.1f})")
                            continue
            
            peak_regions[name] = None
            print(f"[refine] {name} peak NOT FOUND in range {lo}-{hi} cm⁻¹")
        else:
            peak_regions[name] = None

    return peak_regions


def estimate_peak_width(wavelength: np.ndarray,
                       intensity: np.ndarray,
                       peak_idx: int) -> float:
    """
    Estimate FWHM of a peak using linear interpolation at half-maximum.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        peak_idx: Index of peak maximum
        
    Returns:
        Estimated FWHM in wavelength units
    """
    x = np.asarray(wavelength, float)
    y = np.asarray(intensity, float)

    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        return 30.0

    half = 0.5 * y[peak_idx]

    # Find left half-maximum
    li = peak_idx
    while li > 0 and y[li] > half:
        li -= 1
    
    # Find right half-maximum
    ri = peak_idx
    while ri < len(y) - 1 and y[ri] > half:
        ri += 1

    if not (li < peak_idx < ri):
        return 30.0

    def _interp_x(i1, i2):
        y1, y2 = y[i1], y[i2]
        x1, x2 = x[i1], x[i2]
        if y2 == y1:
            return float(x1)
        t = (half - y1) / (y2 - y1)
        return float(x1 + t * (x2 - x1))

    x_left = _interp_x(li, li + 1)
    x_right = _interp_x(ri - 1, ri)
    
    return abs(x_right - x_left)


def auto_adjust_detection_params(intensity: np.ndarray) -> Dict[str, float]:
    """
    Automatically adjust peak detection parameters based on spectrum characteristics.
    
    Args:
        intensity: Intensity array
        
    Returns:
        Dictionary of suggested parameters
    """
    noise_std = np.std(intensity[:100]) if len(intensity) >= 100 else np.std(intensity)
    signal_max = np.max(intensity) if len(intensity) else 0.0
    snr = signal_max / noise_std if noise_std > 0 else 100.0

    if snr > 50:
        prominence = 0.05
    elif snr > 20:
        prominence = 0.1
    else:
        prominence = 0.15

    return {'prominence': float(prominence), 'distance': 100.0}