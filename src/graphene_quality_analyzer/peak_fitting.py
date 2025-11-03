import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Optional, Tuple


def lorentzian(x: np.ndarray, amplitude: float, center: float, width: float, offset: float = 0) -> np.ndarray:
    """
    Lorentzian peak function.
    
    Args:
        x: Independent variable
        amplitude: Peak amplitude
        center: Peak center position
        width: Peak width (FWHM)
        offset: Baseline offset
        
    Returns:
        Lorentzian function values
    """
    return offset + amplitude * (width**2 / ((x - center)**2 + width**2))


def voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, gamma: float, offset: float = 0) -> np.ndarray:
    """
    Voigt profile (convolution of Gaussian and Lorentzian).
    
    Args:
        x: Independent variable
        amplitude: Peak amplitude
        center: Peak center
        sigma: Gaussian width
        gamma: Lorentzian width
        offset: Baseline offset
        
    Returns:
        Voigt function values
    """
    from scipy.special import wofz
    
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    profile = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    return offset + profile


def fit_peak(wavelength: np.ndarray,
             intensity: np.ndarray,
             peak_idx: int,
             model: str = 'lorentzian') -> Optional[Dict]:
    """
    Fit a single peak with specified model.

    Args:
        wavelength: Wavelength array for fitting window
        intensity: Intensity array for fitting window
        peak_idx: Index of peak center (relative to fitting window)
        model: 'lorentzian' or 'voigt'

    Returns:
        Dictionary containing fit parameters and quality metrics, or None if fit fails
    """
    try:
        # Initial parameter guesses
        amplitude_guess = intensity[peak_idx]
        center_guess = wavelength[peak_idx]

        # Estimate width from half-maximum
        half_max = amplitude_guess / 2
        left_idx = peak_idx
        while left_idx > 0 and intensity[left_idx] > half_max:
            left_idx -= 1
        right_idx = peak_idx
        while right_idx < len(intensity) - 1 and intensity[right_idx] > half_max:
            right_idx += 1

        width_guess = (wavelength[right_idx] - wavelength[left_idx]) / 2
        if width_guess <= 0 or width_guess > 200:
            width_guess = 20.0

        offset_guess = np.percentile(intensity, 10)

        # Robust bounds
        amplitude_upper = max(amplitude_guess * 2, amplitude_guess + 10, 10)
        width_lower = max(width_guess, 5)
        width_upper = max(width_guess * 2, width_guess + 20, 20)
        offset_upper = max(offset_guess * 2, offset_guess + 10, 10)

        if model == 'lorentzian':
            p0 = [amplitude_guess, center_guess, width_guess, offset_guess]
            bounds = (
                [0, center_guess - 50, width_lower, 0],
                [amplitude_upper, center_guess + 50, width_upper, offset_upper]
            )
            print(f"Fitting Lorentzian: p0={p0}, bounds={bounds}")

            popt, pcov = curve_fit(
                lorentzian,
                wavelength,
                intensity,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )

            amplitude, center, width, offset = popt
            fitted_curve = lorentzian(wavelength, *popt)
            fwhm = 2 * width
        elif model == 'voigt':
            sigma_lower = 1
            sigma_upper = max(width_guess, 10)
            gamma_lower = 1
            gamma_upper = max(width_guess, 10)
            p0 = [amplitude_guess, center_guess, width_guess/2, width_guess/2, offset_guess]
            bounds = (
                [0, center_guess - 50, sigma_lower, gamma_lower, 0],
                [amplitude_upper, center_guess + 50, sigma_upper, gamma_upper, offset_upper]
            )
            print(f"Fitting Voigt: p0={p0}, bounds={bounds}")

            popt, pcov = curve_fit(
                voigt,
                wavelength,
                intensity,
                p0=p0,
                bounds=bounds,
                maxfev=5000
            )

            amplitude, center, sigma, gamma, offset = popt
            fitted_curve = voigt(wavelength, *popt)
            fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma)**2 + (2.355 * sigma)**2)
        else:
            raise ValueError(f"Unknown model: {model}")

        # Calculate R-squared
        residuals = intensity - fitted_curve
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((intensity - np.mean(intensity))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Extract peak parameters
        result = {
            'model': model,
            'position': center,
            'amplitude': amplitude,
            'fwhm': fwhm,
            'offset': offset,
            'fitted_curve': fitted_curve,
            'wavelength': wavelength,
            'intensity': intensity,
            'r_squared': r_squared,
            'parameters': popt,
            'covariance': pcov
        }

        return result

    except Exception as e:
        print(f"Peak fitting failed: {str(e)}")
        return None

def fit_all_peaks(wavelength: np.ndarray,
                 intensity: np.ndarray,
                 peak_regions: Dict[str, Optional[Tuple[int, int, int]]],
                 model: str = 'lorentzian') -> Dict[str, Optional[Dict]]:
    """
    Fit all detected peaks (D, G, 2D).
    
    Args:
        wavelength: Full wavelength array
        intensity: Full intensity array (baseline corrected)
        peak_regions: Dictionary of peak regions from refine_peak_regions
        model: Fitting model ('lorentzian' or 'voigt')
        
    Returns:
        Dictionary mapping peak names to fit results
    """
    fits = {}
    
    for peak_name, region in peak_regions.items():
        if region is None:
            fits[peak_name] = None
            continue
        
        start_idx, end_idx, peak_idx = region
        
        # Extract fitting window
        wave_window = wavelength[start_idx:end_idx]
        int_window = intensity[start_idx:end_idx]
        
        # Adjust peak_idx to be relative to window
        peak_idx_rel = peak_idx - start_idx
        
        # Fit the peak
        fit_result = fit_peak(wave_window, int_window, peak_idx_rel, model=model)
        
        fits[peak_name] = fit_result
    
    return fits


def refit_peak_manual(wavelength: np.ndarray,
                     intensity: np.ndarray,
                     center_range: Tuple[float, float],
                     model: str = 'lorentzian') -> Optional[Dict]:
    """
    Manually refit a peak with user-specified center range.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        center_range: (min, max) range for peak center
        model: Fitting model
        
    Returns:
        Fit result dictionary or None
    """
    # Find indices in the specified range
    mask = (wavelength >= center_range[0]) & (wavelength <= center_range[1])
    
    if not np.any(mask):
        return None
    
    wave_window = wavelength[mask]
    int_window = intensity[mask]
    
    # Find peak maximum in window
    peak_idx = np.argmax(int_window)
    
    return fit_peak(wave_window, int_window, peak_idx, model=model)