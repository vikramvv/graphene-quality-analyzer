import numpy as np
from typing import Dict, Optional


def calculate_quality_metrics(fits: Dict[str, Optional[Dict]]) -> Dict[str, Optional[float]]:
    """
    Calculate quality metrics from peak fits.
    
    Args:
        fits: Dictionary of peak fit results (D, G, 2D)
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        # Peak positions
        'd_position': None,
        'g_position': None,
        '2d_position': None,
        
        # Peak intensities
        'd_intensity': None,
        'g_intensity': None,
        '2d_intensity': None,
        
        # Peak widths (FWHM)
        'fwhm_d': None,
        'fwhm_g': None,
        'fwhm_2d': None,
        
        # Intensity ratios
        'id_ig_ratio': None,
        'i2d_ig_ratio': None,
        
        # Additional metrics
        'g_2d_separation': None,
    }
    
    # Extract D peak metrics
    if fits.get('D') is not None:
        d_fit = fits['D']
        metrics['d_position'] = d_fit['position']
        metrics['d_intensity'] = d_fit['amplitude']
        metrics['fwhm_d'] = d_fit['fwhm']
    
    # Extract G peak metrics
    if fits.get('G') is not None:
        g_fit = fits['G']
        metrics['g_position'] = g_fit['position']
        metrics['g_intensity'] = g_fit['amplitude']
        metrics['fwhm_g'] = g_fit['fwhm']
    
    # Extract 2D peak metrics
    if fits.get('2D') is not None:
        td_fit = fits['2D']
        metrics['2d_position'] = td_fit['position']
        metrics['2d_intensity'] = td_fit['amplitude']
        metrics['fwhm_2d'] = td_fit['fwhm']
    
    # Calculate intensity ratios
    if metrics['g_intensity'] is not None and metrics['g_intensity'] > 0:
        if metrics['d_intensity'] is not None:
            metrics['id_ig_ratio'] = metrics['d_intensity'] / metrics['g_intensity']
        
        if metrics['2d_intensity'] is not None:
            metrics['i2d_ig_ratio'] = metrics['2d_intensity'] / metrics['g_intensity']
    
    # Calculate G-2D separation
    if metrics['g_position'] is not None and metrics['2d_position'] is not None:
        metrics['g_2d_separation'] = metrics['2d_position'] - metrics['g_position']
    
    return metrics


def interpret_quality(metrics: Dict[str, Optional[float]], 
                     thresholds: Dict[str, float]) -> Dict[str, str]:
    """
    Interpret quality metrics to provide qualitative assessment.
    
    Args:
        metrics: Calculated quality metrics
        thresholds: Quality threshold values
        
    Returns:
        Dictionary with quality interpretations
    """
    quality = {
        'overall': 'Unknown',
        'layer_type': 'Unknown',
        'defect_level': 'Unknown',
        'notes': []
    }
    
    # Determine layer type from I(2D)/I(G) and 2D FWHM
    if metrics['i2d_ig_ratio'] is not None and metrics['fwhm_2d'] is not None:
        if (metrics['i2d_ig_ratio'] > thresholds['i2d_ig_single_layer'] and 
            metrics['fwhm_2d'] < thresholds['fwhm_2d_single']):
            quality['layer_type'] = 'Single Layer'
        elif (metrics['i2d_ig_ratio'] > thresholds['i2d_ig_few_layer'] and 
              metrics['fwhm_2d'] < thresholds['fwhm_2d_few']):
            quality['layer_type'] = 'Few Layers (2-5)'
        else:
            quality['layer_type'] = 'Multi-Layer (>5)'
    
    # Determine defect level from I(D)/I(G)
    if metrics['id_ig_ratio'] is not None:
        if metrics['id_ig_ratio'] < thresholds['id_ig_excellent']:
            quality['defect_level'] = 'Very Low (Excellent)'
        elif metrics['id_ig_ratio'] < thresholds['id_ig_good']:
            quality['defect_level'] = 'Low (Good)'
        elif metrics['id_ig_ratio'] < 1.0:
            quality['defect_level'] = 'Moderate'
        else:
            quality['defect_level'] = 'High'
    
    # Overall quality assessment
    scores = []
    
    # Score based on defects
    if metrics['id_ig_ratio'] is not None:
        if metrics['id_ig_ratio'] < thresholds['id_ig_excellent']:
            scores.append(5)
        elif metrics['id_ig_ratio'] < thresholds['id_ig_good']:
            scores.append(4)
        elif metrics['id_ig_ratio'] < 1.0:
            scores.append(3)
        else:
            scores.append(2)
    
    # Score based on layer quality
    if quality['layer_type'] == 'Single Layer':
        scores.append(5)
    elif quality['layer_type'] == 'Few Layers (2-5)':
        scores.append(4)
    else:
        scores.append(3)
    
    # Score based on 2D peak sharpness
    if metrics['fwhm_2d'] is not None:
        if metrics['fwhm_2d'] < thresholds['fwhm_2d_single']:
            scores.append(5)
        elif metrics['fwhm_2d'] < thresholds['fwhm_2d_few']:
            scores.append(4)
        else:
            scores.append(3)
    
    # Calculate overall score
    if scores:
        avg_score = np.mean(scores)
        if avg_score >= 4.5:
            quality['overall'] = 'Excellent'
        elif avg_score >= 3.5:
            quality['overall'] = 'Good'
        elif avg_score >= 2.5:
            quality['overall'] = 'Fair'
        else:
            quality['overall'] = 'Poor'
    
    # Add interpretive notes
    if metrics['g_position'] is not None:
        if metrics['g_position'] > 1590:
            quality['notes'].append('G peak upshifted - possible tensile strain or doping')
        elif metrics['g_position'] < 1575:
            quality['notes'].append('G peak downshifted - possible compressive strain')
    
    if metrics['2d_position'] is not None:
        if metrics['2d_position'] > 2720:
            quality['notes'].append('2D peak upshifted - possible tensile strain')
        elif metrics['2d_position'] < 2680:
            quality['notes'].append('2D peak downshifted - possible compressive strain or doping')
    
    if metrics['i2d_ig_ratio'] is not None and metrics['i2d_ig_ratio'] < 0.5:
        quality['notes'].append('Very low I(2D)/I(G) - heavily multi-layered or significant disorder')
    
    if metrics['id_ig_ratio'] is not None and metrics['id_ig_ratio'] > 2.0:
        quality['notes'].append('Very high I(D)/I(G) - significant structural defects or amorphization')
    
    return quality


def calculate_crystallite_size(id_ig_ratio: float, laser_wavelength: float = 532) -> Optional[float]:
    """
    Estimate crystallite size from I(D)/I(G) ratio using Tuinstra-Koenig relation.
    
    Args:
        id_ig_ratio: I(D)/I(G) intensity ratio
        laser_wavelength: Laser excitation wavelength in nm (default 532 nm)
        
    Returns:
        Crystallite size in nm, or None if ratio is invalid
    """
    if id_ig_ratio <= 0:
        return None
    
    # Tuinstra-Koenig relation: La (nm) = C(λ) / (I(D)/I(G))
    # C(532 nm) ≈ 4.4 nm
    C_lambda = 4.4 * (514.5 / laser_wavelength)**4  # Scale for different wavelengths
    
    La = C_lambda / id_ig_ratio
    
    return La


def estimate_defect_density(id_ig_ratio: float) -> Optional[float]:
    """
    Estimate defect density from I(D)/I(G) ratio.
    
    Args:
        id_ig_ratio: I(D)/I(G) intensity ratio
        
    Returns:
        Defect density in 10^10 cm^-2, or None if ratio is invalid
    """
    if id_ig_ratio is None or id_ig_ratio <= 0:
        return None
    
    # Empirical relation for low defect regime
    # n_D ≈ (1.8 ± 0.5) × 10^22 / λ_L^4 × (I_D/I_G)
    # For 532 nm laser
    n_D = 1.8e22 / (532**4) * id_ig_ratio * 1e-10  # Convert to 10^10 cm^-2
    
    return n_D