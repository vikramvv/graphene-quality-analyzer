import pandas as pd
import numpy as np
from typing import Dict, Tuple


def load_excel_data(file) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load Raman data from Excel file.
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        Dictionary mapping sheet names to (wavelength, intensity) tuples
    """
    excel_file = pd.ExcelFile(file)
    data_dict = {}
    
    for sheet_name in excel_file.sheet_names:
        # Read the sheet - try with header first
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Extract first two columns (A and B)
        if df.shape[1] < 2:
            raise ValueError(f"Sheet '{sheet_name}' must have at least 2 columns")
        
        # Get first two columns regardless of their names
        wavelength_col = df.iloc[:, 0]
        intensity_col = df.iloc[:, 1]
        
        # Try to convert to numeric, coercing errors
        wavelength = pd.to_numeric(wavelength_col, errors='coerce').values
        intensity = pd.to_numeric(intensity_col, errors='coerce').values
        
        # Remove NaN values (from headers or bad data)
        valid_mask = ~(np.isnan(wavelength) | np.isnan(intensity))
        wavelength = wavelength[valid_mask]
        intensity = intensity[valid_mask]
        
        # Check if we have any valid data
        if len(wavelength) == 0:
            raise ValueError(f"Sheet '{sheet_name}' has no valid numeric data in first two columns")
        
        # Sort by wavelength
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        intensity = intensity[sort_idx]
        
        data_dict[sheet_name] = (wavelength, intensity)
    
    return data_dict


def validate_raman_data(wavelength: np.ndarray, intensity: np.ndarray) -> bool:
    """
    Validate that the data looks like Raman spectroscopy data.
    
    Args:
        wavelength: Wavelength/Raman shift array
        intensity: Intensity array
        
    Returns:
        True if data appears valid
    """
    # Check for minimum length
    if len(wavelength) < 100:
        return False
    
    # Check that wavelength is monotonically increasing
    if not np.all(np.diff(wavelength) > 0):
        return False
    
    # Check reasonable range for Raman shift (typically 500-3500 cm-1)
    if wavelength.min() < 0 or wavelength.max() > 5000:
        return False
    
    # Check intensity is non-negative
    if np.any(intensity < 0):
        return False
    
    return True