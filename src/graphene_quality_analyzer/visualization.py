import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional


def plot_spectrum_with_peaks(data: Dict) -> go.Figure:
    """
    Plot full spectrum with detected peaks and fits.
    
    Args:
        data: Dictionary containing analysis results
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Plot raw spectrum
    fig.add_trace(go.Scatter(
        x=data['wavelength'],
        y=data['intensity_raw'],
        name='Raw Spectrum',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    # Plot baseline
    fig.add_trace(go.Scatter(
        x=data['wavelength'],
        y=data['baseline'],
        name='Baseline',
        line=dict(color='orange', width=1, dash='dash'),
        opacity=0.7
    ))
    
    # Plot corrected spectrum
    fig.add_trace(go.Scatter(
        x=data['wavelength'],
        y=data['intensity_corrected'],
        name='Corrected Spectrum',
        line=dict(color='black', width=2)
    ))
    
    # Add shaded regions for expected peak locations
    peak_ranges = {'D': (1250, 1450), 'G': (1480, 1680), '2D': (2500, 2900)}
    peak_colors = {'D': 'rgba(255,0,0,0.1)', 'G': 'rgba(0,0,255,0.1)', '2D': 'rgba(0,255,0,0.1)'}
    
    max_intensity = data['intensity_corrected'].max()
    for peak_name, (lo, hi) in peak_ranges.items():
        fig.add_vrect(
            x0=lo, x1=hi,
            fillcolor=peak_colors[peak_name],
            layer="below",
            line_width=0,
            annotation_text=peak_name,
            annotation_position="top left"
        )
    
    # Plot fitted peaks
    colors = {'D': 'red', 'G': 'blue', '2D': 'green'}
    
    for peak_name, fit in data['fits'].items():
        if fit is not None:
            fig.add_trace(go.Scatter(
                x=fit['wavelength'],
                y=fit['fitted_curve'],
                name=f'{peak_name} Peak Fit',
                line=dict(color=colors.get(peak_name, 'purple'), width=2, dash='dot')
            ))
            
            # Add vertical line at peak position
            fig.add_vline(
                x=fit['position'],
                line=dict(color=colors.get(peak_name, 'purple'), width=1, dash='dash'),
                opacity=0.5,
                annotation_text=f"{peak_name}: {fit['position']:.0f}",
                annotation_position="top"
            )
    
    fig.update_layout(
        title='Raman Spectrum with Peak Fits',
        xaxis_title='Raman Shift (cm⁻¹)',
        yaxis_title='Intensity (a.u.)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_peak_fit_detail(wavelength: np.ndarray,
                         intensity: np.ndarray,
                         fit_data: Dict,
                         peak_name: str) -> go.Figure:
    """
    Plot detailed view of single peak fit with residuals.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        fit_data: Fit result dictionary
        peak_name: Name of the peak (D, G, or 2D)
        
    Returns:
        Plotly figure with subplots
    """
    # Create figure with secondary y-axis for residuals
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{peak_name} Peak Fit', 'Residuals'),
        vertical_spacing=0.1
    )
    
    # Plot data
    fig.add_trace(
        go.Scatter(
            x=fit_data['wavelength'],
            y=fit_data['intensity'],
            name='Data',
            mode='markers',
            marker=dict(size=4, color='black')
        ),
        row=1, col=1
    )
    
    # Plot fit
    fig.add_trace(
        go.Scatter(
            x=fit_data['wavelength'],
            y=fit_data['fitted_curve'],
            name='Fit',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add peak position marker
    fig.add_vline(
        x=fit_data['position'],
        line=dict(color='blue', width=1, dash='dash'),
        row=1, col=1
    )
    
    # Calculate and plot residuals
    residuals = fit_data['intensity'] - fit_data['fitted_curve']
    
    fig.add_trace(
        go.Scatter(
            x=fit_data['wavelength'],
            y=residuals,
            name='Residuals',
            mode='markers',
            marker=dict(size=3, color='gray')
        ),
        row=2, col=1
    )
    
    # Add zero line for residuals
    fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text='Raman Shift (cm⁻¹)', row=2, col=1)
    fig.update_yaxes(title_text='Intensity (a.u.)', row=1, col=1)
    fig.update_yaxes(title_text='Residual', row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_comparison(data_dict: Dict,
                   materials: List[str],
                   normalize: bool = False,
                   region: str = "Full Spectrum") -> go.Figure:
    """
    Plot comparison of multiple spectra.
    
    Args:
        data_dict: Dictionary of analysis results
        materials: List of material names to compare
        normalize: Whether to normalize intensities
        region: Region to display
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Define wavelength ranges for different regions
    ranges = {
        "Full Spectrum": (1000, 3000),
        "D Region (1200-1500)": (1200, 1500),
        "G Region (1500-1700)": (1500, 1700),
        "2D Region (2500-2900)": (2500, 2900)
    }
    
    wave_min, wave_max = ranges.get(region, (1000, 3000))
    
    for material in materials:
        if material in data_dict:
            data = data_dict[material]
            wavelength = data['wavelength']
            intensity = data['intensity_corrected']
            
            # Filter to region
            mask = (wavelength >= wave_min) & (wavelength <= wave_max)
            wave_region = wavelength[mask]
            int_region = intensity[mask]
            
            # Normalize if requested
            if normalize and len(int_region) > 0:
                int_region = int_region / int_region.max()
            
            fig.add_trace(go.Scatter(
                x=wave_region,
                y=int_region,
                name=material,
                line=dict(width=2),
                mode='lines'
            ))
    
    ylabel = 'Normalized Intensity (a.u.)' if normalize else 'Intensity (a.u.)'
    
    fig.update_layout(
        title=f'Spectrum Comparison - {region}',
        xaxis_title='Raman Shift (cm⁻¹)',
        yaxis_title=ylabel,
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def plot_metrics_comparison(data_dict: Dict, materials: List[str]) -> Dict[str, go.Figure]:
    """
    Create comparison plots for various metrics.
    
    Args:
        data_dict: Dictionary of analysis results
        materials: List of material names to compare
        
    Returns:
        Dictionary of plotly figures
    """
    figures = {}
    
    # Extract metrics for all materials
    id_ig_values = []
    i2d_ig_values = []
    fwhm_2d_values = []
    d_positions = []
    g_positions = []
    td_positions = []
    
    for material in materials:
        if material in data_dict:
            metrics = data_dict[material]['metrics']
            id_ig_values.append(metrics['id_ig_ratio'])
            i2d_ig_values.append(metrics['i2d_ig_ratio'])
            fwhm_2d_values.append(metrics['fwhm_2d'])
            d_positions.append(metrics['d_position'])
            g_positions.append(metrics['g_position'])
            td_positions.append(metrics['2d_position'])
    
    # I(D)/I(G) comparison
    fig_id_ig = go.Figure()
    fig_id_ig.add_trace(go.Bar(
        x=materials,
        y=id_ig_values,
        marker_color='lightcoral',
        name='I(D)/I(G)'
    ))
    fig_id_ig.update_layout(
        title='Defect Density Comparison - I(D)/I(G)',
        yaxis_title='I(D)/I(G) Ratio',
        template='plotly_white',
        height=400
    )
    figures['id_ig'] = fig_id_ig
    
    # I(2D)/I(G) comparison
    fig_i2d_ig = go.Figure()
    fig_i2d_ig.add_trace(go.Bar(
        x=materials,
        y=i2d_ig_values,
        marker_color='lightblue',
        name='I(2D)/I(G)'
    ))
    fig_i2d_ig.update_layout(
        title='Layer Number Indicator - I(2D)/I(G)',
        yaxis_title='I(2D)/I(G) Ratio',
        template='plotly_white',
        height=400
    )
    figures['i2d_ig'] = fig_i2d_ig
    
    # FWHM comparison
    fig_fwhm = go.Figure()
    fig_fwhm.add_trace(go.Bar(
        x=materials,
        y=fwhm_2d_values,
        marker_color='lightgreen',
        name='2D FWHM'
    ))
    fig_fwhm.update_layout(
        title='2D Peak Width - FWHM',
        yaxis_title='FWHM (cm⁻¹)',
        template='plotly_white',
        height=400
    )
    figures['fwhm'] = fig_fwhm
    
    # Peak positions comparison
    fig_positions = go.Figure()
    fig_positions.add_trace(go.Scatter(
        x=materials,
        y=d_positions,
        name='D Peak',
        mode='markers+lines',
        marker=dict(size=10, color='red')
    ))
    fig_positions.add_trace(go.Scatter(
        x=materials,
        y=g_positions,
        name='G Peak',
        mode='markers+lines',
        marker=dict(size=10, color='blue')
    ))
    fig_positions.add_trace(go.Scatter(
        x=materials,
        y=td_positions,
        name='2D Peak',
        mode='markers+lines',
        marker=dict(size=10, color='green')
    ))
    fig_positions.update_layout(
        title='Peak Positions Comparison',
        yaxis_title='Position (cm⁻¹)',
        template='plotly_white',
        height=400
    )
    figures['positions'] = fig_positions
    
    return figures


def create_quality_heatmap(data_dict: Dict, materials: List[str]) -> go.Figure:
    """
    Create a heatmap showing quality metrics across materials.
    
    Args:
        data_dict: Dictionary of analysis results
        materials: List of material names
        
    Returns:
        Plotly figure
    """
    metrics_names = ['I(D)/I(G)', 'I(2D)/I(G)', '2D FWHM', 'Overall Score']
    values = []
    
    for material in materials:
        if material in data_dict:
            metrics = data_dict[material]['metrics']
            quality = data_dict[material]['quality']
            
            # Normalize metrics to 0-1 scale (lower is better for some)
            id_ig_norm = 1 - min(metrics['id_ig_ratio'] or 0, 1)
            i2d_ig_norm = min((metrics['i2d_ig_ratio'] or 0) / 3, 1)
            fwhm_norm = 1 - min((metrics['fwhm_2d'] or 100) / 100, 1)
            
            quality_scores = {'Excellent': 1.0, 'Good': 0.75, 'Fair': 0.5, 'Poor': 0.25}
            overall_norm = quality_scores.get(quality['overall'], 0)
            
            values.append([id_ig_norm, i2d_ig_norm, fwhm_norm, overall_norm])
        else:
            values.append([0, 0, 0, 0])
    
    fig = go.Figure(data=go.Heatmap(
        z=np.array(values).T,
        x=materials,
        y=metrics_names,
        colorscale='RdYlGn',
        text=np.array(values).T,
        texttemplate='%{text:.2f}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Quality Metrics Heatmap',
        template='plotly_white',
        height=400
    )
    
    return fig