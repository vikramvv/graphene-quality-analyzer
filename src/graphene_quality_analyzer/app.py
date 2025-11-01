import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import io

from data_loader import load_excel_data
from preprocessing import baseline_correction
from peak_detection import detect_peaks, refine_peak_regions
from peak_fitting import fit_peak, fit_all_peaks
from metrics import calculate_quality_metrics, interpret_quality
from visualization import (
    plot_spectrum_with_peaks,
    plot_comparison,
    plot_metrics_comparison,
    plot_peak_fit_detail
)
from export import export_results_to_excel, generate_report_text

# Page config
st.set_page_config(
    page_title="Graphene Quality Analyzer",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'analyzed_data' not in st.session_state:
    st.session_state.analyzed_data = {}
if 'verified_materials' not in st.session_state:
    st.session_state.verified_materials = set()
if 'material_notes' not in st.session_state:
    st.session_state.material_notes = {}
if 'quality_thresholds' not in st.session_state:
    st.session_state.quality_thresholds = {
        'id_ig_excellent': 0.1,
        'id_ig_good': 0.5,
        'i2d_ig_single_layer': 2.0,
        'i2d_ig_few_layer': 1.0,
        'fwhm_2d_single': 35,
        'fwhm_2d_few': 50
    }

# Title and description
st.title("🔬 Graphene Quality Analyzer")
st.markdown("*Automated Raman spectroscopy analysis for graphene characterization*")

# Sidebar
with st.sidebar:
    st.header("📁 Data Input")
    
    uploaded_file = st.file_uploader(
        "Upload Excel file with Raman data",
        type=['xlsx', 'xls'],
        help="Each sheet should contain wavelength (Col A) and intensity (Col B)"
    )
    
    if uploaded_file:
        # Load data
        try:
            data_dict = load_excel_data(uploaded_file)
            st.success(f"✅ Loaded {len(data_dict)} sheets")
            
            # Show wavelength range info
            with st.expander("📊 Data Info"):
                for sheet_name, (wave, _) in data_dict.items():
                    st.write(f"**{sheet_name}**: {wave.min():.1f} - {wave.max():.1f} cm⁻¹ ({len(wave)} points)")
            
            # Sheet selection
            st.subheader("📊 Select Materials")
            selected_sheets = st.multiselect(
                "Choose sheets to analyze",
                options=list(data_dict.keys()),
                default=list(data_dict.keys())
            )
            
            st.divider()
            
            # Analysis parameters
            st.subheader("⚙️ Analysis Parameters")
            
            with st.expander("Peak Detection", expanded=False):
                prominence = st.slider(
                    "Peak prominence",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.1,
                    step=0.01,
                    help="Higher values = more selective peak detection"
                )
                
                min_peak_distance = st.slider(
                    "Minimum peak distance (cm⁻¹)",
                    min_value=50,
                    max_value=300,
                    value=100,
                    help="Minimum separation between detected peaks"
                )
            
            with st.expander("Baseline Correction", expanded=False):
                baseline_poly_order = st.slider(
                    "Polynomial order",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Higher = more flexible baseline"
                )
            
            with st.expander("Quality Thresholds", expanded=False):
                st.markdown("**I(D)/I(G) Ratio**")
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.quality_thresholds['id_ig_excellent'] = st.number_input(
                        "Excellent (<)",
                        value=0.1,
                        step=0.05
                    )
                with col2:
                    st.session_state.quality_thresholds['id_ig_good'] = st.number_input(
                        "Good (<)",
                        value=0.5,
                        step=0.1
                    )
                
                st.markdown("**I(2D)/I(G) Ratio**")
                col3, col4 = st.columns(2)
                with col3:
                    st.session_state.quality_thresholds['i2d_ig_single_layer'] = st.number_input(
                        "Single layer (>)",
                        value=2.0,
                        step=0.1
                    )
                with col4:
                    st.session_state.quality_thresholds['i2d_ig_few_layer'] = st.number_input(
                        "Few layers (>)",
                        value=1.0,
                        step=0.1
                    )
                
                st.markdown("**2D Peak FWHM (cm⁻¹)**")
                col5, col6 = st.columns(2)
                with col5:
                    st.session_state.quality_thresholds['fwhm_2d_single'] = st.number_input(
                        "Single layer (<)",
                        value=35,
                        step=5
                    )
                with col6:
                    st.session_state.quality_thresholds['fwhm_2d_few'] = st.number_input(
                        "Few layers (<)",
                        value=50,
                        step=5
                    )
            
            st.divider()
            
            # Auto-save toggle
            auto_save = st.checkbox("Auto-save analysis state", value=True)
            
            # Analyze button
            if st.button("🔍 Analyze Selected Materials", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, sheet_name in enumerate(selected_sheets):
                    status_text.text(f"Analyzing: {sheet_name}")
                    
                    # Get data
                    wavelength, intensity = data_dict[sheet_name]
                    
                    # Baseline correction
                    intensity_corrected, baseline = baseline_correction(
                        wavelength, 
                        intensity,
                        poly_order=baseline_poly_order
                    )
                    
                    # Detect peaks
                    peak_indices = detect_peaks(
                        wavelength,
                        intensity_corrected,
                        prominence=prominence,
                        distance=min_peak_distance
                    )
                    
                    # Refine to D, G, 2D regions
                    peak_regions = refine_peak_regions(wavelength, peak_indices)
                    
                    # Fit peaks
                    fits = fit_all_peaks(wavelength, intensity_corrected, peak_regions)
                    
                    # Calculate metrics
                    metrics = calculate_quality_metrics(fits)
                    quality = interpret_quality(metrics, st.session_state.quality_thresholds)
                    
                    # Store results
                    st.session_state.analyzed_data[sheet_name] = {
                        'wavelength': wavelength,
                        'intensity_raw': intensity,
                        'intensity_corrected': intensity_corrected,
                        'baseline': baseline,
                        'peak_regions': peak_regions,
                        'fits': fits,
                        'metrics': metrics,
                        'quality': quality
                    }
                    
                    progress_bar.progress((idx + 1) / len(selected_sheets))
                
                status_text.text("✅ Analysis complete!")
                st.success(f"Analyzed {len(selected_sheets)} materials")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

# Main content area
if uploaded_file and selected_sheets and st.session_state.analyzed_data:
    
    tabs = st.tabs([
        "📈 Individual Analysis",
        "🔄 Comparison View",
        "📊 Metrics Dashboard",
        "💾 Export Results"
    ])
    
    # TAB 1: Individual Analysis
    with tabs[0]:
        st.header("Individual Material Analysis")
        
        # Material selector
        material = st.selectbox(
            "Select material to analyze",
            options=list(st.session_state.analyzed_data.keys())
        )
        
        if material in st.session_state.analyzed_data:
            data = st.session_state.analyzed_data[material]
            
            # Verification status
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                if material in st.session_state.verified_materials:
                    st.success("✅ Verified")
                else:
                    st.warning("⚠️ Not verified")
            with col2:
                if st.button("✓ Verify", key=f"verify_{material}"):
                    st.session_state.verified_materials.add(material)
                    st.rerun()
            with col3:
                if st.button("↻ Re-analyze", key=f"reanalyze_{material}"):
                    st.session_state.verified_materials.discard(material)
                    st.rerun()
            
            # Notes section
            with st.expander("📝 Add Notes"):
                notes = st.text_area(
                    "Material notes",
                    value=st.session_state.material_notes.get(material, ""),
                    key=f"notes_{material}"
                )
                st.session_state.material_notes[material] = notes
            
            # Debug info
            with st.expander("🔍 Detection Debug Info"):
                st.write("**Peak Detection Results:**")
                for peak_name, region in data['peak_regions'].items():
                    if region:
                        start_idx, end_idx, peak_idx = region
                        peak_pos = data['wavelength'][peak_idx]
                        st.write(f"- {peak_name} peak detected at {peak_pos:.1f} cm⁻¹")
                    else:
                        st.write(f"- {peak_name} peak: NOT DETECTED")
                
                st.write(f"\n**Data Range:** {data['wavelength'].min():.1f} - {data['wavelength'].max():.1f} cm⁻¹")
                st.write(f"**Total Points:** {len(data['wavelength'])}")
            
            st.divider()
            
            # Main spectrum plot
            fig = plot_spectrum_with_peaks(data)
            st.plotly_chart(fig, use_container_width=True, key=f"main_spectrum_{material}")
            
            st.divider()
            
            # Quality summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quality Assessment")
                quality = data['quality']
                
                # Overall quality badge
                if quality['overall'] == 'Excellent':
                    st.success(f"🌟 **Overall Quality: {quality['overall']}**")
                elif quality['overall'] == 'Good':
                    st.info(f"✓ **Overall Quality: {quality['overall']}**")
                elif quality['overall'] == 'Fair':
                    st.warning(f"⚠ **Overall Quality: {quality['overall']}**")
                else:
                    st.error(f"⚠ **Overall Quality: {quality['overall']}**")
                
                st.markdown(f"**Layer Type:** {quality['layer_type']}")
                st.markdown(f"**Defect Level:** {quality['defect_level']}")
            
            with col2:
                st.subheader("Key Metrics")
                metrics = data['metrics']
                
                # Display metrics in a nice format
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("I(D)/I(G)", f"{metrics['id_ig_ratio']:.3f}" if metrics['id_ig_ratio'] else "N/A")
                    st.metric("I(2D)/I(G)", f"{metrics['i2d_ig_ratio']:.3f}" if metrics['i2d_ig_ratio'] else "N/A")
                    st.metric("D Position", f"{metrics['d_position']:.1f} cm⁻¹" if metrics['d_position'] else "N/A")
                with metric_col2:
                    st.metric("2D FWHM", f"{metrics['fwhm_2d']:.1f} cm⁻¹" if metrics['fwhm_2d'] else "N/A")
                    st.metric("G Position", f"{metrics['g_position']:.1f} cm⁻¹" if metrics['g_position'] else "N/A")
                    st.metric("2D Position", f"{metrics['2d_position']:.1f} cm⁻¹" if metrics['2d_position'] else "N/A")
            
            st.divider()
            
            # Individual peak details
            st.subheader("Peak Fitting Details")
            
            peak_tabs = st.tabs(["D Peak (~1350)", "G Peak (~1580)", "2D Peak (~2700)"])
            
            for peak_idx, (peak_name, peak_tab) in enumerate(zip(['D', 'G', '2D'], peak_tabs)):
                with peak_tab:
                    if peak_name in data['fits'] and data['fits'][peak_name] is not None:
                        fit_data = data['fits'][peak_name]
                        
                        # Plot detailed fit
                        fig_detail = plot_peak_fit_detail(
                            data['wavelength'],
                            data['intensity_corrected'],
                            fit_data,
                            peak_name
                        )
                        st.plotly_chart(fig_detail, use_container_width=True, key=f"detail_{material}_{peak_name}")
                        
                        # Fit parameters
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Position", f"{fit_data['position']:.1f} cm⁻¹")
                        with col2:
                            st.metric("Intensity", f"{fit_data['amplitude']:.0f}")
                        with col3:
                            st.metric("FWHM", f"{fit_data['fwhm']:.1f} cm⁻¹")
                        
                        # R-squared
                        st.metric("Fit Quality (R²)", f"{fit_data['r_squared']:.4f}")
                        
                        # Manual adjustment option
                        with st.expander("🔧 Manual Adjustment"):
                            st.warning("Feature coming soon: Manual peak window adjustment")
                    else:
                        st.error(f"{peak_name} peak not detected or fit failed")
    
    # TAB 2: Comparison View
    with tabs[1]:
        st.header("Material Comparison")
        
        # Normalization option
        normalize = st.checkbox("Normalize intensities", value=True)
        
        # Region selector
        region = st.radio(
            "Select region to view",
            options=["Full Spectrum", "D Region (1200-1500)", "G Region (1500-1700)", "2D Region (2500-2900)"],
            horizontal=True
        )
        
        # Create comparison plot
        materials_to_compare = [m for m in selected_sheets if m in st.session_state.analyzed_data]
        
        if materials_to_compare:
            fig = plot_comparison(
                st.session_state.analyzed_data,
                materials_to_compare,
                normalize=normalize,
                region=region
            )
            st.plotly_chart(fig, use_container_width=True, key="comparison_plot")
            
            # Quick comparison table
            st.subheader("Quick Metrics Comparison")
            
            comparison_data = []
            for mat in materials_to_compare:
                metrics = st.session_state.analyzed_data[mat]['metrics']
                quality = st.session_state.analyzed_data[mat]['quality']
                comparison_data.append({
                    'Material': mat,
                    'Overall Quality': quality['overall'],
                    'I(D)/I(G)': f"{metrics['id_ig_ratio']:.3f}" if metrics['id_ig_ratio'] else "N/A",
                    'I(2D)/I(G)': f"{metrics['i2d_ig_ratio']:.3f}" if metrics['i2d_ig_ratio'] else "N/A",
                    '2D FWHM': f"{metrics['fwhm_2d']:.1f}" if metrics['fwhm_2d'] else "N/A",
                    'Layer Type': quality['layer_type'],
                    'Verified': '✅' if mat in st.session_state.verified_materials else '❌'
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No materials analyzed yet. Please run analysis first.")
    
    # TAB 3: Metrics Dashboard
    with tabs[2]:
        st.header("Metrics Dashboard")
        
        materials_to_compare = [m for m in selected_sheets if m in st.session_state.analyzed_data]
        
        if materials_to_compare:
            # Create metrics comparison plots
            figs = plot_metrics_comparison(st.session_state.analyzed_data, materials_to_compare)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(figs['id_ig'], use_container_width=True, key="metrics_id_ig")
                st.plotly_chart(figs['fwhm'], use_container_width=True, key="metrics_fwhm")
            with col2:
                st.plotly_chart(figs['i2d_ig'], use_container_width=True, key="metrics_i2d_ig")
                st.plotly_chart(figs['positions'], use_container_width=True, key="metrics_positions")
            
            st.divider()
            
            # Detailed metrics table
            st.subheader("Detailed Metrics Table")
            
            detailed_data = []
            for mat in materials_to_compare:
                data = st.session_state.analyzed_data[mat]
                metrics = data['metrics']
                quality = data['quality']
                
                detailed_data.append({
                    'Material': mat,
                    'D Position': f"{metrics['d_position']:.1f}" if metrics['d_position'] else "N/A",
                    'D FWHM': f"{metrics['fwhm_d']:.1f}" if metrics['fwhm_d'] else "N/A",
                    'G Position': f"{metrics['g_position']:.1f}" if metrics['g_position'] else "N/A",
                    'G FWHM': f"{metrics['fwhm_g']:.1f}" if metrics['fwhm_g'] else "N/A",
                    '2D Position': f"{metrics['2d_position']:.1f}" if metrics['2d_position'] else "N/A",
                    '2D FWHM': f"{metrics['fwhm_2d']:.1f}" if metrics['fwhm_2d'] else "N/A",
                    'I(D)/I(G)': f"{metrics['id_ig_ratio']:.3f}" if metrics['id_ig_ratio'] else "N/A",
                    'I(2D)/I(G)': f"{metrics['i2d_ig_ratio']:.3f}" if metrics['i2d_ig_ratio'] else "N/A",
                    'Quality': quality['overall'],
                    'Layer Type': quality['layer_type'],
                    'Defect Level': quality['defect_level']
                })
            
            df_detailed = pd.DataFrame(detailed_data)
            st.dataframe(df_detailed, use_container_width=True, hide_index=True)
        else:
            st.info("No materials analyzed yet. Please run analysis first.")
    
    # TAB 4: Export Results
    with tabs[3]:
        st.header("Export Results")
        
        materials_to_export = [m for m in selected_sheets if m in st.session_state.analyzed_data]
        
        if materials_to_export:
            st.subheader("📥 Download Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel export
                if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
                    excel_buffer = export_results_to_excel(
                        st.session_state.analyzed_data,
                        materials_to_export,
                        st.session_state.material_notes,
                        st.session_state.quality_thresholds
                    )
                    
                    st.download_button(
                        label="⬇️ Download Excel Report",
                        data=excel_buffer,
                        file_name="graphene_analysis_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                # Text report
                if st.button("📄 Generate Text Report", type="primary", use_container_width=True):
                    report_text = generate_report_text(
                        st.session_state.analyzed_data,
                        materials_to_export,
                        st.session_state.material_notes,
                        st.session_state.quality_thresholds
                    )
                    
                    st.download_button(
                        label="⬇️ Download Text Report",
                        data=report_text,
                        file_name="graphene_analysis_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            st.divider()
            
            # Individual plot exports
            st.subheader("📈 Export Individual Plots")
            
            export_material = st.selectbox(
                "Select material for plot export",
                options=materials_to_export,
                key="export_material_select"
            )
            
            if export_material:
                plot_type = st.radio(
                    "Select plot type",
                    options=["Full Spectrum with Peaks", "D Peak Detail", "G Peak Detail", "2D Peak Detail"],
                    horizontal=True
                )
                
                # Generate the selected plot
                data = st.session_state.analyzed_data[export_material]
                
                if plot_type == "Full Spectrum with Peaks":
                    fig = plot_spectrum_with_peaks(data)
                else:
                    peak_name = plot_type.split()[0]
                    if peak_name in data['fits'] and data['fits'][peak_name]:
                        fig = plot_peak_fit_detail(
                            data['wavelength'],
                            data['intensity_corrected'],
                            data['fits'][peak_name],
                            peak_name
                        )
                    else:
                        st.error(f"{peak_name} peak not available")
                        fig = None
                
                if fig:
                    # Display plot
                    st.plotly_chart(fig, use_container_width=True, key=f"export_{export_material}_{plot_type}")
                    
                    # Export options
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        img_bytes = fig.to_image(format="png", width=1200, height=800)
                        st.download_button(
                            "⬇️ Download PNG",
                            data=img_bytes,
                            file_name=f"{export_material}_{plot_type.replace(' ', '_')}.png",
                            mime="image/png"
                        )
                    with col2:
                        img_bytes_svg = fig.to_image(format="svg", width=1200, height=800)
                        st.download_button(
                            "⬇️ Download SVG",
                            data=img_bytes_svg,
                            file_name=f"{export_material}_{plot_type.replace(' ', '_')}.svg",
                            mime="image/svg+xml"
                        )
                    with col3:
                        html_str = fig.to_html()
                        st.download_button(
                            "⬇️ Download HTML",
                            data=html_str,
                            file_name=f"{export_material}_{plot_type.replace(' ', '_')}.html",
                            mime="text/html"
                        )
        else:
            st.info("No materials analyzed yet. Please run analysis first.")

else:
    # Welcome screen
    st.info("👈 Upload an Excel file to get started!")
    
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Data**: Upload an Excel file with Raman spectra
           - Each sheet represents one material/sample
           - Column A: Wavelength (cm⁻¹)
           - Column B: Intensity
        
        2. **Select Materials**: Choose which sheets to analyze
        
        3. **Adjust Parameters** (optional): Fine-tune peak detection and baseline correction
        
        4. **Analyze**: Click the analyze button to process all selected materials
        
        5. **Review Results**: 
           - Individual Analysis: Examine each material in detail
           - Comparison View: Compare multiple materials
           - Metrics Dashboard: View quantitative comparisons
        
        6. **Verify & Export**: Verify your results and export to Excel/PDF
        
        ### What the app analyzes:
        - **D Peak** (~1350 cm⁻¹): Defects and disorder
        - **G Peak** (~1580 cm⁻¹): Graphitic structure
        - **2D Peak** (~2700 cm⁻¹): Layer information
        
        ### Quality Metrics:
        - **I(D)/I(G)**: Defect density (lower is better)
        - **I(2D)/I(G)**: Number of layers (>2 = single layer)
        - **2D FWHM**: Peak width (<35 cm⁻¹ = single layer)
        """)