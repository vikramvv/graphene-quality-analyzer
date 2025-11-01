import pandas as pd
import io
from datetime import datetime
from typing import Dict, List


def export_results_to_excel(data_dict: Dict,
                            materials: List[str],
                            notes_dict: Dict[str, str],
                            thresholds: Dict[str, float]) -> io.BytesIO:
    """
    Export analysis results to Excel file.
    
    Args:
        data_dict: Dictionary of analysis results
        materials: List of material names to export
        notes_dict: Dictionary of user notes
        thresholds: Quality threshold values
        
    Returns:
        BytesIO buffer containing Excel file
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1
        })
        
        metric_format = workbook.add_format({'num_format': '0.000', 'border': 1})
        text_format = workbook.add_format({'border': 1})
        
        # Summary sheet
        summary_data = []
        for material in materials:
            if material in data_dict:
                data = data_dict[material]
                metrics = data['metrics']
                quality = data['quality']
                
                summary_data.append({
                    'Material': material,
                    'Overall Quality': quality['overall'],
                    'Layer Type': quality['layer_type'],
                    'Defect Level': quality['defect_level'],
                    'I(D)/I(G)': metrics['id_ig_ratio'],
                    'I(2D)/I(G)': metrics['i2d_ig_ratio'],
                    'D Position (cm⁻¹)': metrics['d_position'],
                    'G Position (cm⁻¹)': metrics['g_position'],
                    '2D Position (cm⁻¹)': metrics['2d_position'],
                    'D FWHM (cm⁻¹)': metrics['fwhm_d'],
                    'G FWHM (cm⁻¹)': metrics['fwhm_g'],
                    '2D FWHM (cm⁻¹)': metrics['fwhm_2d'],
                    'Notes': notes_dict.get(material, '')
                })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format summary sheet
        worksheet = writer.sheets['Summary']
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:D', 15)
        worksheet.set_column('E:L', 12)
        worksheet.set_column('M:M', 30)
        
        # Individual material sheets
        for material in materials:
            if material in data_dict:
                data = data_dict[material]
                metrics = data['metrics']
                quality = data['quality']
                fits = data['fits']
                
                # Create detailed sheet for this material
                sheet_name = material[:31]  # Excel sheet name limit
                
                rows = []
                rows.append(['Material Analysis Report'])
                rows.append(['Material Name:', material])
                rows.append(['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                rows.append([])
                
                rows.append(['QUALITY ASSESSMENT'])
                rows.append(['Overall Quality:', quality['overall']])
                rows.append(['Layer Type:', quality['layer_type']])
                rows.append(['Defect Level:', quality['defect_level']])
                rows.append([])
                
                rows.append(['KEY METRICS'])
                rows.append(['Metric', 'Value', 'Unit'])
                rows.append(['I(D)/I(G) Ratio', metrics['id_ig_ratio'], ''])
                rows.append(['I(2D)/I(G) Ratio', metrics['i2d_ig_ratio'], ''])
                rows.append([])
                
                rows.append(['D PEAK'])
                if fits['D']:
                    rows.append(['Position', fits['D']['position'], 'cm⁻¹'])
                    rows.append(['Intensity', fits['D']['amplitude'], 'a.u.'])
                    rows.append(['FWHM', fits['D']['fwhm'], 'cm⁻¹'])
                    rows.append(['R²', fits['D']['r_squared'], ''])
                else:
                    rows.append(['Not detected', '', ''])
                rows.append([])
                
                rows.append(['G PEAK'])
                if fits['G']:
                    rows.append(['Position', fits['G']['position'], 'cm⁻¹'])
                    rows.append(['Intensity', fits['G']['amplitude'], 'a.u.'])
                    rows.append(['FWHM', fits['G']['fwhm'], 'cm⁻¹'])
                    rows.append(['R²', fits['G']['r_squared'], ''])
                else:
                    rows.append(['Not detected', '', ''])
                rows.append([])
                
                rows.append(['2D PEAK'])
                if fits['2D']:
                    rows.append(['Position', fits['2D']['position'], 'cm⁻¹'])
                    rows.append(['Intensity', fits['2D']['amplitude'], 'a.u.'])
                    rows.append(['FWHM', fits['2D']['fwhm'], 'cm⁻¹'])
                    rows.append(['R²', fits['2D']['r_squared'], ''])
                else:
                    rows.append(['Not detected', '', ''])
                rows.append([])
                
                if quality['notes']:
                    rows.append(['NOTES'])
                    for note in quality['notes']:
                        rows.append([note])
                    rows.append([])
                
                if material in notes_dict and notes_dict[material]:
                    rows.append(['USER NOTES'])
                    rows.append([notes_dict[material]])
                
                df_detail = pd.DataFrame(rows)
                df_detail.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                
                # Format the sheet
                worksheet = writer.sheets[sheet_name]
                worksheet.set_column('A:A', 30)
                worksheet.set_column('B:B', 20)
                worksheet.set_column('C:C', 15)
        
        # Thresholds sheet
        threshold_data = []
        threshold_data.append({'Parameter': 'I(D)/I(G) - Excellent', 'Value': thresholds['id_ig_excellent']})
        threshold_data.append({'Parameter': 'I(D)/I(G) - Good', 'Value': thresholds['id_ig_good']})
        threshold_data.append({'Parameter': 'I(2D)/I(G) - Single Layer', 'Value': thresholds['i2d_ig_single_layer']})
        threshold_data.append({'Parameter': 'I(2D)/I(G) - Few Layers', 'Value': thresholds['i2d_ig_few_layer']})
        threshold_data.append({'Parameter': '2D FWHM - Single Layer', 'Value': thresholds['fwhm_2d_single']})
        threshold_data.append({'Parameter': '2D FWHM - Few Layers', 'Value': thresholds['fwhm_2d_few']})
        
        df_thresholds = pd.DataFrame(threshold_data)
        df_thresholds.to_excel(writer, sheet_name='Thresholds', index=False)
        
        worksheet = writer.sheets['Thresholds']
        worksheet.set_column('A:A', 30)
        worksheet.set_column('B:B', 15)
    
    output.seek(0)
    return output


def generate_report_text(data_dict: Dict,
                        materials: List[str],
                        notes_dict: Dict[str, str],
                        thresholds: Dict[str, float]) -> str:
    """
    Generate a text report of the analysis.
    
    Args:
        data_dict: Dictionary of analysis results
        materials: List of material names
        notes_dict: Dictionary of user notes
        thresholds: Quality threshold values
        
    Returns:
        String containing the formatted report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("GRAPHENE QUALITY ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Material':<25} {'Quality':<15} {'Layer Type':<20} {'I(D)/I(G)':<12} {'I(2D)/I(G)':<12}")
    lines.append("-" * 80)
    
    for material in materials:
        if material in data_dict:
            data = data_dict[material]
            metrics = data['metrics']
            quality = data['quality']
            
            id_ig_str = f"{metrics['id_ig_ratio']:.3f}" if metrics['id_ig_ratio'] else "N/A"
            i2d_ig_str = f"{metrics['i2d_ig_ratio']:.3f}" if metrics['i2d_ig_ratio'] else "N/A"
            
            lines.append(
                f"{material:<25} {quality['overall']:<15} {quality['layer_type']:<20} "
                f"{id_ig_str:<12} {i2d_ig_str:<12}"
            )
    
    lines.append("")
    lines.append("")
    
    # Detailed results for each material
    for material in materials:
        if material in data_dict:
            data = data_dict[material]
            metrics = data['metrics']
            quality = data['quality']
            fits = data['fits']
            
            lines.append("=" * 80)
            lines.append(f"MATERIAL: {material}")
            lines.append("=" * 80)
            lines.append("")
            
            lines.append("Quality Assessment:")
            lines.append(f"  Overall Quality:  {quality['overall']}")
            lines.append(f"  Layer Type:       {quality['layer_type']}")
            lines.append(f"  Defect Level:     {quality['defect_level']}")
            lines.append("")
            
            lines.append("Key Metrics:")
            lines.append(f"  I(D)/I(G) Ratio:  {metrics['id_ig_ratio']:.4f}" if metrics['id_ig_ratio'] else "  I(D)/I(G) Ratio:  N/A")
            lines.append(f"  I(2D)/I(G) Ratio: {metrics['i2d_ig_ratio']:.4f}" if metrics['i2d_ig_ratio'] else "  I(2D)/I(G) Ratio: N/A")
            lines.append("")
            
            lines.append("Peak Details:")
            for peak_name in ['D', 'G', '2D']:
                if fits[peak_name]:
                    fit = fits[peak_name]
                    lines.append(f"  {peak_name} Peak:")
                    lines.append(f"    Position: {fit['position']:.2f} cm⁻¹")
                    lines.append(f"    Intensity: {fit['amplitude']:.0f} a.u.")
                    lines.append(f"    FWHM: {fit['fwhm']:.2f} cm⁻¹")
                    lines.append(f"    R²: {fit['r_squared']:.4f}")
                else:
                    lines.append(f"  {peak_name} Peak: Not detected")
            lines.append("")
            
            if quality['notes']:
                lines.append("Interpretation Notes:")
                for note in quality['notes']:
                    lines.append(f"  • {note}")
                lines.append("")
            
            if material in notes_dict and notes_dict[material]:
                lines.append("User Notes:")
                lines.append(f"  {notes_dict[material]}")
                lines.append("")
    
    lines.append("=" * 80)
    lines.append("ANALYSIS PARAMETERS")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Quality Thresholds:")
    lines.append(f"  I(D)/I(G) - Excellent: < {thresholds['id_ig_excellent']}")
    lines.append(f"  I(D)/I(G) - Good: < {thresholds['id_ig_good']}")
    lines.append(f"  I(2D)/I(G) - Single Layer: > {thresholds['i2d_ig_single_layer']}")
    lines.append(f"  I(2D)/I(G) - Few Layers: > {thresholds['i2d_ig_few_layer']}")
    lines.append(f"  2D FWHM - Single Layer: < {thresholds['fwhm_2d_single']} cm⁻¹")
    lines.append(f"  2D FWHM - Few Layers: < {thresholds['fwhm_2d_few']} cm⁻¹")
    lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def export_individual_spectrum(wavelength, intensity, filename: str) -> io.BytesIO:
    """
    Export individual spectrum data to CSV.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        filename: Name for the file
        
    Returns:
        BytesIO buffer containing CSV data
    """
    df = pd.DataFrame({
        'Wavelength (cm-1)': wavelength,
        'Intensity (a.u.)': intensity
    })
    
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    return output