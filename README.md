# Graphene Quality Analyzer

A Streamlit-based application for automated Raman spectroscopy analysis of graphene materials.

## Features

- 📊 **Multi-material Analysis**: Analyze multiple samples from a single Excel file
- 🔍 **Automated Peak Detection**: Automatically identifies D, G, and 2D peaks
- 📈 **Peak Fitting**: Lorentzian fitting with quality metrics (R²)
- ✅ **Verification Workflow**: Review and verify each analysis
- 📝 **Quality Assessment**: Automatic quality grading based on standard metrics
- 🔄 **Comparison Tools**: Compare multiple materials side-by-side
- 💾 **Export Options**: Generate Excel reports, text summaries, and plots

## Installation

### Using Hatch (Recommended)

```bash
# Clone the repository
git clone https://github.com/vikramvv/graphene-quality-analyzer.git
cd graphene-quality-analyzer

# Create and activate environment with Hatch
hatch env create

# Run the app
hatch run app
```

### Using pip

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the app
streamlit run src/graphene_quality_analyzer/app.py
```

## Project Structure

```
graphene-quality-analyzer/
├── pyproject.toml              # Project configuration
├── README.md                   # This file
└── src/
    └── graphene_quality_analyzer/
        ├── __init__.py
        ├── app.py              # Main Streamlit application
        ├── data_loader.py      # Excel data loading
        ├── preprocessing.py    # Baseline correction
        ├── peak_detection.py   # Peak detection algorithms
        ├── peak_fitting.py     # Peak fitting functions
        ├── metrics.py          # Quality metrics calculation
        ├── visualization.py    # Plotting functions
        └── export.py           # Results export
```

## Usage

### Input Data Format

Your Excel file should have:
- **Multiple sheets**: Each sheet represents one material/sample
- **Column A**: Wavelength or Raman shift (cm⁻¹)
- **Column B**: Intensity (arbitrary units)

Example:
```
| Wavelength | Intensity |
|------------|-----------|
| 1000       | 150       |
| 1001       | 152       |
| ...        | ...       |
```

### Workflow

1. **Upload Data**: Click "Upload Excel file" in the sidebar
2. **Select Materials**: Choose which sheets to analyze
3. **Adjust Parameters** (optional): 
   - Peak detection sensitivity
   - Baseline correction settings
   - Quality thresholds
4. **Run Analysis**: Click "Analyze Selected Materials"
5. **Review Results**:
   - **Individual Analysis**: Examine each material, verify fits
   - **Comparison View**: Compare spectra and metrics
   - **Metrics Dashboard**: View quantitative comparisons
6. **Export**: Download Excel reports, text summaries, or individual plots

### Quality Metrics

The app analyzes three key peaks:

- **D Peak (~1350 cm⁻¹)**: Defects and disorder
- **G Peak (~1580 cm⁻¹)**: Graphitic structure  
- **2D Peak (~2700 cm⁻¹)**: Layer information

Key metrics calculated:

- **I(D)/I(G)**: Defect density (lower is better)
  - < 0.1: Excellent quality
  - < 0.5: Good quality
- **I(2D)/I(G)**: Layer number indicator
  - \> 2: Single layer graphene
  - \> 1: Few layers (2-5)
  - < 1: Multi-layer (>5)
- **2D FWHM**: Peak width
  - < 35 cm⁻¹: Single layer
  - < 50 cm⁻¹: Few layers

## Customization

### Quality Thresholds

You can adjust quality thresholds in the sidebar:
- I(D)/I(G) thresholds for excellent/good quality
- I(2D)/I(G) thresholds for layer classification
- 2D FWHM thresholds for layer determination

### Peak Detection Parameters

- **Prominence**: Higher values = more selective (fewer peaks detected)
- **Distance**: Minimum separation between peaks
- **Baseline order**: Polynomial order for baseline correction

## Development

### Running Tests

```bash
hatch run test
```

### Code Formatting

```bash
hatch run black src/
hatch run ruff check src/
```

## References

- Tuinstra, F., & Koenig, J. L. (1970). Raman Spectrum of Graphite. *J. Chem. Phys.*, 53, 1126.
- Ferrari, A. C., et al. (2006). Raman Spectrum of Graphene and Graphene Layers. *Phys. Rev. Lett.*, 97, 187401.
- Malard, L. M., et al. (2009). Raman spectroscopy in graphene. *Physics Reports*, 473(5-6), 51-87.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please open an issue on GitHub.
