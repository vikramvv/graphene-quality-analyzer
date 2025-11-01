# Quick Setup Guide

## Directory Structure

Create the following structure:

```
graphene-quality-analyzer/
├── pyproject.toml
├── README.md
├── SETUP.md (this file)
└── src/
    └── graphene_quality_analyzer/
        ├── __init__.py
        ├── app.py
        ├── data_loader.py
        ├── preprocessing.py
        ├── peak_detection.py
        ├── peak_fitting.py
        ├── metrics.py
        ├── visualization.py
        └── export.py
```

## Step-by-Step Setup

### 1. Create Directory Structure

```bash
mkdir -p graphene-quality-analyzer/src/graphene_quality_analyzer
cd graphene-quality-analyzer
```

### 2. Copy Files

Place all the Python files in the `src/graphene_quality_analyzer/` directory:
- `__init__.py`
- `app.py`
- `data_loader.py`
- `preprocessing.py`
- `peak_detection.py`
- `peak_fitting.py`
- `metrics.py`
- `visualization.py`
- `export.py`

Place these files in the root directory:
- `pyproject.toml`
- `README.md`

### 3. Initialize with Hatch

```bash
# Make sure hatch is installed
pip install hatch

# Create the environment
hatch env create

# This will install all dependencies automatically
```

### 4. Run the Application

```bash
# Using hatch
hatch run app

# Or directly with streamlit
hatch shell
streamlit run src/graphene_quality_analyzer/app.py
```

The app will open in your browser at `http://localhost:8501`

## Testing Your Setup

### 1. Prepare Test Data

Create a test Excel file with:
- Sheet 1: "Sample_A"
  - Column A: Wavelength values from 1000 to 3000 cm⁻¹
  - Column B: Corresponding intensity values

### 2. Upload and Analyze

1. Open the app
2. Upload your test Excel file
3. Select materials to analyze
4. Click "Analyze Selected Materials"
5. Review results in the tabs

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the hatch environment:

```bash
hatch shell
```

### Module Not Found

If you get "module not found" errors, install in development mode:

```bash
pip install -e .
```

### Streamlit Won't Start

Try running directly:

```bash
cd graphene-quality-analyzer
python -m streamlit run src/graphene_quality_analyzer/app.py
```

### Port Already in Use

Change the port:

```bash
streamlit run src/graphene_quality_analyzer/app.py --server.port 8502
```

## Development Tips

### Running in Development Mode

```bash
# Activate hatch shell
hatch shell

# Run with auto-reload
streamlit run src/graphene_quality_analyzer/app.py --server.runOnSave true
```

### Viewing Logs

Streamlit logs appear in the terminal. For more detailed logs:

```bash
streamlit run src/graphene_quality_analyzer/app.py --logger.level debug
```

### Code Formatting

```bash
# Format code
hatch run black src/

# Check for issues
hatch run ruff check src/
```

## Next Steps

1. Test with your own Raman data
2. Adjust quality thresholds for your specific requirements
3. Customize visualizations if needed
4. Export and share reports

## Common Issues

### "No module named 'graphene_quality_analyzer'"

Solution:
```bash
pip install -e .
```

### Plotly plots not showing

Make sure you have the latest plotly:
```bash
pip install --upgrade plotly
```

### Excel export failing

Ensure xlsxwriter is installed:
```bash
pip install xlsxwriter
```

## Getting Help

- Check the README.md for detailed documentation
- Review the code comments in each module
- Open an issue if you encounter bugs