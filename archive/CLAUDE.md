# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LinkedIn Learning course repository for "Build with AI: Executing and Evaluating Hugging Face Models". The course focuses on strategically selecting, implementing, and rigorously evaluating pretrained models from Hugging Face for organizational use cases.

## Development Environment

### Python Environment Management
- Use `uv` for running Python scripts and managing virtual environments (as per user's global Claude instructions)
- To set up the environment: `uv venv` followed by `uv pip install -r requirements.txt` (when dependencies are added)

### Primary Development Platform
- The main course content is delivered via Google Colab notebook: https://colab.research.google.com/drive/1qEqr_iG45mc4R6BY3Dj9cwTC9mW1GcKk#scrollTo=0ax79bW5McJw
- Local development can be done with VS Code or GitHub Codespaces (devcontainer configuration provided)

## Project Structure

### Notebook Content
- `course_notebook.ipynb` - Main course notebook containing the flight cancellation predictor implementation
  - Data loading and preprocessing
  - Feature engineering (holidays, time indicators, airport characteristics)
  - Model training using AutoGluon with TabPFNMix
  - Comprehensive evaluation metrics (ROC-AUC, precision, recall, F1)
  - Feature importance analysis
  - Calibration threshold analysis

### Data Files
- `T_ONTIME_REPORTING11_A.parquet` - Flight on-time reporting data (Part A)
- `T_ONTIME_REPORTING11_B.parquet` - Flight on-time reporting data (Part B)
- `ARP-NPIAS-2025-2029-AppendixA.xlsx` - Airport reference data (NPIAS - National Plan of Integrated Airport Systems)

These are Apache Parquet and Excel files containing flight and airport data used for training the cancellation prediction model.

### Key Configurations
- `.devcontainer/devcontainer.json` - VS Code devcontainer setup with Python and Jupyter extensions
- `.github/workflows/main.yml` - GitHub Actions workflow for branch management (course structure)

## Course-Specific Notes

### Working with Course Materials
- The primary learning happens in the Google Colab notebook linked in the README
- This repository serves as a companion for the course with sample data files
- The repository uses a branch structure for different course sections (managed via GitHub Actions)

### Data Processing
When working with the provided data files:
- Use pandas or polars for reading Parquet files
- The flight data is split into two parts (A and B) - consider both when doing analysis
- Airport reference data in Excel format can be read with pandas or openpyxl

## Development Commands

### Setting Up Local Environment
```bash
# Create virtual environment with uv
uv venv

# Install core dependencies used in the notebook
uv pip install pandas numpy holidays matplotlib seaborn sweetviz
uv pip install autogluon evaluate
```

### Python Script Execution
```bash
# Run a Python script with uv
uv run script.py

# Install dependencies if requirements.txt is populated
uv pip install -r requirements.txt
```

### Working with Jupyter Notebooks Locally
```bash
# If you need to work with notebooks locally instead of Colab
uv pip install jupyter
uv run jupyter notebook course_notebook.ipynb
```

## Key Libraries and Models

### Core Dependencies
- **AutoGluon**: TabularPredictor for automated machine learning
- **TabPFNMix**: Primary model used for flight cancellation prediction
- **pandas/numpy**: Data manipulation and processing
- **holidays**: US holiday detection for feature engineering
- **evaluate**: Hugging Face evaluation metrics library
- **sweetviz**: Automated EDA and data profiling

### Model Configuration
The notebook uses TabPFNMix with specific configuration:
- Model path: `autogluon/tabpfn-mix-1.0-regressor`
- Evaluation metric: ROC-AUC
- Binary classification task (cancelled/not cancelled)

## Important Considerations

- This is an educational repository for a LinkedIn Learning course
- The main instructional content is in the external Colab notebook
- Local files primarily contain sample datasets for the course exercises
- When adding new functionality, ensure it aligns with the educational goals of teaching Hugging Face model evaluation