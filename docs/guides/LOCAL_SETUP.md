# Local Environment Setup Guide

## ✅ Setup Complete!

Your environment is now ready to run the flight cancellation prediction model locally using AutoGluon and TabPFNMix.

## 📁 Files Created

1. **`course_notebook_local.ipynb`** - Adapted Jupyter notebook for local execution
2. **`run_training.py`** - Standalone Python script to run the training
3. **`autogluon_tabpfnmix_guide.md`** - Comprehensive guide for AutoGluon/TabPFNMix
4. **`convert_notebook.py`** - Utility to convert notebooks from Colab to local

## 🚀 Quick Start

### Option 1: Run as Jupyter Notebook
```bash
# Start Jupyter with uv
uv run jupyter notebook course_notebook_local.ipynb
```

### Option 2: Run as Python Script
```bash
# Run the training script directly
uv run python run_training.py
```

### Option 3: Interactive Python
```bash
# Start Python REPL
uv run python

# Then import and use
>>> from run_training import main
>>> predictor = main()
```

## 🛠️ Environment Details

### Installed Packages
- **AutoGluon** - Complete AutoML framework with TabPFNMix
- **pandas, numpy** - Data manipulation
- **holidays** - Holiday feature engineering
- **matplotlib, seaborn** - Visualization
- **evaluate** - Hugging Face evaluation metrics
- **jupyter** - Notebook environment

### Virtual Environment
- Created with: `uv venv`
- Python version: 3.11.13
- Location: `.venv/`

## 📊 Data Files Required

Make sure these files are in the current directory:
- `T_ONTIME_REPORTING11_A.parquet` ✓
- `T_ONTIME_REPORTING11_B.parquet` ✓
- `ARP-NPIAS-2025-2029-AppendixA.xlsx` ✓

## 🎯 Key Features

### What's Different from Colab

1. **No pip installs** - All dependencies pre-installed in venv
2. **Local file paths** - Data loads from current directory
3. **Local model saves** - Models saved to `./AutogluonModels/`
4. **Optimized for local** - Memory settings adjusted for local machine

### Training Features

- **TabPFNMix Model** - Pre-trained Hugging Face model
- **Automatic Feature Engineering** - Holiday detection, time features
- **Custom Thresholds** - Adjustable decision threshold (default: 0.02)
- **Comprehensive Evaluation** - ROC-AUC, precision, recall, F1

## 📖 Learning Resources

### For sklearn/PyTorch Users

Read `autogluon_tabpfnmix_guide.md` for:
- Conceptual mapping from sklearn/PyTorch to AutoGluon
- Detailed parameter explanations
- Code comparisons and migration tips
- Best practices and troubleshooting

### Key Concepts

```python
# sklearn way
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# AutoGluon way
train, test = train_test_split(df)  # Keep target in DataFrame!
predictor = TabularPredictor(label='target').fit(train)
```

## 🔧 Customization

### Adjust Training Parameters

Edit `run_training.py`:
```python
# Change sample size (line ~180)
df = df.sample(n=10000, random_state=123)  # Increase for better accuracy

# Change model config (line ~130)
tabpfnmix_cfg = {
    "n_ensembles": 5,  # More ensembles
    "max_epochs": 10   # More training
}
```

### Change Evaluation Metrics

```python
# In train_model function
predictor = TabularPredictor(
    label='cancelled',
    eval_metric='f1'  # Change from 'roc_auc'
)
```

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce memory usage in ag_args_fit
ag_args_fit={"ag.max_memory_usage_ratio": 2}  # Lower value
```

### Slow Training
- Reduce sample size
- Use fewer ensembles
- Disable bagging

### Import Errors
```bash
# Verify environment is activated
which python  # Should show .venv/bin/python

# Reinstall if needed
uv pip install autogluon --upgrade
```

## 📈 Expected Results

With the sample of 5000 records:
- **Training time**: 2-5 minutes
- **ROC-AUC**: ~0.94-0.95
- **Precision**: ~0.9-1.0
- **Recall**: ~0.8-0.9

## 🎉 Next Steps

1. **Full Dataset Training**: Remove the sampling to train on all 600K+ records
2. **Hyperparameter Tuning**: Try different n_ensembles and max_epochs
3. **Add More Models**: Include GBM, XGBoost for ensemble
4. **Feature Engineering**: Add domain-specific features
5. **Production Deployment**: Use saved model for inference

## 💡 Tips

- Start with small samples for quick iteration
- Use verbosity=4 for debugging
- Monitor memory usage with system tools
- Save intermediate results frequently

Happy modeling! 🚀