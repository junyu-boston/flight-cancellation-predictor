# AutoGluon TabPFNMix Training Guide for sklearn/PyTorch Users

## Table of Contents
1. [Introduction to AutoGluon](#introduction-to-autogluon)
2. [TabPFNMix Model Overview](#tabpfnmix-model-overview)
3. [Environment Setup](#environment-setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Configuration](#model-configuration)
6. [Training the Model](#training-the-model)
7. [Prediction and Evaluation](#prediction-and-evaluation)
8. [Advanced Features](#advanced-features)
9. [Best Practices](#best-practices)

## Introduction to AutoGluon

AutoGluon is an AutoML framework that automates machine learning tasks. Think of it as a high-level wrapper that combines the simplicity of sklearn with the power of deep learning frameworks.

### Key Differences from sklearn/PyTorch:
- **sklearn**: Manual feature engineering, model selection, hyperparameter tuning
- **PyTorch**: Manual architecture design, training loops, optimization
- **AutoGluon**: Automated feature engineering, model selection, and ensemble creation

```python
# sklearn approach
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# PyTorch approach
import torch.nn as nn
class CustomModel(nn.Module):
    def __init__(self):
        # Define layers

# AutoGluon approach
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='target').fit(train_data)
```

## TabPFNMix Model Overview

TabPFNMix (Tabular Prior-Fitted Networks Mix) is a meta-learned model from Hugging Face that's been pre-trained on thousands of tabular datasets. It's particularly effective for small to medium-sized datasets.

### Key Characteristics:
- **Zero-shot learning**: Pre-trained on diverse tabular data
- **Fast inference**: No iterative training required
- **Ensemble approach**: Combines multiple TabPFN models
- **Automatic feature handling**: Works with mixed data types

## Environment Setup

```bash
# Using uv (recommended)
uv venv
uv pip install autogluon evaluate pandas numpy scikit-learn

# Or using pip
pip install autogluon evaluate pandas numpy scikit-learn
```

## Dataset Preparation

### Loading and Splitting Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('your_data.csv')

# AutoGluon works directly with DataFrames (unlike sklearn which needs X, y separation)
# No need to separate features and target!

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Alternative: AutoGluon can handle the split internally
# Just pass the full dataset and use validation parameters
```

### Key Differences from sklearn:
1. **No X, y separation needed**: AutoGluon works with complete DataFrames
2. **Automatic type inference**: Handles categoricals, numerics, dates automatically
3. **Missing value handling**: Built-in imputation strategies

```python
# sklearn way
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# AutoGluon way
train, test = train_test_split(df, test_size=0.2)  # Keep target in DataFrame!
```

## Model Configuration

### TabPFNMix Configuration

```python
from autogluon.tabular import TabularPredictor

# Configure TabPFNMix hyperparameters
tabpfnmix_cfg = {
    "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",  # HuggingFace model
    "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",
    "n_ensembles": 1,        # Number of ensemble members (1-10)
    "max_epochs": 3,         # Training iterations for ensemble
    "subsample_features": False,  # Whether to subsample features
    "use_py_boost": False,   # Use PyBoost for additional performance
}

# Create hyperparameters dictionary
hyperparameters = {
    "TABPFNMIX": [tabpfnmix_cfg],  # Can specify multiple configs for tuning
}
```

### Parameter Explanation:
- **model_path**: Points to the Hugging Face model hub
- **n_ensembles**: More ensembles = better accuracy but slower (like n_estimators in RandomForest)
- **max_epochs**: How many times to train the ensemble (not the base model, which is pre-trained)
- **subsample_features**: Useful for high-dimensional data

## Training the Model

### Basic Training

```python
# Initialize predictor
predictor = TabularPredictor(
    label='target_column',           # Name of target column
    eval_metric='roc_auc',          # Metric to optimize
    problem_type='binary'            # 'binary', 'multiclass', or 'regression'
)

# Fit the model
predictor.fit(
    train_data=train,
    tuning_data=test,                # Validation data for early stopping
    hyperparameters=hyperparameters,
    presets=None,                    # Use custom config instead of presets
    use_bag_holdout=True,            # Use bagging with holdout validation
    num_bag_folds=0,                 # Number of bagging folds (0 = no bagging)
    num_bag_sets=0,                  # Number of bagging sets
    num_stack_levels=0,              # Stacking levels (0 = no stacking)
    verbosity=4,                     # Logging level (0-4)
    ag_args_fit={
        "ag.max_memory_usage_ratio": 5  # Memory management
    }
)
```

### Understanding the Parameters:

#### Similar to sklearn concepts:
- **label**: Like `y` in sklearn
- **eval_metric**: Like `scoring` in GridSearchCV
- **train_data/tuning_data**: Like X_train/X_val split

#### New AutoGluon concepts:
- **use_bag_holdout**: Creates multiple models with different data samples
- **num_stack_levels**: Creates meta-models (ensemble of ensembles)
- **ag_args_fit**: Runtime configuration

### Comparison with sklearn/PyTorch:

```python
# sklearn equivalent
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')

# PyTorch equivalent
model = CustomModel()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(num_epochs):
    # Training loop

# AutoGluon - all of the above in one call!
predictor = TabularPredictor(label='target', eval_metric='roc_auc').fit(train_data)
```

## Prediction and Evaluation

### Making Predictions

```python
# Single model predictions (TabPFNMix only)
predictions = predictor.predict(test, model="TabPFNMix")

# Probability predictions for classification
pred_proba = predictor.predict_proba(test, model="TabPFNMix")

# Get probabilities for each class
pred_proba_class_0 = pred_proba[0]  # Probability of class 0
pred_proba_class_1 = pred_proba[1]  # Probability of class 1

# Custom threshold (for binary classification)
threshold = 0.3
custom_predictions = (pred_proba[1] > threshold).astype(int)
```

### Model Evaluation

```python
# AutoGluon's built-in evaluation
leaderboard = predictor.leaderboard(
    test,
    silent=True,
    extra_metrics=['roc_auc', 'recall', 'f1', 'precision', 'accuracy']
)

# Using evaluate library (Hugging Face)
import evaluate

# Load metric
auc_metric = evaluate.load("roc_auc")

# Compute metric
results = auc_metric.compute(
    prediction_scores=pred_proba[1].tolist(),
    references=test['target'].tolist()
)
```

### Feature Importance

```python
# Similar to sklearn's feature_importances_
importance = predictor.feature_importance(test)

# Permutation importance (like sklearn.inspection.permutation_importance)
importance_df = predictor.feature_importance(
    test.head(1000),  # Use subset for speed
    num_shuffle_sets=5  # Number of permutations
)
```

## Advanced Features

### 1. Model Persistence

```python
# Save model
predictor.save("./models/my_model")

# Load model
loaded_predictor = TabularPredictor.load("./models/my_model")

# Make predictions with specific model
predictions = loaded_predictor.predict(test, model="TabPFNMix")
```

### 2. Custom Evaluation Metrics

```python
# Define custom metric (similar to sklearn's make_scorer)
def custom_metric(y_true, y_pred):
    # Your custom logic
    return score

# Use in training
predictor = TabularPredictor(
    label='target',
    eval_metric=custom_metric
).fit(train_data)
```

### 3. Ensemble Control

```python
# Multiple model configurations
hyperparameters = {
    "TABPFNMIX": [
        {"n_ensembles": 1, "max_epochs": 3},
        {"n_ensembles": 5, "max_epochs": 5},
    ],
    "GBM": {},  # Add other models
    "NN_TORCH": {},
}

# AutoGluon will train all models and create an ensemble
predictor.fit(train_data, hyperparameters=hyperparameters)
```

### 4. Memory and Performance Optimization

```python
# Resource constraints
predictor.fit(
    train_data,
    time_limit=3600,  # 1 hour time limit
    ag_args_fit={
        "ag.max_memory_usage_ratio": 0.8,  # Use 80% of available memory
        "num_cpus": 8,  # Limit CPU cores
        "num_gpus": 1,  # Use GPU if available
    }
)
```

## Best Practices

### 1. Data Preprocessing

```python
# AutoGluon handles most preprocessing, but you can help:

# Convert obvious categoricals
df['category_col'] = df['category_col'].astype('category')

# Create date features if needed
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Handle high cardinality categoricals
high_card_threshold = 100
for col in df.select_dtypes(include=['object']).columns:
    if df[col].nunique() > high_card_threshold:
        # Consider encoding or dropping
        pass
```

### 2. Model Selection Strategy

```python
# Quick prototype (like sklearn's DummyClassifier)
quick_predictor = TabularPredictor(label='target').fit(
    train_data,
    presets='medium_quality',
    time_limit=120
)

# Production model
production_predictor = TabularPredictor(label='target').fit(
    train_data,
    hyperparameters=hyperparameters,
    num_bag_folds=5,
    num_stack_levels=1
)
```

### 3. Debugging and Monitoring

```python
# High verbosity for debugging
predictor.fit(train_data, verbosity=4)

# Check what happened during training
print(predictor.fit_summary())

# Get detailed model info
model_info = predictor.info()
```

## Common Patterns

### Binary Classification with Custom Threshold

```python
# Train model
predictor = TabularPredictor(
    label='cancelled',
    eval_metric='roc_auc',
    problem_type='binary'
).fit(
    train_data=train,
    hyperparameters={"TABPFNMIX": [tabpfnmix_cfg]}
)

# Get probabilities
pred_proba = predictor.predict_proba(test, model="TabPFNMix")

# Apply custom threshold
threshold = 0.02  # Adjust based on precision/recall needs
predictions = (pred_proba[1] > threshold).astype(int)

# Add to DataFrame
test['pred'] = predictions
test['pred_proba_0'] = pred_proba[0]
test['pred_proba_1'] = pred_proba[1]
```

### Regression with TabPFNMix

```python
# Configure for regression
tabpfnmix_regression = {
    "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
    "n_ensembles": 3,
    "max_epochs": 5
}

# Train regressor
predictor = TabularPredictor(
    label='price',
    eval_metric='rmse',  # or 'mae', 'r2'
    problem_type='regression'
).fit(
    train_data=train,
    hyperparameters={"TABPFNMIX": [tabpfnmix_regression]}
)

# Get predictions
predictions = predictor.predict(test)
```

## Migration Tips from sklearn/PyTorch

### sklearn Users:
1. **Pipeline → AutoGluon handles it**: No need for sklearn Pipelines
2. **GridSearchCV → hyperparameters dict**: Define multiple configs
3. **cross_val_score → num_bag_folds**: Built-in cross-validation
4. **make_scorer → eval_metric**: Pass function directly

### PyTorch Users:
1. **DataLoader → DataFrame**: Work with pandas directly
2. **Training loop → .fit()**: One function call
3. **model.eval() → predictor.predict()**: Automatic inference mode
4. **Optimizer → Handled internally**: No manual optimization

## Troubleshooting

### Common Issues and Solutions:

```python
# Out of memory
predictor.fit(
    train_data,
    ag_args_fit={"ag.max_memory_usage_ratio": 0.5}  # Reduce memory usage
)

# Slow training
hyperparameters = {
    "TABPFNMIX": [{"n_ensembles": 1, "max_epochs": 1}]  # Reduce complexity
}

# Poor performance
# Try ensemble with multiple models
hyperparameters = {
    "TABPFNMIX": {},
    "GBM": {},
    "CAT": {},
    "XGB": {}
}
```

## Complete Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
import evaluate

# Load data
df = pd.read_parquet('flight_data.parquet')

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Configure TabPFNMix
tabpfnmix_cfg = {
    "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
    "n_ensembles": 3,
    "max_epochs": 5
}

# Train model
predictor = TabularPredictor(
    label='cancelled',
    eval_metric='roc_auc'
).fit(
    train_data=train,
    tuning_data=test,
    hyperparameters={"TABPFNMIX": [tabpfnmix_cfg]},
    verbosity=2
)

# Evaluate
leaderboard = predictor.leaderboard(test)
print(leaderboard)

# Save model
predictor.save('./models/flight_cancellation_model')
```

## Next Steps

1. **Experiment with presets**: Try `presets='best_quality'` for competitions
2. **Add more models**: Include GBM, NN_TORCH for better ensembles
3. **Feature engineering**: Although AutoGluon handles most, domain features help
4. **Hyperparameter tuning**: Use multiple configs in hyperparameters dict
5. **Production deployment**: Use `predictor.save()` and load for inference

## Resources

- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [TabPFN Paper](https://arxiv.org/abs/2207.01848)
- [Hugging Face Model Hub](https://huggingface.co/autogluon)