# AutoGluon Leaderboard vs Hugging Face Evaluate Library: Complete Guide

## Overview

When evaluating models in AutoGluon, you have two powerful options: the built-in **leaderboard** and the Hugging Face **Evaluate library**. Each serves different purposes and excels in different scenarios.

## Quick Comparison

| Feature | AutoGluon Leaderboard | Evaluate Library |
|---------|----------------------|------------------|
| **Scope** | AutoGluon models only | Any model/framework |
| **Metrics** | Standard ML metrics + runtime | 100+ metrics across domains |
| **Output** | Comprehensive DataFrame | Metric values only |
| **Runtime Info** | ✅ Yes (fit_time, pred_time) | ❌ No |
| **Independence** | Requires AutoGluon predictor | Standalone |
| **Domains** | Tabular focus | Multi-domain (NLP, CV, Audio) |
| **Custom Metrics** | Limited | Extensive |

## When to Use Each

### Use AutoGluon Leaderboard When:

- Working within AutoGluon ecosystem
- Need runtime performance metrics
- Comparing multiple AutoGluon models
- Want comprehensive model metadata
- Need quick model selection

### Use Evaluate Library When:

- Working across multiple frameworks
- Need specialized metrics (BLEU, ROUGE, etc.)
- Evaluating non-tabular models
- Creating custom evaluation pipelines
- Need framework-agnostic evaluation

## Detailed Comparison

### 1. AutoGluon Leaderboard

The leaderboard provides a complete picture of model performance INCLUDING operational metrics.

```python
from autogluon.tabular import TabularPredictor

# Train model
predictor = TabularPredictor(label='target').fit(train_data)

# Get comprehensive evaluation
leaderboard = predictor.leaderboard(
    test_data,
    silent=True,
    extra_metrics=['roc_auc', 'recall', 'f1', 'precision', 'accuracy']
)

# What you get:
# - score_test: Primary metric score
# - All extra_metrics values
# - pred_time_test: Inference time
# - fit_time: Training time
# - pred_time_test_marginal: Additional time for this model
# - stack_level: Ensemble depth
# - can_infer: Whether model can make predictions
```

#### Leaderboard Output Example:

```
                 model  score_test   roc_auc  recall    f1  precision  accuracy  pred_time_test  fit_time
0            TabPFNMix    0.949398  0.949398   0.818  0.90        1.0     0.998        35.52     258.48
1                  GBM    0.943201  0.943201   0.800  0.88        0.95    0.995        12.34     145.23
2  WeightedEnsemble_L2    0.952145  0.952145   0.825  0.91        0.98    0.997        47.86     403.71
```

### 2. Hugging Face Evaluate Library

The Evaluate library is your Swiss Army knife for evaluation - versatile, independent, and comprehensive.

```python
import evaluate

# Load any metric
accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
perplexity = evaluate.load("perplexity")

# Works with ANY model output
predictions = model.predict(test_data)  # Can be sklearn, PyTorch, TF, anything!

# Calculate metrics
results = accuracy.compute(
    predictions=predictions,
    references=true_labels
)
```

#### Key Advantages:

1. **Framework Independence**
```python
# Works with sklearn
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Works with PyTorch
import torch
torch_model = MyModel()
torch_preds = torch_model(X_test)

# Works with TensorFlow
import tensorflow as tf
tf_model = tf.keras.Sequential([...])
tf_preds = tf_model.predict(X_test)

# Evaluate ALL with same library!
accuracy = evaluate.load("accuracy")
for preds in [rf_preds, torch_preds, tf_preds]:
    score = accuracy.compute(predictions=preds, references=y_test)
    print(score)
```

2. **Multi-Domain Support**
```python
# NLP Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# Computer Vision
mean_iou = evaluate.load("mean_iou")
psnr = evaluate.load("psnr")

# Audio
wer = evaluate.load("wer")  # Word Error Rate
cer = evaluate.load("cer")  # Character Error Rate

# Multimodal
clip_score = evaluate.load("clip_score")
```

3. **Custom Metrics**
```python
# Define custom metric
def custom_business_metric(predictions, references):
    # Your custom logic
    # E.g., weighted by business impact
    tp = sum((p == 1 and r == 1) for p, r in zip(predictions, references))
    fp = sum((p == 1 and r == 0) for p, r in zip(predictions, references))

    # Business rule: false positives cost 10x more
    business_score = tp - (10 * fp)
    return {"business_score": business_score}

# Use with Evaluate
custom_metric = evaluate.Metric(
    compute=custom_business_metric,
    name="business_metric"
)
```

## Practical Examples

### Example 1: Complete AutoGluon Evaluation

```python
# Train multiple models
predictor = TabularPredictor(label='cancelled').fit(
    train_data,
    hyperparameters={
        'GBM': {},
        'TABPFNMIX': {},
        'NN_TORCH': {}
    }
)

# Get leaderboard for model selection
leaderboard = predictor.leaderboard(test_data)
best_model = leaderboard.iloc[0]['model']
print(f"Best model: {best_model}")

# But also get specialized metrics with Evaluate
from evaluate import load

# For detailed classification analysis
precision_metric = load("precision")
recall_metric = load("recall")

predictions = predictor.predict(test_data, model=best_model)
precision = precision_metric.compute(
    predictions=predictions,
    references=test_data['cancelled'],
    average='weighted'
)
recall = recall_metric.compute(
    predictions=predictions,
    references=test_data['cancelled'],
    average='weighted'
)
```

### Example 2: Cross-Framework Comparison

```python
import evaluate
from sklearn.ensemble import RandomForestClassifier
from autogluon.tabular import TabularPredictor

# Train models in different frameworks
# AutoGluon
ag_predictor = TabularPredictor(label='target').fit(train_data)
ag_preds = ag_predictor.predict(test_data)

# Sklearn
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Use Evaluate library for fair comparison
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

# Compare both
ag_accuracy = accuracy.compute(predictions=ag_preds, references=y_test)
rf_accuracy = accuracy.compute(predictions=rf_preds, references=y_test)

print(f"AutoGluon Accuracy: {ag_accuracy['accuracy']:.4f}")
print(f"RandomForest Accuracy: {rf_accuracy['accuracy']:.4f}")
```

### Example 3: Production Monitoring

```python
# Use leaderboard for model selection and baseline
leaderboard = predictor.leaderboard(test_data)
production_model = leaderboard.iloc[0]['model']
baseline_metrics = leaderboard.iloc[0]

# Use Evaluate for ongoing monitoring
import evaluate
from datetime import datetime

roc_auc = evaluate.load("roc_auc")

def monitor_model_performance(predictor, new_data, model_name):
    """Monitor model in production"""
    # Get predictions
    predictions = predictor.predict_proba(new_data, model=model_name)

    # Calculate current metrics
    current_auc = roc_auc.compute(
        prediction_scores=predictions[1],
        references=new_data['target']
    )

    # Log with timestamp
    log_entry = {
        'timestamp': datetime.now(),
        'auc': current_auc['roc_auc'],
        'n_samples': len(new_data)
    }

    return log_entry
```

## Best Practices

### 1. Use Both Together

```python
# Use leaderboard for initial model selection
leaderboard = predictor.leaderboard(test_data)
best_model = leaderboard.iloc[0]['model']

# Use Evaluate for detailed analysis
import evaluate

# Get detailed confusion matrix
confusion = evaluate.load("confusion_matrix")
predictions = predictor.predict(test_data, model=best_model)
cm = confusion.compute(predictions=predictions, references=test_data['target'])

# Get classification report
classification = evaluate.load("precision_recall_f1_micro")
report = classification.compute(predictions=predictions, references=test_data['target'])
```

### 2. Runtime vs Accuracy Trade-offs

```python
# Leaderboard shows runtime - crucial for production
leaderboard_df = predictor.leaderboard(test_data)

# Find best model under latency constraint
max_latency_ms = 100  # 100ms constraint
valid_models = leaderboard_df[leaderboard_df['pred_time_test'] < max_latency_ms]
best_fast_model = valid_models.iloc[0]['model']

# Verify quality with Evaluate
accuracy = evaluate.load("accuracy")
fast_predictions = predictor.predict(test_data, model=best_fast_model)
fast_accuracy = accuracy.compute(
    predictions=fast_predictions,
    references=test_data['target']
)
```

### 3. Custom Business Metrics

```python
# Leaderboard for standard metrics
leaderboard = predictor.leaderboard(test_data)

# Evaluate for business metrics
def revenue_impact_score(predictions, references, revenue_per_customer=100):
    """Custom metric considering business impact"""
    tp = sum((p == 1 and r == 1) for p, r in zip(predictions, references))
    fn = sum((p == 0 and r == 1) for p, r in zip(predictions, references))

    # Missed opportunities cost
    missed_revenue = fn * revenue_per_customer
    captured_revenue = tp * revenue_per_customer

    return {
        "captured_revenue": captured_revenue,
        "missed_revenue": missed_revenue,
        "revenue_score": captured_revenue / (captured_revenue + missed_revenue)
    }

# Apply to best model
best_model = leaderboard.iloc[0]['model']
predictions = predictor.predict(test_data, model=best_model)
business_metrics = revenue_impact_score(predictions, test_data['target'])
```

## Summary

### Leaderboard Strengths
- ✅ One-line comprehensive evaluation
- ✅ Runtime and resource metrics
- ✅ Easy model comparison
- ✅ AutoGluon integration
- ✅ Automatic ensemble evaluation

### Evaluate Library Strengths
- ✅ Framework agnostic
- ✅ 100+ built-in metrics
- ✅ Multi-domain support (NLP, CV, Audio)
- ✅ Custom metric creation
- ✅ Standardized API across all metrics

### Combined Workflow

```python
# 1. Train with AutoGluon
predictor = TabularPredictor(label='target').fit(train_data)

# 2. Use leaderboard for model selection
leaderboard = predictor.leaderboard(test_data)
best_model = leaderboard.iloc[0]['model']
print(f"Selected: {best_model} with {leaderboard.iloc[0]['score_test']:.4f}")

# 3. Use Evaluate for detailed/custom metrics
import evaluate

# Standard metrics
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

# Get predictions
preds = predictor.predict(test_data, model=best_model)

# Calculate all metrics
metrics = {
    'f1': f1.compute(predictions=preds, references=test_data['target']),
    'precision': precision.compute(predictions=preds, references=test_data['target']),
    'recall': recall.compute(predictions=preds, references=test_data['target'])
}

# 4. Production decision based on both
if (leaderboard.iloc[0]['pred_time_test'] < 100 and  # Fast enough
    metrics['precision']['precision'] > 0.95):       # Accurate enough
    print("✅ Model approved for production")
```

This combined approach gives you the best of both worlds: AutoGluon's comprehensive model management and Evaluate's flexible, extensive metric capabilities.