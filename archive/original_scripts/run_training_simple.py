#!/usr/bin/env python
"""
Simplified training script that uses AutoGluon's built-in evaluation
Bypasses the evaluate library import issue
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("✈️  Flight Cancellation Predictor Training (Simplified)", flush=True)
print("=" * 50, flush=True)

print("\n📦 Importing libraries...", flush=True)

import pandas as pd
print("  ✓ pandas imported", flush=True)

import numpy as np
print("  ✓ numpy imported", flush=True)

import holidays
from datetime import timedelta
print("  ✓ holidays imported", flush=True)

from sklearn.model_selection import train_test_split
print("  ✓ sklearn imported", flush=True)

from autogluon.tabular import TabularPredictor
print("  ✓ AutoGluon imported", flush=True)

print("✅ All imports successful!\n", flush=True)

def main():
    """Main training pipeline"""

    # Load data
    print("📊 Loading data files...", flush=True)

    print("  Loading flight data A...", flush=True)
    df_A = pd.read_parquet("T_ONTIME_REPORTING11_A.parquet")
    print(f"  ✓ Loaded {len(df_A):,} records", flush=True)

    print("  Loading flight data B...", flush=True)
    df_B = pd.read_parquet("T_ONTIME_REPORTING11_B.parquet")
    print(f"  ✓ Loaded {len(df_B):,} records", flush=True)

    # Combine data
    df = pd.concat([df_A, df_B], ignore_index=True)
    print(f"  ✓ Combined: {len(df):,} total records", flush=True)

    # Load airport data
    print("  Loading airport data...", flush=True)
    arp = pd.read_excel("ARP-NPIAS-2025-2029-AppendixA.xlsx")
    print(f"  ✓ Loaded {len(arp):,} airport records", flush=True)

    # Merge
    df = pd.merge(df, arp, how='left', left_on='ORIGIN', right_on='LocID')
    df.columns = df.columns.str.lower()

    print(f"\n📊 Dataset ready: {len(df):,} flight records", flush=True)

    # Basic feature engineering
    print("\n🔧 Engineering features...", flush=True)

    # Data types
    df['cancelled'] = df['cancelled'].astype('category')

    # Extract hour from times
    df['scheduled_hour_of_departure'] = df['crs_dep_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)
    df['scheduled_hour_of_arrival'] = df['crs_arr_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)

    # Simple indicators
    df['ind_is_summer'] = np.where((df['month'] >= 6) & (df['month'] <= 8), 1, 0)
    df['ind_early_departure'] = np.where(df['dep_delay'] < 0, 1, 0)

    print("  ✓ Features created", flush=True)

    # Select features
    features = [
        'year', 'month', 'day_of_month', 'day_of_week',
        'scheduled_hour_of_departure', 'scheduled_hour_of_arrival',
        'origin', 'dest', 'op_unique_carrier',
        'dep_delay', 'dep_delay_new', 'dep_del15',
        'distance', 'crs_elapsed_time',
        'cancelled',  # Target
        'ind_is_summer', 'ind_early_departure'
    ]

    # Keep only available features
    available_features = [f for f in features if f in df.columns]
    df = df[available_features]

    # Sample for faster training
    print(f"\n📌 Sampling 5,000 records for quick demo...", flush=True)
    df = df.sample(n=5000, random_state=123)

    # Check class balance
    cancellation_rate = df['cancelled'].value_counts(normalize=True).get(1.0, 0)
    print(f"✈️  Cancellation rate: {cancellation_rate:.2%}", flush=True)

    # Split data
    train, test = train_test_split(df, test_size=0.4, random_state=123)
    print(f"\n📊 Train: {len(train):,} | Test: {len(test):,}", flush=True)

    # Configure TabPFNMix
    print("\n🚀 Training TabPFNMix model...", flush=True)
    print("  This may take 2-5 minutes...", flush=True)

    tabpfnmix_cfg = {
        "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
        "n_ensembles": 1,
        "max_epochs": 3
    }

    hyperparameters = {
        "TABPFNMIX": [tabpfnmix_cfg],
    }

    # Train model
    predictor = TabularPredictor(
        label='cancelled',
        eval_metric='roc_auc',
        verbosity=2  # Show training progress
    ).fit(
        train_data=train,
        tuning_data=test,
        hyperparameters=hyperparameters,
        presets=None,
        use_bag_holdout=True,
        num_bag_folds=0,
        num_bag_sets=0,
        num_stack_levels=0,
        ag_args_fit={"ag.max_memory_usage_ratio": 5}
    )

    print("\n✅ Training complete!", flush=True)

    # Evaluate using AutoGluon's leaderboard
    print("\n📈 Evaluating model...", flush=True)

    leaderboard = predictor.leaderboard(
        test,
        silent=True,
        extra_metrics=['roc_auc', 'recall', 'f1', 'precision', 'accuracy']
    )

    print("\n📊 Model Performance:", flush=True)
    print("=" * 50, flush=True)

    # Display key metrics
    best_model = leaderboard.iloc[0]
    print(f"Model: {best_model['model']}", flush=True)
    print(f"ROC-AUC: {best_model['roc_auc']:.4f}", flush=True)
    print(f"Accuracy: {best_model['accuracy']:.4f}", flush=True)
    print(f"Precision: {best_model['precision']:.4f}", flush=True)
    print(f"Recall: {best_model['recall']:.4f}", flush=True)
    print(f"F1 Score: {best_model['f1']:.4f}", flush=True)
    print(f"Training time: {best_model['fit_time']:.1f}s", flush=True)
    print(f"Inference time: {best_model['pred_time_test']:.1f}s", flush=True)

    # Custom threshold evaluation
    print("\n🎯 Applying custom threshold for rare events...", flush=True)

    pred_proba = predictor.predict_proba(test, model="TabPFNMix")
    threshold = 0.02  # Lower threshold for rare events
    custom_predictions = (pred_proba[1] > threshold).astype(int)

    # Calculate metrics with custom threshold
    from sklearn.metrics import classification_report
    report = classification_report(
        test['cancelled'],
        custom_predictions,
        output_dict=True
    )

    print(f"\nWith threshold = {threshold}:", flush=True)
    print(f"Precision: {report['1']['precision']:.4f}", flush=True)
    print(f"Recall: {report['1']['recall']:.4f}", flush=True)
    print(f"F1 Score: {report['1']['f1-score']:.4f}", flush=True)

    # Save model
    model_path = "./AutogluonModels/flight_cancellation_model"
    predictor.save(model_path)
    print(f"\n💾 Model saved to: {model_path}", flush=True)

    print("\n✅ All done! The model is ready for use.", flush=True)
    print("\n📝 Calibration Insights Applied:", flush=True)
    print("  • Custom threshold (0.02) for rare event detection", flush=True)
    print("  • Lower threshold captures more cancellations", flush=True)
    print("  • Trade-off: More false positives but fewer missed flights", flush=True)

    return predictor

if __name__ == "__main__":
    try:
        predictor = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)