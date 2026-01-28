#!/usr/bin/env python
"""
Simple script to run the flight cancellation prediction model training
Can be run directly with: uv run python run_training.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("📦 Importing libraries...", flush=True)

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

import evaluate
print("  ✓ evaluate imported", flush=True)

import matplotlib.pyplot as plt
import seaborn as sns
print("  ✓ visualization libraries imported", flush=True)
print("✅ All imports successful!\n", flush=True)

def load_and_prepare_data():
    """Load and prepare the flight data"""
    print("📊 Loading data files...")

    # Load parquet files
    df_A = pd.read_parquet("T_ONTIME_REPORTING11_A.parquet")
    df_B = pd.read_parquet("T_ONTIME_REPORTING11_B.parquet")

    # Stack the dataframes
    df = pd.concat([df_A, df_B], ignore_index=True)

    # Load airport data
    arp = pd.read_excel("ARP-NPIAS-2025-2029-AppendixA.xlsx")

    # Join with airport data
    df = pd.merge(df, arp, how='left', left_on='ORIGIN', right_on='LocID')

    # Lowercase column names
    df.columns = df.columns.str.lower()

    print(f"✅ Loaded {len(df)} flight records")
    return df

def add_holiday_features(df):
    """Add holiday-related features"""
    print("🎄 Adding holiday features...")

    # Initialize US holidays
    US_holidays = holidays.US(years=[2023, 2024, 2025])

    # Convert flight date to datetime
    df['fl_date_dt'] = pd.to_datetime(df['fl_date'])

    # Check if date is a holiday
    df['ind_is_holiday'] = df['fl_date_dt'].apply(
        lambda x: 1.0 if US_holidays.get(x.date()) is not None else 0.0
    )

    # Check if near a holiday (within 3 days)
    df['ind_is_near_holiday_down'] = df['fl_date_dt'].apply(
        lambda x: any((x.date() - timedelta(days=i)) in US_holidays for i in range(3))
    )
    df['ind_is_near_holiday_up'] = df['fl_date_dt'].apply(
        lambda x: any((x.date() + timedelta(days=i)) in US_holidays for i in range(3))
    )

    df['ind_is_near_holiday'] = np.where(
        (df['ind_is_near_holiday_down'] == True) | (df['ind_is_near_holiday_up'] == True),
        1.0, 0.0
    )

    return df

def engineer_features(df):
    """Add engineered features"""
    print("🔧 Engineering features...")

    # Change data types
    df['cancelled'] = df['cancelled'].astype('category')
    df['op_unique_carrier'] = df['op_unique_carrier'].astype('string')

    # Extract hour from departure/arrival times
    df['scheduled_hour_of_departure'] = df['crs_dep_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)
    df['scheduled_hour_of_arrival'] = df['crs_arr_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)

    # Indicator variables
    df['ind_is_summer'] = np.where((df['month'] >= 6) & (df['month'] <= 8), 1, 0)
    df['ind_late_scheduled_hour_of_departure'] = np.where(
        (df['scheduled_hour_of_departure'] >= 20) & (df['scheduled_hour_of_departure'] <= 22), 1, 0
    )
    df['ind_early_departure'] = np.where(df['dep_delay'] < 0, 1, 0)
    df['ind_dest_TX_long_dist'] = np.where(
        (df['dest_state_abr'] == 'TX') & (df['distance'] >= 1000), 1, 0
    )

    return df

def select_features(df):
    """Select relevant features for modeling"""
    features = [
        # Date/time features
        'year', 'quarter', 'month', 'day_of_month', 'day_of_week',
        'scheduled_hour_of_departure', 'scheduled_hour_of_arrival', 'fl_date_dt',

        # Flight features
        'op_unique_carrier', 'origin', 'dest',
        'dep_delay', 'dep_delay_new', 'dep_del15', 'dep_delay_group',
        'taxi_in', 'cancelled',  # Target variable
        'crs_elapsed_time', 'flights', 'distance', 'distance_group',
        'carrier_delay', 'weather_delay', 'nas_delay', 'late_aircraft_delay',

        # Holiday features
        'ind_is_holiday', 'ind_is_near_holiday',

        # Other indicators
        'ind_is_summer', 'ind_late_scheduled_hour_of_departure',
        'ind_late_scheduled_hour_of_arrival', 'ind_early_departure',
        'ind_dest_TX_long_dist'
    ]

    # Keep only features that exist in the dataframe
    available_features = [f for f in features if f in df.columns]
    return df[available_features]

def train_model(train_data, test_data):
    """Train the TabPFNMix model"""
    print("🚀 Training TabPFNMix model...")

    # Configure TabPFNMix
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
        eval_metric='roc_auc'
    ).fit(
        train_data=train_data,
        tuning_data=test_data,
        hyperparameters=hyperparameters,
        presets=None,
        use_bag_holdout=True,
        num_bag_folds=0,
        num_bag_sets=0,
        num_stack_levels=0,
        verbosity=2,
        ag_args_fit={"ag.max_memory_usage_ratio": 5}
    )

    return predictor

def evaluate_model(predictor, test_data):
    """Evaluate the trained model"""
    print("📈 Evaluating model...")

    # Get predictions
    predictions = predictor.predict(test_data, model="TabPFNMix")
    pred_proba = predictor.predict_proba(test_data, model="TabPFNMix")

    # Apply custom threshold
    threshold = 0.02
    custom_predictions = (pred_proba[1] > threshold).astype(int)

    # Calculate metrics
    leaderboard = predictor.leaderboard(
        test_data,
        silent=True,
        extra_metrics=['roc_auc', 'recall', 'f1', 'precision', 'accuracy']
    )

    print("\n📊 Model Performance:")
    print(leaderboard)

    # Use Hugging Face evaluate
    auc_metric = evaluate.load("roc_auc")
    auc_result = auc_metric.compute(
        prediction_scores=pred_proba[1].tolist(),
        references=test_data['cancelled'].tolist()
    )

    print(f"\n🎯 ROC-AUC Score: {auc_result['roc_auc']:.4f}")

    return predictions, pred_proba

def main():
    """Main training pipeline"""
    print("✈️  Flight Cancellation Predictor Training")
    print("=" * 50)

    # Load and prepare data
    df = load_and_prepare_data()

    # Add features
    df = add_holiday_features(df)
    df = engineer_features(df)
    df = select_features(df)

    # Sample for faster training (remove this for full training)
    print("📌 Sampling 5000 records for quick training...")
    df = df.sample(n=5000, random_state=123)

    # Split data
    train, test = train_test_split(df, test_size=0.4, train_size=0.6, random_state=123)
    print(f"📊 Train size: {len(train)}, Test size: {len(test)}")

    # Check class balance
    print(f"✈️  Cancellation rate: {df['cancelled'].value_counts(normalize=True)[1.0]:.2%}")

    # Train model
    predictor = train_model(train, test)

    # Evaluate model
    predictions, pred_proba = evaluate_model(predictor, test)

    # Save model
    model_path = "./AutogluonModels/flight_cancellation_model"
    predictor.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")

    print("\n✅ Training complete!")

    return predictor

if __name__ == "__main__":
    predictor = main()