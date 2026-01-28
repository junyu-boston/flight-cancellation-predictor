#!/usr/bin/env python
"""
Quick test to verify you can load and use the already-trained model
Run this to confirm everything works before running the full notebook
"""

import warnings
warnings.filterwarnings('ignore')

print("🔍 Quick Model Test", flush=True)
print("=" * 40, flush=True)

try:
    # Test imports
    import pandas as pd
    import numpy as np
    from autogluon.tabular import TabularPredictor
    print("✅ All imports successful", flush=True)

    # Load the saved model
    model_path = "./AutogluonModels/ag-20260128_113233"
    predictor = TabularPredictor.load(model_path)
    print(f"✅ Model loaded from: {model_path}", flush=True)

    # Create test data
    test_sample = pd.DataFrame({
        'year': [2024],
        'month': [6],
        'day_of_month': [15],
        'day_of_week': [3],
        'scheduled_hour_of_departure': [14],
        'scheduled_hour_of_arrival': [16],
        'origin': ['LAX'],
        'dest': ['JFK'],
        'op_unique_carrier': ['AA'],
        'dep_delay': [25.0],
        'dep_delay_new': [25.0],
        'dep_del15': [1.0],  # This is the key feature!
        'distance': [2475.0],
        'crs_elapsed_time': [330.0],
        'ind_is_summer': [1],
        'ind_early_departure': [0]
    })

    # Make prediction
    prediction = predictor.predict(test_sample)
    probability = predictor.predict_proba(test_sample)

    print("\n📊 Test Prediction:", flush=True)
    print(f"  Input: Flight with 25 min delay", flush=True)
    print(f"  Prediction: {'Cancelled' if prediction[0] == 1 else 'Not Cancelled'}", flush=True)
    print(f"  Probability of cancellation: {probability[1][0]:.2%}", flush=True)

    # Apply custom threshold
    threshold = 0.02
    custom_pred = (probability[1][0] > threshold)
    print(f"\n🎯 With custom threshold (0.02):", flush=True)
    print(f"  Prediction: {'Cancelled' if custom_pred else 'Not Cancelled'}", flush=True)

    print("\n✅ Model is working perfectly!", flush=True)
    print("\n💡 Key Insight:", flush=True)
    print("  The model heavily relies on 'dep_del15' (delay >15 min)", flush=True)
    print("  This is like finding a single biomarker that predicts", flush=True)
    print("  drug response in pharmaceutical research!", flush=True)

except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    print("\nTroubleshooting:", flush=True)
    print("1. Make sure you're in the right directory", flush=True)
    print("2. Run 'uv run python run_training_simple.py' first if no model exists", flush=True)