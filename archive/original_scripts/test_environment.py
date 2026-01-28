#!/usr/bin/env python
"""Quick test to verify environment and AutoGluon setup"""

import sys
print("🔍 Testing environment...", flush=True)

# Test imports
try:
    import pandas as pd
    print("✅ pandas imported", flush=True)
except ImportError as e:
    print(f"❌ pandas import failed: {e}", flush=True)
    sys.exit(1)

try:
    import numpy as np
    print("✅ numpy imported", flush=True)
except ImportError as e:
    print(f"❌ numpy import failed: {e}", flush=True)
    sys.exit(1)

try:
    from autogluon.tabular import TabularPredictor
    print("✅ AutoGluon imported", flush=True)
except ImportError as e:
    print(f"❌ AutoGluon import failed: {e}", flush=True)
    sys.exit(1)

try:
    import evaluate
    print("✅ evaluate library imported", flush=True)
except ImportError as e:
    print(f"❌ evaluate import failed: {e}", flush=True)
    sys.exit(1)

# Test data loading
print("\n📊 Testing data loading...", flush=True)
try:
    # Test loading parquet
    df_A = pd.read_parquet("T_ONTIME_REPORTING11_A.parquet")
    print(f"✅ Loaded parquet file A: {len(df_A)} rows", flush=True)

    df_B = pd.read_parquet("T_ONTIME_REPORTING11_B.parquet")
    print(f"✅ Loaded parquet file B: {len(df_B)} rows", flush=True)

    # Test Excel loading
    arp = pd.read_excel("ARP-NPIAS-2025-2029-AppendixA.xlsx")
    print(f"✅ Loaded Excel file: {len(arp)} rows", flush=True)

except Exception as e:
    print(f"❌ Data loading failed: {e}", flush=True)
    sys.exit(1)

# Quick AutoGluon test
print("\n🚀 Testing AutoGluon with sample data...", flush=True)
try:
    # Create sample data
    sample_df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })

    # Split
    train = sample_df[:80]
    test = sample_df[80:]

    # Train minimal model
    predictor = TabularPredictor(
        label='target',
        verbosity=0
    ).fit(
        train_data=train,
        hyperparameters={'GBM': {'num_boost_round': 2}},
        time_limit=10
    )

    # Predict
    predictions = predictor.predict(test)
    print(f"✅ AutoGluon test successful! Made {len(predictions)} predictions", flush=True)

except Exception as e:
    print(f"❌ AutoGluon test failed: {e}", flush=True)
    sys.exit(1)

print("\n✅ All tests passed! Environment is ready.", flush=True)
print("You can now run: uv run python run_training.py", flush=True)