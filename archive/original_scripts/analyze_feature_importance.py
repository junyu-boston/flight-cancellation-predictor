#!/usr/bin/env python
"""
Feature importance analysis for the trained model
Implements the pharmaceutical research best practices you mentioned
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import seaborn as sns

print("🔬 Feature Importance Analysis", flush=True)
print("=" * 50, flush=True)

# Load the saved model
model_path = "./AutogluonModels/ag-20260128_113233"
print(f"\n📂 Loading model from: {model_path}", flush=True)

try:
    predictor = TabularPredictor.load(model_path)
    print("✅ Model loaded successfully", flush=True)
except Exception as e:
    print(f"⚠️  Using most recent model directory", flush=True)
    import os
    import glob

    # Find most recent model
    model_dirs = glob.glob("./AutogluonModels/ag-*")
    if model_dirs:
        model_path = max(model_dirs, key=os.path.getctime)
        predictor = TabularPredictor.load(model_path)
        print(f"✅ Loaded model from: {model_path}", flush=True)
    else:
        print("❌ No models found. Please run training first.", flush=True)
        exit(1)

# Load test data for feature importance calculation
print("\n📊 Loading test data...", flush=True)

# Recreate test data (same as training)
df_A = pd.read_parquet("T_ONTIME_REPORTING11_A.parquet")
df_B = pd.read_parquet("T_ONTIME_REPORTING11_B.parquet")
df = pd.concat([df_A, df_B], ignore_index=True)
arp = pd.read_excel("ARP-NPIAS-2025-2029-AppendixA.xlsx")
df = pd.merge(df, arp, how='left', left_on='ORIGIN', right_on='LocID')
df.columns = df.columns.str.lower()

# Basic feature engineering
df['cancelled'] = df['cancelled'].astype('category')
df['scheduled_hour_of_departure'] = df['crs_dep_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)
df['scheduled_hour_of_arrival'] = df['crs_arr_time'].astype(str).str.zfill(4).str.slice(0, 2).astype(int)
df['ind_is_summer'] = np.where((df['month'] >= 6) & (df['month'] <= 8), 1, 0)
df['ind_early_departure'] = np.where(df['dep_delay'] < 0, 1, 0)

# Select same features
features = [
    'year', 'month', 'day_of_month', 'day_of_week',
    'scheduled_hour_of_departure', 'scheduled_hour_of_arrival',
    'origin', 'dest', 'op_unique_carrier',
    'dep_delay', 'dep_delay_new', 'dep_del15',
    'distance', 'crs_elapsed_time',
    'cancelled',
    'ind_is_summer', 'ind_early_departure'
]
available_features = [f for f in features if f in df.columns]
df = df[available_features]

# Sample for analysis
test_sample = df.sample(n=1000, random_state=42)

print(f"✅ Loaded {len(test_sample)} test samples", flush=True)

# Calculate feature importance
print("\n🔬 Calculating feature importance using permutation method...", flush=True)
print("  (This shuffles each feature and measures performance drop)", flush=True)

importance = predictor.feature_importance(
    test_sample,
    num_shuffle_sets=5,  # More shuffles = more stable estimates
    subsample_size=None,  # Use all data
    silent=False
)

print("\n📊 Feature Importance Results:", flush=True)
print("=" * 50, flush=True)

# Display importance scores
print("\nTop Features (Most Important):", flush=True)
print("-" * 40, flush=True)

# Sort by importance
importance_sorted = importance.sort_values('importance', ascending=False)

# Categorize features
high_impact = importance_sorted[importance_sorted['importance'] > 0.01]
medium_impact = importance_sorted[(importance_sorted['importance'] > 0.001) &
                                  (importance_sorted['importance'] <= 0.01)]
low_impact = importance_sorted[(importance_sorted['importance'] > 0) &
                              (importance_sorted['importance'] <= 0.001)]
negative_impact = importance_sorted[importance_sorted['importance'] <= 0]

print("\n🔴 HIGH IMPACT FEATURES (>0.01):", flush=True)
if len(high_impact) > 0:
    for idx, row in high_impact.iterrows():
        print(f"  • {idx:30s}: {row['importance']:.4f} (±{row['stddev']:.4f})", flush=True)
else:
    print("  None", flush=True)

print("\n🟡 MEDIUM IMPACT FEATURES (0.001-0.01):", flush=True)
if len(medium_impact) > 0:
    for idx, row in medium_impact.iterrows():
        print(f"  • {idx:30s}: {row['importance']:.4f} (±{row['stddev']:.4f})", flush=True)
else:
    print("  None", flush=True)

print("\n🟢 LOW IMPACT FEATURES (0-0.001):", flush=True)
if len(low_impact) > 0:
    for idx, row in low_impact.iterrows():
        print(f"  • {idx:30s}: {row['importance']:.4f} (±{row['stddev']:.4f})", flush=True)
else:
    print("  None", flush=True)

print("\n⚪ NEGATIVE/NOISE FEATURES (≤0):", flush=True)
if len(negative_impact) > 0:
    for idx, row in negative_impact.iterrows():
        print(f"  • {idx:30s}: {row['importance']:.4f} (±{row['stddev']:.4f})", flush=True)
    print("\n  💡 Recommendation: Remove these features to improve model", flush=True)
else:
    print("  None - All features contribute positively!", flush=True)

# Pharmaceutical research insights
print("\n💊 Pharmaceutical Research Analogy:", flush=True)
print("=" * 50, flush=True)

print("\nIf this were a drug response prediction model:", flush=True)
print("-" * 40, flush=True)

# Map to pharma context
pharma_mapping = {
    'dep_del15': 'Previous Treatment Response',
    'dep_delay': 'Biomarker Level',
    'origin': 'Genetic Variant',
    'dest': 'Target Organ',
    'op_unique_carrier': 'Drug Type',
    'scheduled_hour_of_departure': 'Dosing Time',
    'distance': 'Drug Concentration',
    'day_of_week': 'Treatment Day'
}

print("\nFeature Importance Translation:", flush=True)
for feature in importance_sorted.head(5).index:
    pharma_equiv = pharma_mapping.get(feature, feature)
    imp_value = importance_sorted.loc[feature, 'importance']

    if imp_value > 0.01:
        impact = "CRITICAL"
    elif imp_value > 0.001:
        impact = "Moderate"
    else:
        impact = "Minor"

    print(f"  • {feature:20s} → {pharma_equiv:25s} [{impact}]", flush=True)

# Model refinement recommendations
print("\n🔧 Model Refinement Recommendations:", flush=True)
print("=" * 50, flush=True)

if len(negative_impact) > 0:
    print("\n1. Remove noise features:", flush=True)
    for feature in negative_impact.index[:3]:
        print(f"   - {feature}", flush=True)

print("\n2. Focus on high-impact features:", flush=True)
for feature in high_impact.index[:3]:
    print(f"   + {feature} (importance: {high_impact.loc[feature, 'importance']:.4f})", flush=True)

print("\n3. Consider feature engineering for:", flush=True)
for feature in medium_impact.index[:3]:
    print(f"   ? {feature} (could be enhanced)", flush=True)

# Save importance results
output_file = "feature_importance_analysis.csv"
importance_sorted.to_csv(output_file)
print(f"\n💾 Results saved to: {output_file}", flush=True)

# Visualization
print("\n📊 Creating visualization...", flush=True)

plt.figure(figsize=(10, 6))
top_features = importance_sorted.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features.index)
plt.xlabel('Feature Importance Score')
plt.title('Top 10 Most Important Features\n(Permutation-based importance)')
plt.tight_layout()

plot_file = "feature_importance_plot.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"📈 Plot saved to: {plot_file}", flush=True)

print("\n✅ Feature importance analysis complete!", flush=True)
print("\n🎯 Key Takeaway:", flush=True)
print("   Just like identifying which patient characteristics predict", flush=True)
print("   drug response, we've identified which flight attributes", flush=True)
print("   predict cancellations. Focus on the high-impact features", flush=True)
print("   and remove the noise for better model performance.", flush=True)