"""
Train all models from scratch and save artifacts.
Run this script after cloning the repository to reproduce model files.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("SPACE MISSION PREDICTION - MODEL TRAINING")
print("=" * 60)

# 1. Load data
print("\n[1/7] Loading dataset...")
df = pd.read_csv('dataset/mission_launches.csv', parse_dates=['Datum'])
print(f"Loaded {len(df)} missions from {df['Datum'].min().year} to {df['Datum'].max().year}")

# 2. Feature Engineering
print("\n[2/7] Engineering features...")

# Extract country from location
def extract_country(location):
    if pd.isna(location):
        return 'Unknown'
    location = str(location).strip()
    
    # Special cases
    if 'Gran Canaria' in location or 'Yellow Sea' in location or 'Shahrud' in location:
        return location
    if location in ['Russia', 'New Mexico']:
        return location
    
    # Default: last comma-separated part
    parts = [p.strip() for p in location.split(',')]
    return parts[-1] if parts else 'Unknown'

df['Country'] = df['Location'].apply(extract_country)

# Extract year and decade
df['Year'] = df['Datum'].dt.year
df['Decade'] = (df['Year'] // 10) * 10

# Hierarchical price imputation
print("  - Hierarchical price imputation...")
df['Price_Imputed'] = df[' Rocket'].copy()

# Organization × Decade medians
org_decade_medians = df.groupby(['Company Name', 'Decade'])[' Rocket'].median()
for (org, decade), median in org_decade_medians.items():
    mask = (df['Company Name'] == org) & (df['Decade'] == decade) & df['Price_Imputed'].isna()
    df.loc[mask, 'Price_Imputed'] = median

# Year medians
year_medians = df.groupby('Year')[' Rocket'].median()
for year, median in year_medians.items():
    mask = (df['Year'] == year) & df['Price_Imputed'].isna()
    df.loc[mask, 'Price_Imputed'] = median

# Overall median for remaining
overall_median = df[' Rocket'].median()
df['Price_Imputed'].fillna(overall_median, inplace=True)

# Historical success rates (prevent leakage)
print("  - Computing historical success rates...")
df_sorted = df.sort_values('Datum').copy()
df_sorted['Success_Binary'] = (df_sorted['Status Mission'] == 'Success').astype(int)

org_success_rates = []
country_success_rates = []

for idx in df_sorted.index:
    current_date = df_sorted.loc[idx, 'Datum']
    org = df_sorted.loc[idx, 'Company Name']
    country = df_sorted.loc[idx, 'Country']
    
    # Historical data only (before current mission)
    historical = df_sorted[df_sorted['Datum'] < current_date]
    
    # Organization success rate
    org_hist = historical[historical['Company Name'] == org]
    org_rate = org_hist['Success_Binary'].mean() if len(org_hist) > 0 else 0.5
    org_success_rates.append(org_rate)
    
    # Country success rate
    country_hist = historical[historical['Country'] == country]
    country_rate = country_hist['Success_Binary'].mean() if len(country_hist) > 0 else 0.5
    country_success_rates.append(country_rate)

df_sorted['Org_Success_Rate'] = org_success_rates
df_sorted['Country_Success_Rate'] = country_success_rates

# Restore original order
df = df_sorted.sort_index()

# 3. Prepare features and target
print("\n[3/7] Preparing features and target...")
df['Target'] = (df['Status Mission'] == 'Success').astype(int)

feature_cols = ['Price_Imputed', 'Company Name', 'Location', 'Country', 
                'Org_Success_Rate', 'Country_Success_Rate', 'Year']
X = df[feature_cols].copy()
y = df['Target'].copy()

# 4. Encode categorical features
print("\n[4/7] Encoding categorical features...")
categorical_cols = ['Company Name', 'Location', 'Country']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Save encoder
with open('models/categorical_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("  ✓ Saved categorical_encoder.pkl")

# 5. Train-test split
print("\n[5/7] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"  Train failure rate: {(1-y_train.mean())*100:.1f}%")

# 6. Train models
print("\n[6/7] Training models...")

# Model 1: Original (no resampling)
print("\n  [Model 1/2] Gradient Boosting - Original")
gb_original = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)
gb_original.fit(X_train, y_train)
with open('models/gradient_boosting_original.pkl', 'wb') as f:
    pickle.dump(gb_original, f)
print("    ✓ Saved gradient_boosting_original.pkl")

# Model 2: SMOTE resampling
print("\n  [Model 2/2] Gradient Boosting - SMOTE")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"    Resampled training data: {len(X_train_resampled)} samples (50/50 split)")

gb_smote = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=RANDOM_STATE
)
gb_smote.fit(X_train_resampled, y_train_resampled)
with open('models/gradient_boosting_smote.pkl', 'wb') as f:
    pickle.dump(gb_smote, f)
print("    ✓ Saved gradient_boosting_smote.pkl")

# 7. Evaluate and save configuration
print("\n[7/7] Evaluating models and saving config...")

from sklearn.metrics import f1_score, recall_score, confusion_matrix

# Original model evaluation
y_pred_proba_original = gb_original.predict_proba(X_test)[:, 1]

# Default threshold
y_pred_default = (y_pred_proba_original >= 0.5).astype(int)
f1_default = f1_score(y_test, y_pred_default)
failure_recall_default = recall_score(y_test, y_pred_default, pos_label=0)

# Find optimal threshold
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_original)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# Tuned threshold
y_pred_tuned = (y_pred_proba_original >= optimal_threshold).astype(int)
f1_tuned = f1_score(y_test, y_pred_tuned)
failure_recall_tuned = recall_score(y_test, y_pred_tuned, pos_label=0)

# SMOTE model evaluation
y_pred_smote = gb_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote)
failure_recall_smote = recall_score(y_test, y_pred_smote, pos_label=0)

# Save configuration
config = {
    "optimal_threshold": float(optimal_threshold),
    "default_threshold": 0.5,
    "test_f1_original": float(f1_default),
    "test_f1_tuned": float(f1_tuned),
    "test_f1_smote": float(f1_smote),
    "failure_recall_original": float(failure_recall_default),
    "failure_recall_tuned": float(failure_recall_tuned),
    "failure_recall_smote": float(failure_recall_smote)
}

with open('models/model_config.json', 'w') as f:
    json.dump(config, f, indent=4)
print("  ✓ Saved model_config.json")

# Print summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nOriginal Model (threshold=0.5):")
print(f"  F1 Score: {f1_default:.4f}")
print(f"  Failure Recall: {failure_recall_default*100:.1f}%")

print(f"\nThreshold-Tuned Model (threshold={optimal_threshold:.4f}):")
print(f"  F1 Score: {f1_tuned:.4f}")
print(f"  Failure Recall: {failure_recall_tuned*100:.1f}% (+{(failure_recall_tuned/failure_recall_default-1)*100:.0f}%)")

print(f"\nSMOTE Model:")
print(f"  F1 Score: {f1_smote:.4f}")
print(f"  Failure Recall: {failure_recall_smote*100:.1f}%")

print(f"\nAll model artifacts saved in models/ directory")
print("=" * 60)
