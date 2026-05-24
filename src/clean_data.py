"""
src/clean_data.py
-----------------
Cleans raw insurance data and encodes categorical columns for ML.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data, engineer_base_features

RAW_PATH     = "data/insurance_data.txt"
CLEANED_PATH = "data/insurance_data_cleaned.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data(RAW_PATH)
df = engineer_base_features(df)

# ── Basic Cleaning ────────────────────────────────────────────────────────────
df = df.dropna(subset=["TotalPremium", "TotalClaims"])

cap = df["LossRatio"].quantile(0.99)
df["LossRatio"] = df["LossRatio"].clip(upper=cap)

# ── Encode Categorical Columns ────────────────────────────────────────────────
# Drop high-cardinality or ID-like columns not useful for ML
drop_cols = ["PolicyID", "UnderwrittenCoverID"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Drop datetime (already extracted what we need)
df = df.drop(columns=["TransactionMonth"], errors="ignore")

# Label-encode all remaining object columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
print(f"Encoding {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes  
    # -1 = was NaN, 0+ = encoded category

# ── Final Check ───────────────────────────────────────────────────────────────
print(f"Cleaned shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Remaining nulls:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

df.to_csv(CLEANED_PATH, index=False)
print(f"Saved → {CLEANED_PATH}")