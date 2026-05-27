"""
src/modeling.py
===============
Model training, evaluation, and SHAP interpretability for Task 4.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import root_mean_squared_error, r2_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.impute          import SimpleImputer

import xgboost as xgb
import shap


# ─── Feature preparation ──────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    target: str = "TotalClaims",
    claims_only: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    df = df.copy()

    # Strip comma-formatted numbers e.g. "44,069,150.0000" -> 44069150.0
    for col in df.select_dtypes(include="object").columns:
        stripped = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        if converted.dtype != object:
            df[col] = converted

    if claims_only:
        claim_col = _find_col(df, "TotalClaims")
        df = df[pd.to_numeric(df[claim_col], errors="coerce") > 0].copy()

    if "RegistrationYear" in df.columns:
        reg_year = pd.to_numeric(df["RegistrationYear"], errors="coerce")
        df["VehicleAge"] = 2015 - reg_year

    for col, new_col in [("AlarmImmobiliser", "HasAlarm"),
                         ("TrackingDevice",   "HasTracking")]:
        if col in df.columns:
            df[new_col] = df[col].astype(str).str.lower().str.contains("yes").astype(int)

    drop_cols = [
        "LossRatio", "Margin", "_Margin",
        "UnderwrittenCoverID", "PolicyID",
        "TransactionMonth", "VehicleIntroDate",
        "AlarmImmobiliser", "TrackingDevice",
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    target_col = _find_col(df, target)
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.drop(columns=[target_col])

    le = LabelEncoder()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = le.fit_transform(X[col].astype(str).fillna("missing"))

    # Reset index so numpy array from imputer aligns correctly
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Ensure all numeric columns are truly numeric
    for col in X.columns:
       X[col] = pd.to_numeric(X[col], errors="coerce")

    num_cols = X.select_dtypes(include=[np.number]).columns

    imp = SimpleImputer(strategy="median")

# Convert imputed array back to DataFrame
    # Select numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Remove columns that are entirely NaN
    valid_num_cols = [
        col for col in num_cols
        if not X[col].isna().all()
    ]

    imp = SimpleImputer(strategy="median")

# Impute only valid numeric columns
    X[valid_num_cols] = pd.DataFrame(
       imp.fit_transform(X[valid_num_cols]),
       columns=valid_num_cols,
       index=X.index
    )

    # Remove rows where target is NaN
    mask = y.notna()
    X, y = X[mask], y[mask]

# Final NaN cleanup
    X = X.replace([np.inf, -np.inf], np.nan)

# Fill remaining NaNs
    X = X.fillna(0)

# Optional: ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return train_test_split(X,y,test_size=test_size,random_state=random_state)


# ─── Model training ───────────────────────────────────────────────────────────

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=15, random_state=42):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42):
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name=""):
    y_pred = model.predict(X_test)
    return {
        "model":  model_name or type(model).__name__,
        "rmse":   round(root_mean_squared_error(y_test, y_pred), 2),
        "r2":     round(r2_score(y_test, y_pred), 4),
        "n_test": len(y_test),
    }


def compare_models(results):
    df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    df.index += 1
    return df


# ─── SHAP interpretability ────────────────────────────────────────────────────

def plot_shap_summary(model, X_test, model_name="Best Model", max_display=15):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False, plot_type="dot")
    plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold", pad=12)
    plt.tight_layout()
    return plt.gcf()


def interpret_shap(model, X_test, top_n=10):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    importance  = pd.DataFrame({
        "feature":       X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    importance.index += 1
    return importance


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, name: str) -> str:
    for col in df.columns:
        if col.strip().lower() == name.lower():
            return col
    raise KeyError(f"Column '{name}' not found. Available: {df.columns.tolist()}")