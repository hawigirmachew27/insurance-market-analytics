"""
src/data_loader.py
----------------
Reusable data loader helper functions for the Insurance data
"""
from __future__ import annotations



import pandas as pd
import numpy as np


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw insurance data.

    Parameters
    ----------
    filepath : str
        Path to insurance_data.txt (pipe-delimited)
    """
    df = pd.read_csv(
        filepath,
        sep='|',
        engine='python',
        on_bad_lines='warn',   # skip/warn on malformed rows instead of crashing
        quoting=3,             # QUOTE_NONE — don't try to interpret quote chars
        encoding='utf-8',
        encoding_errors='replace',
    )
    
    return df

# ─── Quality Checks ───────────────────────────────────────────────────────────

def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising missing values and dtypes."""
    report = pd.DataFrame({
        'dtype':         df.dtypes,
        'missing_count': df.isnull().sum(),       # was 'missing'
        'missing_pct':   (df.isnull().mean() * 100).round(2),  # was 'missing%'
        'unique':        df.nunique(),
    })
    return report
# in data_loader.py

def engineer_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Parse TransactionMonth to datetime
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")

    premium = pd.to_numeric(df["TotalPremium"], errors="coerce")
    claims  = pd.to_numeric(df["TotalClaims"],  errors="coerce")

    df["LossRatio"] = claims / premium.replace(0, np.nan)
    df["Margin"]    = premium - claims
    return df