"""
data_loader.py
==============
Handles all data ingestion, type enforcement, and derived metric computation
for the ACIS insurance dataset.

Responsibilities
----------------
- Load the raw pipe-delimited CSV from disk.
- Cast columns to their correct dtypes (numeric, categorical, datetime).
- Compute the two core business metrics: Loss Ratio and Margin.
- Provide a lightweight data-quality summary (missing values, duplicates).

Usage
-----
    from src.data_loader import load_data, get_quality_report

    df = load_data("data/insurance_data.csv")
    report = get_quality_report(df)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Column type mappings
# ---------------------------------------------------------------------------

# Columns that should be read as plain Python floats / ints.
NUMERIC_COLS = [
    "TotalPremium",
    "TotalClaims",
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "CustomValueEstimate",
    "CapitalOutstanding",
    "Kilowatts",
    "Cubiccapacity",
    "Cylinders",
    "NumberOfDoors",
    "NumberOfVehiclesInFleet",
    "RegistrationYear",
]

# Columns that represent discrete groups and should be stored as pandas
# Categoricals to reduce memory footprint.
CATEGORICAL_COLS = [
    "IsVATRegistered",
    "Citizenship",
    "LegalType",
    "Title",
    "Language",
    "Bank",
    "AccountType",
    "MaritalStatus",
    "Gender",
    "Country",
    "Province",
    "PostalCode",
    "MainCrestaZone",
    "SubCrestaZone",
    "ItemType",
    "VehicleType",
    "Make",
    "Model",
    "Bodytype",
    "AlarmImmobiliser",
    "TrackingDevice",
    "NewVehicle",
    "WrittenOff",
    "Rebuilt",
    "Converted",
    "CrossBorder",
    "CoverCategory",
    "CoverType",
    "CoverGroup",
    "Section",
    "Product",
    "StatutoryClass",
    "StatutoryRiskType",
    "TermFrequency",
    "ExcessSelected",
]

# Columns that encode calendar months and should be parsed as datetimes.
DATE_COLS = ["TransactionMonth", "VehicleIntroDate"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(filepath: str | Path, sep: str = "|") -> pd.DataFrame:
    """
    Load the raw ACIS insurance CSV into a DataFrame with correct types.

    The function reads the file, coerces numeric columns with
    ``pd.to_numeric`` (non-parseable values become NaN), casts categorical
    columns to the ``category`` dtype, and parses date columns with
    ``pd.to_datetime``.  Two derived business metrics are appended:
    ``LossRatio`` and ``Margin``.

    Parameters
    ----------
    filepath : str or Path
        Path to the pipe-delimited CSV file.
    sep : str, optional
        Column delimiter (default ``"|"``).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for EDA.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist on disk.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # Read all columns as strings first to avoid mixed-type inference issues.
    df = pd.read_csv(filepath, sep=sep, low_memory=False, dtype=str)

    # --- Numeric coercion ---
    # Replace any non-numeric strings (e.g. empty strings) with NaN.
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Categorical coercion ---
    # Using the category dtype saves memory and speeds up groupby operations.
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Datetime parsing ---
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Derived metrics ---
    df = _add_derived_metrics(df)

    return df


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``LossRatio`` and ``Margin`` columns to the DataFrame.

    - **LossRatio** = TotalClaims / TotalPremium
      Policies with zero premium are set to NaN to avoid division by zero.
    - **Margin** = TotalPremium − TotalClaims
      The per-policy profit contribution before expenses.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that must contain ``TotalPremium`` and ``TotalClaims``.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with two new columns appended in place.
    """
    if "TotalPremium" in df.columns and "TotalClaims" in df.columns:
        # Replace 0-premium rows with NaN to prevent inf values in LossRatio.
        premium_safe = df["TotalPremium"].replace(0, np.nan)
        df["LossRatio"] = df["TotalClaims"] / premium_safe
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]
    return df


def get_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data-quality summary for all columns.

    The report shows, for each column:
    - The inferred dtype.
    - The count of missing values.
    - The percentage of missing values (relative to total rows).
    - The number of unique values.

    Parameters
    ----------
    df : pd.DataFrame
        Any DataFrame, typically the output of ``load_data``.

    Returns
    -------
    pd.DataFrame
        A summary DataFrame indexed by column name, sorted by
        ``missing_pct`` descending so the most problematic columns
        appear first.
    """
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique_values": df.nunique(),
    })
    return report.sort_values("missing_pct", ascending=False)
