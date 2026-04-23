"""
data_validation.py
------------------
Validates the raw Flight Price Prediction dataset against expected schema
and data quality rules.

This module is READ-ONLY. It checks and reports — it does NOT modify,
clean, or save the data. Any cleaning (duplicate removal, column drops,
etc.) happens downstream in data_transformer.py.

Can be run standalone OR imported and called from main.py.
"""

import sys
import pandas as pd
from data_ingestion import load_data


# ---------------------------------------------------------------------
# Expected schema — locked in based on EDA findings
# ---------------------------------------------------------------------
EXPECTED_COLUMNS = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
    "price",
]

# Columns we expect to drop later but tolerate at ingestion time
TOLERATED_EXTRA_COLUMNS = ["Unnamed: 0", "flight"]


# ---------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------
def check_schema(df: pd.DataFrame) -> None:
    """
    Verify the dataframe has the expected 10 columns.
    Tolerates 'Unnamed: 0' and 'flight' since they are dropped later.
    Exits if any required column is missing.
    """
    print("\n[data_validation] --- Schema Check ---")

    actual_columns = set(df.columns)
    expected_set = set(EXPECTED_COLUMNS)

    missing = expected_set - actual_columns
    extra = actual_columns - expected_set - set(TOLERATED_EXTRA_COLUMNS)

    if missing:
        print(f"[data_validation] ERROR: Missing required columns → {missing}")
        sys.exit(1)

    if extra:
        print(f"[data_validation] WARNING: Unexpected extra columns → {extra}")

    print(f"[data_validation] Schema OK — all {len(EXPECTED_COLUMNS)} required columns present.")


def check_missing_values(df: pd.DataFrame) -> None:
    """
    Report missing values per column.
    - If ANY column has >50% missing → exit (data unusable)
    - If any column has <50% missing → warn and continue
    - If no missing values → pass silently
    """
    print("\n[data_validation] --- Missing Values Check ---")

    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    cols_with_missing = missing_counts[missing_counts > 0]

    if cols_with_missing.empty:
        print("[data_validation] No missing values found.")
        return

    print("[data_validation] Missing values detected:")
    for col in cols_with_missing.index:
        print(f"    {col}: {cols_with_missing[col]} ({missing_percent[col]:.2f}%)")

    critical_cols = missing_percent[missing_percent > 50]
    if not critical_cols.empty:
        print(f"[data_validation] ERROR: Columns exceeding 50% missing → {list(critical_cols.index)}")
        sys.exit(1)

    print("[data_validation] WARNING: Missing values present but under 50% threshold — continuing.")


def check_duplicates(df: pd.DataFrame) -> None:
    """
    Report number and percentage of duplicate rows.
    Does NOT drop them — dropping happens in data_transformer.py.
    """
    print("\n[data_validation] --- Duplicates Check ---")

    dup_count = df.duplicated().sum()
    dup_percent = (dup_count / len(df)) * 100

    if dup_count == 0:
        print("[data_validation] No duplicate rows found.")
    else:
        print(f"[data_validation] Found {dup_count} duplicate rows ({dup_percent:.2f}%) — will be handled in transformer stage.")


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> None:
    """
    Run all validation checks in sequence.
    Read-only — does not modify or return the dataframe.
    """
    print("[data_validation] Starting validation...")
    print(f"[data_validation] Dataframe shape: {df.shape}")

    check_schema(df)
    check_missing_values(df)
    check_duplicates(df)

    print("\n[data_validation] Validation complete — no modifications made.")


if __name__ == "__main__":
    # Allows running standalone: python src/data_validation.py
    df = load_data()
    validate_data(df)