"""
data_transformer.py
-------------------
Transforms the raw Flight Price Prediction dataset into model-ready form:
cleans, splits, encodes, scales, and saves all artifacts.

Output locations:
    - data/processed/ -> X_train.csv, X_test.csv, y_train.csv, y_test.csv
    - models/         -> encoders, scalers (ohe.pkl, ordinal_encoder_*.pkl, scaler.pkl)

Pipeline order (strict, to prevent data leakage):
    1. Load raw data
    2. Clean (drop junk columns, drop duplicates)
    3. Separate X and y
    4. Train-test split (80/20, random_state=42)
    5. Log-transform y (AFTER split)
    6. Encode categoricals (fit on train only)
    7. Scale numericals (fit on train only)
    8. Save CSVs to data/processed/ and pickles to models/

Can be run standalone OR imported and called from main.py.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from data_ingestion import load_data


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DATA_OUTPUT_DIR = "data/processed"  # CSVs go here
MODELS_OUTPUT_DIR = "models"        # Pickles go here

TARGET_COLUMN = "price"
COLUMNS_TO_DROP = ["Unnamed: 0", "flight"]

# Encoding groups
ONEHOT_COLUMNS = [
    "airline",
    "source_city",
    "departure_time",
    "arrival_time",
    "destination_city",
]
ORDINAL_STOPS_ORDER = [["zero", "one", "two_or_more"]]
ORDINAL_CLASS_ORDER = [["Economy", "Business"]]

# Scaling group
NUMERICAL_COLUMNS = ["duration", "days_left"]

# Split config
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop non-predictive columns ('Unnamed: 0', 'flight') if present
    and remove duplicate rows.
    """
    print("\n[data_transformer] --- Cleaning ---")
    initial_shape = df.shape

    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        print(f"[data_transformer] Dropped columns: {cols_present}")

    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before_dedup - len(df)
    print(f"[data_transformer] Dropped {dropped} duplicate rows")
    print(f"[data_transformer] Shape: {initial_shape} -> {df.shape}")

    return df


def split_features_target(df: pd.DataFrame):
    """Separate features (X) from target (y)."""
    print("\n[data_transformer] --- Splitting features and target ---")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    print(f"[data_transformer] X shape: {X.shape} | y shape: {y.shape}")
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    """80/20 train-test split with fixed random_state."""
    print("\n[data_transformer] --- Train-test split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[data_transformer] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[data_transformer] y_train: {y_train.shape} | y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def log_transform_target(y_train: pd.Series, y_test: pd.Series):
    """
    Apply np.log1p to the target — AFTER split, to prevent leakage.
    log1p is used instead of log to safely handle any zero values.
    """
    print("\n[data_transformer] --- Log-transforming target ---")
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    print(f"[data_transformer] y_train range: [{y_train_log.min():.4f}, {y_train_log.max():.4f}]")
    print(f"[data_transformer] y_test  range: [{y_test_log.min():.4f}, {y_test_log.max():.4f}]")
    return y_train_log, y_test_log


def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Encode all categorical features.

    - 'class'  -> OrdinalEncoder with order ['Economy', 'Business']
    - 'stops'  -> OrdinalEncoder with order ['zero', 'one', 'two_or_more']
    - others   -> OneHotEncoder with drop='first'

    Every encoder is fit on X_train only, then used to transform both.
    """
    print("\n[data_transformer] --- Encoding categoricals ---")

    # Work on copies so the originals are not mutated
    X_train = X_train.copy()
    X_test = X_test.copy()

    # 1. Ordinal encode 'class' (Economy=0, Business=1)
    ord_enc_class = OrdinalEncoder(categories=ORDINAL_CLASS_ORDER)
    X_train["class"] = ord_enc_class.fit_transform(X_train[["class"]])
    X_test["class"] = ord_enc_class.transform(X_test[["class"]])
    print("[data_transformer] Encoded 'class' (Economy=0, Business=1)")

    # 2. Ordinal encode 'stops' (zero=0, one=1, two_or_more=2)
    ord_enc_stops = OrdinalEncoder(categories=ORDINAL_STOPS_ORDER)
    X_train["stops"] = ord_enc_stops.fit_transform(X_train[["stops"]])
    X_test["stops"] = ord_enc_stops.transform(X_test[["stops"]])
    print("[data_transformer] Encoded 'stops' (zero=0, one=1, two_or_more=2)")

    # 3. One-hot encode the remaining nominal columns
    ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_train[ONEHOT_COLUMNS])

    ohe_train_arr = ohe.transform(X_train[ONEHOT_COLUMNS])
    ohe_test_arr = ohe.transform(X_test[ONEHOT_COLUMNS])

    ohe_feature_names = ohe.get_feature_names_out(ONEHOT_COLUMNS)
    ohe_train_df = pd.DataFrame(ohe_train_arr, columns=ohe_feature_names, index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test_arr, columns=ohe_feature_names, index=X_test.index)

    # Drop original nominal columns and concat the encoded ones
    X_train = pd.concat([X_train.drop(columns=ONEHOT_COLUMNS), ohe_train_df], axis=1)
    X_test = pd.concat([X_test.drop(columns=ONEHOT_COLUMNS), ohe_test_df], axis=1)
    print(f"[data_transformer] One-hot encoded {len(ONEHOT_COLUMNS)} columns -> {len(ohe_feature_names)} new columns")

    print(f"[data_transformer] X_train shape after encoding: {X_train.shape}")
    print(f"[data_transformer] X_test  shape after encoding: {X_test.shape}")

    return X_train, X_test, ord_enc_class, ord_enc_stops, ohe


def scale_numerical(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standard-scale 'duration' and 'days_left'.
    Scaler is fit on X_train only.
    """
    print("\n[data_transformer] --- Scaling numerical features ---")

    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    print(f"[data_transformer] Scaled columns: {NUMERICAL_COLUMNS}")
    return X_train, X_test, scaler


def save_artifacts(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ord_enc_class: OrdinalEncoder,
    ord_enc_stops: OrdinalEncoder,
    ohe: OneHotEncoder,
    scaler: StandardScaler,
) -> None:
    """
    Save CSVs to data/processed/ and pickles to models/.
    """
    print("\n[data_transformer] --- Saving artifacts ---")

    try:
        # Create both directories
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

        # Save CSVs to data/processed/
        X_train.to_csv(os.path.join(DATA_OUTPUT_DIR, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_OUTPUT_DIR, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(DATA_OUTPUT_DIR, "y_train.csv"), index=False, header=True)
        y_test.to_csv(os.path.join(DATA_OUTPUT_DIR, "y_test.csv"), index=False, header=True)
        print(f"[data_transformer] CSVs saved to: {DATA_OUTPUT_DIR}/")

        # Save pickles to models/
        joblib.dump(ord_enc_class, os.path.join(MODELS_OUTPUT_DIR, "ordinal_encoder_class.pkl"))
        joblib.dump(ord_enc_stops, os.path.join(MODELS_OUTPUT_DIR, "ordinal_encoder_stops.pkl"))
        joblib.dump(ohe, os.path.join(MODELS_OUTPUT_DIR, "ohe.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_OUTPUT_DIR, "scaler.pkl"))
        print(f"[data_transformer] Encoders/scalers saved to: {MODELS_OUTPUT_DIR}/")

    except Exception as e:
        print(f"[data_transformer] ERROR: Failed to save artifacts — {e}")
        sys.exit(1)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def transform_data() -> None:
    """Run the full transformation pipeline end-to-end."""
    print("[data_transformer] Starting transformation pipeline...")

    df = load_data()
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    y_train, y_test = log_transform_target(y_train, y_test)

    X_train, X_test, ord_enc_class, ord_enc_stops, ohe = encode_features(X_train, X_test)
    X_train, X_test, scaler = scale_numerical(X_train, X_test)

    save_artifacts(
        X_train, X_test, y_train, y_test,
        ord_enc_class, ord_enc_stops, ohe, scaler,
    )

    print("\n[data_transformer] Transformation complete.")


if __name__ == "__main__":
    transform_data()