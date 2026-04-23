"""
model_trainer.py
----------------
Trains two regression models on the preprocessed flight price data:

    1. Linear Regression  — baseline (sklearn)
    2. ANN                — main model (Keras/TensorFlow)

Both models are evaluated on the test set using R2, RMSE, and MAE,
computed AFTER inverse-log-transforming predictions back to rupee space.

Pipeline:
    - Load preprocessed train/test splits from data/processed/
    - Train linear regression baseline
    - Train ANN (128 -> 64 -> 32 -> 1, ReLU, Adam, MSE, early stopping)
    - Evaluate both on the test set (inverse-transformed)
    - Save models, training history, and a metrics summary

Can be run standalone OR imported and called from main.py.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
PROCESSED_DIR = "data/processed"
MODELS_DIR = "data/processed"   # keeping all artifacts in one place

# Reproducibility
SEED = 42

# ANN architecture
HIDDEN_UNITS = [128, 64, 32]
HIDDEN_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "linear"

# ANN training config
LEARNING_RATE = 0.001
LOSS = "mse"
BATCH_SIZE = 256
MAX_EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seeds(seed: int = SEED) -> None:
    """Lock random seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_processed_data():
    """Load train/test splits produced by data_transformer.py."""
    print("\n[model_trainer] --- Loading processed data ---")
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).squeeze()
        y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    except FileNotFoundError as e:
        print(f"[model_trainer] ERROR: Missing processed file — {e}")
        print("[model_trainer] Run data_transformer.py first.")
        sys.exit(1)

    print(f"[model_trainer] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[model_trainer] y_train: {y_train.shape} | y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# Baseline: Linear Regression
# ---------------------------------------------------------------------
def train_linear_regression(X_train, y_train):
    """Fit sklearn LinearRegression on log-transformed target."""
    print("\n[model_trainer] --- Training Linear Regression (baseline) ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("[model_trainer] Linear Regression trained.")
    return lr


# ---------------------------------------------------------------------
# ANN: build and train
# ---------------------------------------------------------------------
def build_ann(input_dim: int) -> Sequential:
    """
    Build the ANN:
        Input(29) -> Dense(128, relu) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, linear)
    """
    model = Sequential(name="flight_price_ann")
    model.add(Input(shape=(input_dim,)))

    for units in HIDDEN_UNITS:
        model.add(Dense(units, activation=HIDDEN_ACTIVATION))

    model.add(Dense(1, activation=OUTPUT_ACTIVATION))

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=LOSS,
        metrics=["mae"],
    )
    return model


def train_ann(X_train, y_train):
    """Build and train the ANN with early stopping on validation loss."""
    print("\n[model_trainer] --- Training ANN ---")
    model = build_ann(input_dim=X_train.shape[1])
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    print(f"[model_trainer] ANN training complete after {len(history.history['loss'])} epochs.")
    return model, history


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_model(model, X_test, y_test_log, model_name: str, is_keras: bool = False):
    """
    Evaluate a model on the test set, computing R2 / RMSE / MAE in rupee space.

    Both predictions and ground truth are inverse-transformed with np.expm1
    before computing metrics — so the numbers are interpretable in rupees.
    """
    print(f"\n[model_trainer] --- Evaluating {model_name} ---")

    y_pred_log = model.predict(X_test)
    if is_keras:
        y_pred_log = y_pred_log.flatten()

    # Inverse the log1p transform applied in data_transformer.py
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test_log)

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    print(f"[model_trainer] {model_name} — R2  : {r2:.4f}")
    print(f"[model_trainer] {model_name} — RMSE: ₹{rmse:,.2f}")
    print(f"[model_trainer] {model_name} — MAE : ₹{mae:,.2f}")

    return {"model": model_name, "r2": r2, "rmse": rmse, "mae": mae}


# ---------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------
def save_artifacts(lr_model, ann_model, history, metrics):
    """Save both trained models, ANN training history, and a metrics CSV."""
    print("\n[model_trainer] --- Saving model artifacts ---")

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Linear regression
        joblib.dump(lr_model, os.path.join(MODELS_DIR, "linear_model.pkl"))

        # ANN (native Keras format)
        ann_model.save(os.path.join(MODELS_DIR, "ann_model.keras"))

        # Training history (epoch-level loss/val_loss/etc.) for later plotting
        joblib.dump(history.history, os.path.join(MODELS_DIR, "ann_history.pkl"))

        # Metrics summary as CSV for easy portfolio reference
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(MODELS_DIR, "metrics_summary.csv"), index=False)

        print(f"[model_trainer] Saved: linear_model.pkl, ann_model.keras, "
              f"ann_history.pkl, metrics_summary.csv")
        print(f"[model_trainer] Location: {MODELS_DIR}/")

    except Exception as e:
        print(f"[model_trainer] ERROR: Failed to save artifacts — {e}")
        sys.exit(1)


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
def train_models() -> None:
    """Run the full model training and evaluation pipeline."""
    print("[model_trainer] Starting model training pipeline...")
    set_seeds()

    X_train, X_test, y_train, y_test = load_processed_data()

    # Baseline
    lr_model = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Linear Regression", is_keras=False)

    # ANN
    ann_model, history = train_ann(X_train, y_train)
    ann_metrics = evaluate_model(ann_model, X_test, y_test, "ANN", is_keras=True)

    # Side-by-side comparison
    print("\n[model_trainer] --- Final Comparison ---")
    comparison = pd.DataFrame([lr_metrics, ann_metrics])
    print(comparison.to_string(index=False))

    save_artifacts(lr_model, ann_model, history, [lr_metrics, ann_metrics])

    print("\n[model_trainer] Model training pipeline complete.")


if __name__ == "__main__":
    train_models()