"""
inference.py
------------
Loads the trained model and preprocessors, transforms user input,
and returns predicted flight price in rupees.
"""

import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# ---------------------------------------------------------------------
# Constants (hardcoded from your training pipeline)
# ---------------------------------------------------------------------
NUMERICAL_COLUMNS = ["duration", "days_left"]
ONEHOT_COLUMNS = [
    "airline",
    "source_city",
    "departure_time",
    "arrival_time",
    "destination_city",
]

REQUIRED_FIELDS = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
    "duration",
    "days_left",
]


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


# ---------------------------------------------------------------------
# Global cache (load once, reuse forever)
# ---------------------------------------------------------------------
_ARTIFACTS = None


class InferenceError(Exception):
    """Raised when prediction fails."""
    pass


def load_artifacts():
    """Load all models and preprocessors once and cache them."""
    global _ARTIFACTS
    
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    
    try:
        # Load model from models/
        ann_model = load_model(os.path.join(MODELS_DIR, "ann_model.h5"))
        
        # Load preprocessors from models/
        ohe = joblib.load(os.path.join(MODELS_DIR, "ohe.pkl"))
        ord_enc_class = joblib.load(os.path.join(MODELS_DIR, "ordinal_encoder_class.pkl"))
        ord_enc_stops = joblib.load(os.path.join(MODELS_DIR, "ordinal_encoder_stops.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        
        # Get feature column order from data/processed/X_train.csv
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"), nrows=0)
        feature_columns = X_train.columns.tolist()
        
        _ARTIFACTS = {
            "model": ann_model,
            "ohe": ohe,
            "ord_enc_class": ord_enc_class,
            "ord_enc_stops": ord_enc_stops,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }
        
        print("[inference] Artifacts loaded successfully.")
        return _ARTIFACTS
        
    except FileNotFoundError as e:
        raise InferenceError(f"Model file not found: {e}. Run training pipeline first.")
    except Exception as e:
        raise InferenceError(f"Failed to load artifacts: {e}")


def validate_input(data):
    """Check that all required fields are present and valid."""
    # Check all fields exist
    for field in REQUIRED_FIELDS:
        if field not in data or data[field] in (None, ""):
            raise InferenceError(f"Missing required field: '{field}'")
    
    # Validate numeric fields
    try:
        duration = float(data["duration"])
        if duration <= 0:
            raise InferenceError("Duration must be greater than 0")
    except (TypeError, ValueError):
        raise InferenceError("Duration must be a valid number")
    
    try:
        days_left = int(data["days_left"])
        if days_left < 0:
            raise InferenceError("Days left must be 0 or greater")
    except (TypeError, ValueError):
        raise InferenceError("Days left must be a valid integer")
    
    return True


def preprocess_input(data, artifacts):
    """Transform raw user input into model-ready features."""
    # Create dataframe from input
    input_df = pd.DataFrame([data], columns=REQUIRED_FIELDS)
    
    # Work on a copy
    df = input_df.copy()
    
    # Encode 'class' (Economy=0, Business=1)
    df["class"] = artifacts["ord_enc_class"].transform(df[["class"]])
    
    # Encode 'stops' (zero=0, one=1, two_or_more=2)
    df["stops"] = artifacts["ord_enc_stops"].transform(df[["stops"]])
    
    # One-hot encode the 5 categorical columns
    ohe_array = artifacts["ohe"].transform(df[ONEHOT_COLUMNS])
    ohe_cols = artifacts["ohe"].get_feature_names_out(ONEHOT_COLUMNS)
    ohe_df = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
    
    # Drop original categorical columns and add encoded ones
    df = pd.concat([df.drop(columns=ONEHOT_COLUMNS), ohe_df], axis=1)
    
    # Scale numerical columns
    df[NUMERICAL_COLUMNS] = artifacts["scaler"].transform(df[NUMERICAL_COLUMNS])
    
    # Reorder columns to match training
    df = df.reindex(columns=artifacts["feature_columns"], fill_value=0.0)
    
    return df


def predict_price(data):
    """
    Main prediction function.
    
    Parameters
    ----------
    data : dict
        User input with keys: airline, source_city, departure_time, stops,
        arrival_time, destination_city, class, duration, days_left
    
    Returns
    -------
    dict
        Contains 'predicted_price' (float) and 'model_name' (str)
    """
    # Load artifacts (cached after first call)
    artifacts = load_artifacts()
    
    # Validate input
    validate_input(data)
    
    # Transform input
    X = preprocess_input(data, artifacts)
    
    # Predict in log-space
    prediction_log = artifacts["model"].predict(X, verbose=0).flatten()[0]
    
    # Convert back to rupees
    predicted_price = float(np.expm1(prediction_log))
    
    return {
        "predicted_price": round(predicted_price, 2),
        "model_name": "ANN (R² = 96.7%)",
    }


# ---------------------------------------------------------------------
# Test when run directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_input = {
        "airline": "Vistara",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "stops": "one",
        "arrival_time": "Afternoon",
        "destination_city": "Mumbai",
        "class": "Economy",
        "duration": 2.17,
        "days_left": 15,
    }
    
    try:
        result = predict_price(test_input)
        print("✓ Prediction successful!")
        print(f"  Predicted price: ₹{result['predicted_price']:,.2f}")
        print(f"  Model: {result['model_name']}")
    except InferenceError as e:
        print(f"✗ Prediction failed: {e}")
