"""
inference.py
------------
Inference utilities for Flight Price Prediction.

This module loads the saved preprocessing artifacts and trained models,
transforms raw user inputs into model-ready features, and returns the
predicted ticket price in the original rupee scale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from data_transformer import NUMERICAL_COLUMNS, ONEHOT_COLUMNS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

DEFAULT_MODEL_TYPE = "ann"
VALID_MODEL_TYPES = {"ann", "linear"}

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


class InferenceError(Exception):
    """Raised when prediction-time validation or artifact loading fails."""


@dataclass
class InferenceArtifacts:
    """Container for everything needed during prediction."""

    ordinal_encoder_class: Any
    ordinal_encoder_stops: Any
    one_hot_encoder: Any
    scaler: Any
    feature_columns: list[str]
    ann_model: Any | None = None
    linear_model: Any | None = None


def _artifact_path(filename: str) -> str:
    """Build an absolute path inside data/processed/."""
    return os.path.join(PROCESSED_DIR, filename)


def _ensure_artifact_exists(filename: str) -> str:
    """Return artifact path if present, otherwise raise a helpful error."""
    path = _artifact_path(filename)
    if not os.path.exists(path):
        raise InferenceError(
            f"Required artifact not found: {path}. Run the training pipeline first."
        )
    return path


def load_inference_artifacts(model_type: str = DEFAULT_MODEL_TYPE) -> InferenceArtifacts:
    """
    Load preprocessors, feature schema, and the requested trained model.

    Parameters
    ----------
    model_type : str
        Either 'ann' or 'linear'.
    """
    model_type = model_type.lower()
    if model_type not in VALID_MODEL_TYPES:
        raise InferenceError(
            f"Unsupported model_type='{model_type}'. Expected one of {sorted(VALID_MODEL_TYPES)}."
        )

    ordinal_encoder_class = joblib.load(_ensure_artifact_exists("ordinal_encoder_class.pkl"))
    ordinal_encoder_stops = joblib.load(_ensure_artifact_exists("ordinal_encoder_stops.pkl"))
    one_hot_encoder = joblib.load(_ensure_artifact_exists("ohe.pkl"))
    scaler = joblib.load(_ensure_artifact_exists("scaler.pkl"))

    feature_columns = pd.read_csv(_ensure_artifact_exists("X_train.csv"), nrows=0).columns.tolist()

    artifacts = InferenceArtifacts(
        ordinal_encoder_class=ordinal_encoder_class,
        ordinal_encoder_stops=ordinal_encoder_stops,
        one_hot_encoder=one_hot_encoder,
        scaler=scaler,
        feature_columns=feature_columns,
    )

    if model_type == "ann":
        artifacts.ann_model = load_model(_ensure_artifact_exists("ann_model.keras"))
    else:
        artifacts.linear_model = joblib.load(_ensure_artifact_exists("linear_model.pkl"))

    return artifacts


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a trimmed payload with consistent value types."""
    normalized: dict[str, Any] = {}

    for field in REQUIRED_FIELDS:
        if field not in payload:
            raise InferenceError(f"Missing required field: '{field}'.")

        value = payload[field]
        if isinstance(value, str):
            value = value.strip()

        if value in ("", None):
            raise InferenceError(f"Field '{field}' cannot be empty.")

        normalized[field] = value

    try:
        normalized["duration"] = float(normalized["duration"])
    except (TypeError, ValueError) as exc:
        raise InferenceError("Field 'duration' must be a numeric value.") from exc

    try:
        normalized["days_left"] = float(normalized["days_left"])
    except (TypeError, ValueError) as exc:
        raise InferenceError("Field 'days_left' must be a numeric value.") from exc

    if normalized["duration"] <= 0:
        raise InferenceError("Field 'duration' must be greater than 0.")

    if normalized["days_left"] < 0:
        raise InferenceError("Field 'days_left' must be 0 or greater.")

    return normalized


def _validate_categories(payload: dict[str, Any], artifacts: InferenceArtifacts) -> None:
    """Ensure incoming categories are compatible with fitted encoders."""
    class_values = set(artifacts.ordinal_encoder_class.categories_[0].tolist())
    stops_values = set(artifacts.ordinal_encoder_stops.categories_[0].tolist())

    if payload["class"] not in class_values:
        raise InferenceError(
            f"Invalid value for 'class': {payload['class']}. Expected one of {sorted(class_values)}."
        )

    if payload["stops"] not in stops_values:
        raise InferenceError(
            f"Invalid value for 'stops': {payload['stops']}. Expected one of {sorted(stops_values)}."
        )

    for column, categories in zip(
        ONEHOT_COLUMNS, artifacts.one_hot_encoder.categories_, strict=True
    ):
        allowed_values = set(categories.tolist())
        if payload[column] not in allowed_values:
            raise InferenceError(
                f"Invalid value for '{column}': {payload[column]}. "
                f"Expected one of {sorted(allowed_values)}."
            )


def preprocess_input(
    payload: dict[str, Any],
    artifacts: InferenceArtifacts,
) -> pd.DataFrame:
    """
    Convert raw user input into the exact feature matrix expected by the model.
    """
    payload = _normalize_payload(payload)
    _validate_categories(payload, artifacts)

    input_df = pd.DataFrame([payload], columns=REQUIRED_FIELDS)
    processed_df = input_df.copy()

    processed_df["class"] = artifacts.ordinal_encoder_class.transform(processed_df[["class"]])
    processed_df["stops"] = artifacts.ordinal_encoder_stops.transform(processed_df[["stops"]])

    encoded_array = artifacts.one_hot_encoder.transform(processed_df[ONEHOT_COLUMNS])
    encoded_columns = artifacts.one_hot_encoder.get_feature_names_out(ONEHOT_COLUMNS)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=processed_df.index)

    processed_df = pd.concat(
        [processed_df.drop(columns=ONEHOT_COLUMNS), encoded_df],
        axis=1,
    )

    processed_df[NUMERICAL_COLUMNS] = artifacts.scaler.transform(processed_df[NUMERICAL_COLUMNS])
    processed_df = processed_df.reindex(columns=artifacts.feature_columns, fill_value=0.0)

    return processed_df


def predict_price(
    payload: dict[str, Any],
    model_type: str = DEFAULT_MODEL_TYPE,
) -> dict[str, float | str]:
    """
    Run end-to-end prediction and return the price in original rupee scale.
    """
    artifacts = load_inference_artifacts(model_type=model_type)
    processed_df = preprocess_input(payload, artifacts)

    if model_type == "ann":
        prediction_log = float(artifacts.ann_model.predict(processed_df, verbose=0).flatten()[0])
        model_name = "ANN"
    else:
        prediction_log = float(artifacts.linear_model.predict(processed_df)[0])
        model_name = "Linear Regression"

    predicted_price = float(np.expm1(prediction_log))

    return {
        "predicted_price": round(predicted_price, 2),
        "model_type": model_type,
        "model_name": model_name,
    }


if __name__ == "__main__":
    sample_payload = {
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
        result = predict_price(sample_payload)
        print("Prediction successful.")
        print(result)
    except InferenceError as error:
        print(f"Inference failed: {error}")
