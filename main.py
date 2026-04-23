"""
main.py
-------
Project entrypoint for the Flight Price Prediction pipeline.

Runs the full workflow in order:
    1. Data ingestion
    2. Data validation
    3. Data transformation
    4. Model training
"""

import os
import sys

# Add src/ to Python path so local modules can be imported cleanly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_ingestion import load_data
from data_validation import validate_data
from data_transformer import transform_data
from model_trainer import train_models


def main() -> None:
    """Run the full ML pipeline end-to-end."""
    print("=" * 60)
    print("Flight Price Prediction Pipeline Started")
    print("=" * 60)

    # Step 1: Load raw data
    df = load_data()

    # Step 2: Validate raw data
    validate_data(df)

    # Step 3: Transform and save processed artifacts
    transform_data()

    # Step 4: Train and evaluate models
    train_models()

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
