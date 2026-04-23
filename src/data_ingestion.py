"""
data_ingestion.py
-----------------
Loads the raw Flight Price Prediction dataset from data/raw/clean_dataset.csv
and returns it as a pandas DataFrame.

This module is the first step of the ML pipeline. It does ONE thing:
load the raw data. No cleaning, no validation, no transformation.

Can be run standalone OR imported and called from main.py.
"""

import pandas as pd
import sys


def load_data(file_path: str = "data/raw/clean_dataset.csv") -> pd.DataFrame:
    """
    Load raw dataset from the given CSV file path.

    Parameters
    ----------
    file_path : str
        Path to the raw CSV file. Defaults to 'data/raw/clean_dataset.csv'.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[data_ingestion] Data loaded successfully from: {file_path}")
        print(f"[data_ingestion] Shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"[data_ingestion] ERROR: File not found at '{file_path}'")
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print(f"[data_ingestion] ERROR: File at '{file_path}' is empty")
        sys.exit(1)

    except Exception as e:
        print(f"[data_ingestion] ERROR: Unexpected error while loading data — {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Allows running this script standalone: python src/data_ingestion.py
    df = load_data()
    print(df.head())
