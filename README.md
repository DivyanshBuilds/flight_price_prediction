# Flight Price Prediction

A machine learning project that predicts flight ticket prices using structured flight data.  
The project follows a simple end-to-end ML pipeline: data ingestion, validation, transformation, model training, and evaluation.

## Project Overview

This project uses a flight pricing dataset and builds two regression models:

1. Linear Regression as a baseline model
2. Artificial Neural Network (ANN) as the main model

The preprocessing pipeline handles:
- data loading
- schema validation
- missing value checks
- duplicate checks
- categorical encoding
- numerical feature scaling
- log transformation of the target variable

The trained models are evaluated using:
- R2 Score
- RMSE
- MAE

## Project Structure

```bash
flight_price_prediction/
│
├── data/
│   ├── raw/
│   │   └── clean_dataset.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── scaler.pkl
│       ├── ordinal_encoder_class.pkl
│       ├── ordinal_encoder_stops.pkl
│       ├── ohe.pkl
│       ├── linear_model.pkl
│       ├── ann_model.keras
│       ├── ann_history.pkl
│       └── metrics_summary.csv
│
├── src/
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── data_transformer.py
│   └── model_trainer.py
│
├── config.py
├── main.py
├── requirements.txt
├── eda.ipynb
├── LICENSE
└── README.md
Pipeline Flow
1. Data Ingestion
data_ingestion.py loads the raw dataset from:

bash

data/raw/clean_dataset.csv
2. Data Validation
data_validation.py performs:

schema validation
missing value checks
duplicate row checks
3. Data Transformation
data_transformer.py performs:

dropping unnecessary columns
removing duplicates
train-test split
log transformation of target (price)
ordinal encoding for class and stops
one-hot encoding for nominal categorical columns
standard scaling for numerical columns
4. Model Training
model_trainer.py trains:

Linear Regression
ANN using TensorFlow/Keras
5. Evaluation
Both models are evaluated after reversing the log transformation so results remain interpretable in the original price scale.

Model Performance
Current saved results from metrics_summary.csv:

Model	R2 Score	RMSE	MAE
Linear Regression	0.8792	7877.97	4616.97
ANN	0.9676	4077.18	2174.98
The ANN performs significantly better than the baseline Linear Regression model.

Requirements
Install dependencies using:

bash

pip install -r requirements.txt
Main dependencies:

pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
jupyter
ipykernel
statsmodels
How to Run
Run the full pipeline from the project root:

bash

python main.py
Notes
The dataset file should be present at data/raw/clean_dataset.csv
Processed files and trained model artifacts are saved inside data/processed/
eda.ipynb contains exploratory data analysis used before building the pipeline
Future Improvements
add better experiment tracking
add hyperparameter tuning
add model persistence/loading for inference
add a prediction script or web app interface
improve project documentation and exception handling
License
This project is licensed under the terms of the LICENSE file included in this repository.