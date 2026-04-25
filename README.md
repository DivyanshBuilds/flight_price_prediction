 ✈️ Flight Price Predictor

A machine learning web application that predicts domestic flight ticket prices in India using an Artificial Neural Network (ANN).

[![Python](https://img.shields.io/badge/Python-3.10.20-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📊 Project Overview

This project trains a neural network on **300,000+ flight records** from EaseMyTrip to predict ticket prices based on route, timing, class, and booking window. The trained model achieves **96.7% R²** on the test set and is deployed as an interactive web application.

**Live Demo:** [Coming Soon]

---

## 🎯 Features

- **High Accuracy:** ANN model with 96.7% R² score (MAE: ₹2,175)
- **Real-time Predictions:** Instant price estimates via web interface
- **Clean Pipeline:** End-to-end ML workflow from raw data to deployment
- **Interactive UI:** Modern, responsive design with intuitive form inputs
- **Production Ready:** Deployed with Flask + Gunicorn

---

## 📁 Project Structure
flight_price_prediction/
├── data/
│   ├── raw/
│   │   └── Clean_Dataset.csv          # Original dataset (300k+ flights)
│   └── processed/                      # Generated after running pipeline
│       ├── X_train.csv                 # Training features
│       ├── X_test.csv                  # Test features
│       ├── y_train.csv                 # Training targets (log-transformed)
│       └── y_test.csv                  # Test targets (log-transformed)
├── models/                              # Generated after training
│   ├── ann_model.keras                 # Trained neural network
│   ├── linear_model.pkl                # Baseline linear regression
│   ├── ohe.pkl                         # OneHotEncoder for categoricals
│   ├── ordinal_encoder_class.pkl       # Class encoder (Economy/Business)
│   ├── ordinal_encoder_stops.pkl       # Stops encoder (0/1/2+)
│   ├── scaler.pkl                      # StandardScaler for numerics
│   ├── ann_history.pkl                 # Training history
│   └── metrics_summary.csv             # Model performance metrics
├── src/
│   ├── data_ingestion.py               # Load raw data
│   ├── data_validation.py              # Schema and quality checks
│   ├── data_transformer.py             # Feature engineering pipeline
│   ├── model_trainer.py                # Train ANN and baseline models
│   └── inference.py                    # Prediction logic for Flask app
├── templates/
│   └── index.html                      # Web UI
├── static/
│   └── style.css                       # Styling
├── app.py                               # Flask application
├── main.py                              # Run full pipeline
├── requirements.txt                     # Python dependencies
├── Procfile                             # Deployment config (Render/Heroku)
├── runtime.txt                          # Python version for deployment
└── README.md

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/DivyanshBuilds/flight_price_prediction.git
cd flight_price_prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Training Pipeline

```bash
python main.py
```

This will:
- Validate and clean the raw data
- Perform feature engineering (encoding, scaling, log transformation)
- Train the ANN model (128→64→32→1 architecture)
- Save all artifacts to `data/processed/` and `models/`

### 4️⃣ Start the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🧠 Model Architecture

**Neural Network:**
Input (29 features)
↓
Dense(128, relu)
↓
Dense(64, relu)
↓
Dense(32, relu)
↓
Dense(1, linear)  → Log-transformed price

**Training Details:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE (in log-space)
- Batch Size: 256
- Early Stopping: Patience=10 on validation loss
- Train/Val/Test Split: 64%/16%/20%

---

## 📈 Model Performance

| Model | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| **ANN** | **0.9676** | **₹4,077** | **₹2,175** | **12.70%** |
| Linear Regression | 0.8792 | ₹7,878 | ₹4,617 | 12.70% |

**Per-Class Performance (ANN):**

| Class | R² | RMSE | MAE | MAPE |
|-------|-----|------|-----|------|
| Economy | 0.7738 | ₹1,770 | ₹1,006 | 14.43% |
| Business | 0.7249 | ₹6,813 | ₹4,757 | 8.88% |

---

## 🔍 Key Features Used

| Feature | Type | Encoding |
|---------|------|----------|
| **airline** | Categorical | OneHotEncoder (6 airlines) |
| **source_city** | Categorical | OneHotEncoder (6 cities) |
| **destination_city** | Categorical | OneHotEncoder (6 cities) |
| **departure_time** | Categorical | OneHotEncoder (6 time slots) |
| **arrival_time** | Categorical | OneHotEncoder (6 time slots) |
| **stops** | Ordinal | OrdinalEncoder (0/1/2+) |
| **class** | Ordinal | OrdinalEncoder (Economy=0, Business=1) |
| **duration** | Numerical | StandardScaler |
| **days_left** | Numerical | StandardScaler |

**Target:** `price` (log-transformed with `np.log1p`)

---

## 🛠️ Tech Stack

**Machine Learning:**
- TensorFlow/Keras 2.15.0
- scikit-learn 1.3.2
- NumPy, Pandas

**Web Framework:**
- Flask 3.0.0
- Gunicorn 21.2.0

**Deployment:**
- Render (Production)
- Local testing with Flask dev server

---

## 📊 Dataset

- **Source:** [Kaggle - Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
- **Size:** 300,153 rows → 297,940 after cleaning (2,213 duplicates removed)
- **Features:** 12 columns (9 used for modeling)
- **Target Range:** ₹1,105 – ₹123,071
- **Classes:** Economy (69%), Business (31%)


---

## 🔧 Development

### Run Tests

```bash
python src/inference.py
```

### Check Pipeline Outputs

```bash
# Verify data processing
ls data/processed/

# Verify model artifacts
ls models/
```

### Local Development

```bash
# Run in debug mode
python app.py

# Production mode
gunicorn app:app
```

---

## 📝 API Endpoints

### Web Interface
- **GET /** - Render prediction form
- **POST /predict** - Submit form and get prediction

### JSON API
- **POST /api/predict** - Programmatic access

**Example Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "Vistara",
    "source_city": "Delhi",
    "destination_city": "Mumbai",
    "departure_time": "Morning",
    "arrival_time": "Afternoon",
    "stops": "one",
    "class": "Economy",
    "duration": 2.17,
    "days_left": 15
  }'
```

**Example Response:**
```json
{
  "predicted_price": 5953.42,
  "model_name": "ANN (R² = 96.7%)"
}
```
---

## 📚 Future Improvements

- [ ] Add more features (booking source, fare class, seat availability)
- [ ] Experiment with ensemble models (XGBoost, LightGBM)
- [ ] Implement hyperparameter tuning (Grid/Random Search)
- [ ] Add model explainability (SHAP values)
- [ ] Create separate models for Economy/Business classes
- [ ] Real-time price tracking and alerts
- [ ] Mobile app version

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
## 👤 Author

**Divyansh**

- GitHub: [@DivyanshBuilds](https://github.com/DivyanshBuilds)

---

## 🙏 Acknowledgments

- Dataset provided by [Shubham Bathwal](https://www.kaggle.com/shubhambathwal) on Kaggle
- Inspired by real-world dynamic pricing challenges in aviation

---

## ⭐ Star This Repository

If you found this project helpful, please give it a ⭐️!

---
