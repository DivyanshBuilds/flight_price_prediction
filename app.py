"""
app.py
------
Flask web application for Flight Price Prediction.

Routes:
    GET  /           - Render the prediction form
    POST /predict   - Handle form submission and return prediction
    POST /api/predict - JSON API endpoint for programmatic access
"""

import os
import sys
from flask import Flask, jsonify, render_template, request

# Add src/ to path so we can import inference
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import InferenceError, predict_price


app = Flask(__name__)


# ---------------------------------------------------------------------
# Form dropdown options (must match training data categories exactly)
# ---------------------------------------------------------------------
FORM_OPTIONS = {
    "airline": ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
    "source_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "destination_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "departure_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "arrival_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "stops": ["zero", "one", "two_or_more"],
    "class": ["Economy", "Business"],
}


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    """Render the prediction form (initial page load)."""
    return render_template(
        "index.html",
        options=FORM_OPTIONS,
        prediction=None,
        error=None,
        form_data={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form submission, run prediction, and return result.
    
    On success: renders form with prediction displayed
    On error: renders form with error message and preserves user input
    """
    # Collect all form fields into a dict
    form_data = {
        "airline": request.form.get("airline", ""),
        "source_city": request.form.get("source_city", ""),
        "destination_city": request.form.get("destination_city", ""),
        "departure_time": request.form.get("departure_time", ""),
        "arrival_time": request.form.get("arrival_time", ""),
        "stops": request.form.get("stops", ""),
        "class": request.form.get("class", ""),
        "duration": request.form.get("duration", ""),
        "days_left": request.form.get("days_left", ""),
    }

    try:
        # Run prediction through inference.py
        result = predict_price(form_data)
        
        return render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=result,
            error=None,
            form_data=form_data,
        )
        
    except InferenceError as e:
        # Validation or model error — show friendly message
        return render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=None,
            error=str(e),
            form_data=form_data,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic access.
    
    Request body (JSON):
        {
            "airline": "Vistara",
            "source_city": "Delhi",
            "destination_city": "Mumbai",
            "departure_time": "Morning",
            "arrival_time": "Afternoon",
            "stops": "one",
            "class": "Economy",
            "duration": 2.17,
            "days_left": 15
        }
    
    Response (JSON):
        {
            "predicted_price": 5953.42,
            "model_name": "ANN (R² = 96.7%)"
        }
    """
    payload = request.get_json(silent=True) or {}

    try:
        result = predict_price(payload)
        return jsonify(result), 200
        
    except InferenceError as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)