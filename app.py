"""
Flask app for Flight Price Prediction.
"""

import os
import sys

from flask import Flask, jsonify, render_template, request


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference import InferenceError, predict_price


app = Flask(__name__)


FORM_OPTIONS = {
    "airline": ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
    "source_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "departure_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "stops": ["zero", "one", "two_or_more"],
    "arrival_time": ["Afternoon", "Early_Morning", "Evening", "Late_Night", "Morning", "Night"],
    "destination_city": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
    "class": ["Economy", "Business"],
}


def get_form_data(form) -> dict[str, str]:
    """Collect incoming form values into one payload."""
    return {
        "airline": form.get("airline", ""),
        "source_city": form.get("source_city", ""),
        "departure_time": form.get("departure_time", ""),
        "stops": form.get("stops", ""),
        "arrival_time": form.get("arrival_time", ""),
        "destination_city": form.get("destination_city", ""),
        "class": form.get("class", ""),
        "duration": form.get("duration", ""),
        "days_left": form.get("days_left", ""),
    }


@app.route("/", methods=["GET"])
def home():
    """Render the prediction form."""
    return render_template(
        "index.html",
        options=FORM_OPTIONS,
        prediction=None,
        error=None,
        form_data={},
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission and render the prediction result."""
    form_data = get_form_data(request.form)

    try:
        result = predict_price(form_data, model_type="ann")
        return render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=result,
            error=None,
            form_data=form_data,
        )
    except InferenceError as error:
        return render_template(
            "index.html",
            options=FORM_OPTIONS,
            prediction=None,
            error=str(error),
            form_data=form_data,
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Optional JSON endpoint for programmatic prediction."""
    payload = request.get_json(silent=True) or {}

    try:
        result = predict_price(payload, model_type="ann")
        return jsonify(result), 200
    except InferenceError as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(debug=True)
