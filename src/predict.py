
from pathlib import Path
import json
import joblib
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "job_fraud_random_forest.joblib"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "job_fraud_threshold.json"

# Load model and threshold
model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold = json.load(file)["fraud_threshold"]


def predict_job(features: dict) -> dict:
    """Predict whether a structured job posting may be fraudulent."""

    input_df = pd.DataFrame([features])

    fraud_probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    prediction = (
        "Potentially Fraudulent"
        if fraud_probability >= threshold
        else "Genuine"
    )

    return {
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 4),
        "threshold": threshold
    }
