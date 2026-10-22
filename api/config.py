
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Model path
MODEL_PATH = BASE_DIR / "models" / "job_fraud_tfidf_model.joblib"

# API information
API_TITLE = "Job Posting Fraud Detection API"
API_DESCRIPTION = (
    "API for predicting whether a job posting is potentially fraudulent "
    "using a TF-IDF machine learning model."
)
API_VERSION = "1.0.0"
