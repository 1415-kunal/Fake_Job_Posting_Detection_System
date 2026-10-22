
import joblib
import pandas as pd
from scipy.sparse import hstack

from api.config import MODEL_PATH


# Load the saved model components once
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
vectorizer = saved_model["vectorizer"]
numeric_pipeline = saved_model["numeric_pipeline"]
numeric_columns = saved_model["numeric_columns"]


def predict_job_fraud(
    job_text: str,
    telecommuting: int = 0,
    has_company_logo: int = 0,
    has_questions: int = 0,
) -> dict:

    text = job_text.strip()

    if not text:
        raise ValueError("Job posting text cannot be empty.")

    numeric_values = pd.DataFrame([{
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
    }])[numeric_columns]

    text_features = vectorizer.transform([text.lower()])
    numeric_features = numeric_pipeline.transform(numeric_values)

    features = hstack(
        [text_features, numeric_features]
    ).tocsr()

    fraud_score = float(model.predict_proba(features)[0, 1])

    return {
        "prediction": (
            "Potentially Fraudulent"
            if fraud_score >= 0.5
            else "Likely Genuine"
        ),
        "fraud_score": round(fraud_score, 4),
        "fraud_score_percent": round(fraud_score * 100, 2),
        "warning": (
            "This is a model estimate, not proof that a "
            "job posting is genuine or fraudulent."
        ),
    }
