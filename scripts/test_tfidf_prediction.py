
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/job_fraud_tfidf_model.joblib")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
vectorizer = saved["vectorizer"]
numeric_pipeline = saved["numeric_pipeline"]
numeric_columns = saved["numeric_columns"]


job_text = """
Urgent Work From Home Job! Guaranteed High Salary!

We are hiring freshers for easy online data entry work.
No experience, no interview, and no technical skills required.

Earn ₹8,000 per day with guaranteed income.
Selected candidates must pay a refundable registration fee
of ₹2,000 before receiving their offer letter.

Send your Aadhaar card, bank account details, and payment
receipt through WhatsApp immediately.

Limited vacancies! Apply today to secure your position.
"""


numeric_values = pd.DataFrame([{
    "telecommuting": 0,
    "has_company_logo": 1,
    "has_questions": 1,
}])[numeric_columns]

text_features = vectorizer.transform([job_text.lower()])
numeric_features = numeric_pipeline.transform(numeric_values)

from scipy.sparse import hstack

features = hstack([text_features, numeric_features]).tocsr()

probability = model.predict_proba(features)[0, 1]
prediction = "Potentially Fraudulent" if probability >= 0.5 else "Likely Genuine"

print("Prediction:", prediction)
print(f"Fraud score: {probability:.2%}")
