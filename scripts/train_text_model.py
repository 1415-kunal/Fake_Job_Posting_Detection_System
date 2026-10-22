from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "job_postings_features.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "job_fraud_text_model.joblib"

FEATURE_COLUMNS = [
    "title_char_length",
    "description_char_length",
    "has_company_profile",
    "has_salary_range",
    "has_location",
    "is_remote",
    "skill_count",
    "has_python",
    "missing_important_info_count",
]
def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Genuine", "Fraudulent"],
        zero_division=0,
    ))

    print(
        "PR-AUC:",
        round(average_precision_score(y_test, probabilities), 4),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModel saved to:", MODEL_PATH)
    print("Features used:", len(FEATURE_COLUMNS))


if __name__ == "__main__":
    main()