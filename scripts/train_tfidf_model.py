
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Project paths
DATA_PATH = Path("data/fake_job_postings.csv")
MODEL_PATH = Path("models/job_fraud_tfidf_model.joblib")

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

# Load data
df = pd.read_csv(DATA_PATH)

df["full_text"] = (
    df[TEXT_COLUMNS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

X_text = df["full_text"]
X_numeric = df[NUMERIC_COLUMNS].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)
y = df["fraudulent"].astype(int)

# Split before fitting the vectorizer
X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = (
    train_test_split(
        X_text,
        X_numeric,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
)

# Convert job-posting text into TF-IDF features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
    strip_accents="unicode",
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Scale numeric features and combine them with text
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

X_train_numeric = numeric_pipeline.fit_transform(X_train_num)
X_test_numeric = numeric_pipeline.transform(X_test_num)

X_train = hstack([X_train_tfidf, X_train_numeric]).tocsr()
X_test = hstack([X_test_tfidf, X_test_numeric]).tocsr()

# Train a class-imbalance-aware model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

print("Training TF-IDF model...")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Genuine", "Fraudulent"],
    digits=4,
    zero_division=0,
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(
    "Fraud PR-AUC:",
    round(average_precision_score(y_test, y_prob), 4),
)

# Save all components needed for future predictions
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer,
        "numeric_pipeline": numeric_pipeline,
        "numeric_columns": NUMERIC_COLUMNS,
        "text_columns": TEXT_COLUMNS,
    },
    MODEL_PATH,
)

print(f"\nModel saved to: {MODEL_PATH}")
