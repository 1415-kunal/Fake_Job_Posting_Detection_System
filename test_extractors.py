
from src.nlp.extractors import (
    extract_job_title,
    extract_company,
    extract_location,
    extract_salary,
    extract_experience,
    extract_education,
    extract_employment_type,
    extract_skills,
)

functions = {
    "title": extract_job_title,
    "company": extract_company,
    "location": extract_location,
    "salary": extract_salary,
    "experience": extract_experience,
    "education": extract_education,
    "employment_type": extract_employment_type,
    "skills": extract_skills,
}

print("EMPTY INPUT TEST")
for name, func in functions.items():
    print(f"{name}: {func('')}")

print("\nSECOND JOB POST TEST")

text = (
    "Data Scientist at BrightData Solutions. "
    "Location: Bengaluru, Karnataka. "
    "Salary: 8-12 LPA. "
    "Requires 2-4 years of experience. "
    "Full-time role. "
    "Skills: Python, SQL, Pandas, Machine Learning."
)

for name, func in functions.items():
    print(f"{name}: {func(text)}")


import pandas as pd
from src.features.text_features import build_text_features

job_text = """
Senior AI Engineer at TechNova, Pune.
Salary: ₹12,00,000 - ₹18,00,000 per year.
Requires Python, Machine Learning, FastAPI and Docker.
Full-time role requiring 3+ years of experience.
"""

features = build_text_features(job_text)
features_df = pd.DataFrame([features])
print("Missing feature names:")
print(features_df.columns[features_df.isnull().any()].tolist())

print("Total features:", len(features))
print("Missing values:", features_df.isnull().sum().sum())
print(
    "Infinite values:",
    features_df.select_dtypes(include="number")
    .isin([float("inf"), float("-inf")]).sum().sum(),
)
