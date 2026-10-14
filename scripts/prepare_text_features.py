
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.text_features import build_text_features

INPUT_PATH = PROJECT_ROOT / "data" / "fake_job_postings.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "text_features.csv"

def combine_job_text(row):
    columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "location",
        "salary_range",
        "employment_type",
        "required_experience",
        "required_education",
    ]

    sections = []
    for column in columns:
        value = str(row[column]).strip()
        if value:
            sections.append(f"{column.replace('_', ' ').title()}: {value}")

    return "\n".join(sections)


def main():
    df = pd.read_csv(INPUT_PATH).fillna("")
    records = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Extracting features"
    ):
        features = build_text_features(combine_job_text(row))
        features["fraudulent"] = int(row["fraudulent"])
        records.append(features)

    features_df = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset saved:", OUTPUT_PATH)
    print("Dataset shape:", features_df.shape)
    print("Missing values:", features_df.isnull().sum().sum())
    print(features_df["fraudulent"].value_counts())


if __name__ == "__main__":
    main()
