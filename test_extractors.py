
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
