
import re

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


def build_text_features(job_text: str) -> dict:
    """Extract reproducible features from one complete job posting."""

    if not isinstance(job_text, str) or not job_text.strip():
        raise ValueError("job_text must be a non-empty string.")

    text = re.sub(r"\s+", " ", job_text).strip()
    text_lower = text.lower()

    # NLP extraction
    title = extract_job_title(text)
    company = extract_company(text)
    location = extract_location(text)
    salary = extract_salary(text)
    experience = extract_experience(text)
    education = extract_education(text)
    employment_type = extract_employment_type(text)
    skills = extract_skills(text) or []

    # Salary features
    salary_numbers = []

    if salary:
        salary_numbers = [
            float(value.replace(",", ""))
            for value in re.findall(r"\d[\d,]*(?:\.\d+)?", salary)
        ]

    salary_min = salary_numbers[0] if salary_numbers else 0.0
    salary_max = salary_numbers[1] if len(salary_numbers) > 1 else salary_min
    salary_is_range = int(len(salary_numbers) > 1)

    # Experience and education
    experience_text = (experience or "").lower()
    education_text = " ".join(education or []).lower()

    is_entry_level = int(
        any(
            term in experience_text
            for term in ["fresher", "entry-level", "no experience"]
        )
    )
    is_internship_experience = int("intern" in experience_text)

    # Location indicators
    location_text = (location or "").lower()
    is_remote = int(
        "remote" in location_text
        or bool(re.search(r"\bremote\b", text_lower))
    )
    is_hybrid = int(
        "hybrid" in location_text
        or bool(re.search(r"\bhybrid\b", text_lower))
    )

    # Skill indicators
    skills_lower = {skill.lower() for skill in skills}

    has_python = int("python" in skills_lower)
    has_ml_skill = int(
        any("machine learning" in skill for skill in skills_lower)
    )
    has_data_skill = int(
        bool(
            skills_lower.intersection(
                {"pandas", "numpy", "data science", "sql"}
            )
        )
    )

    technical_terms = {
        "python", "java", "c++", "sql", "machine learning",
        "deep learning", "fastapi", "docker", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "javascript",
        "html", "css", "git", "aws", "azure",
    }

    has_technical_skills = int(
        bool(skills_lower.intersection(technical_terms))
    )

    # Text length and section headings
    words = re.findall(r"\b\w+\b", text)

    sections = re.findall(
        r"(?im)^\s*(?:requirements|responsibilities|"
        r"qualifications|benefits|about us|about the company|"
        r"skills|what you'll do|what you will do)\s*:?\s*$",
        job_text,
    )

    # Information-presence indicators
    has_salary = int(salary is not None)
    has_experience = int(experience is not None)
    has_education = int(bool(education))
    has_employment_type = int(bool(employment_type))
    has_location = int(location is not None)
    has_company_extracted = int(company is not None)

    missing_important_info_count = sum(
        [
            not has_salary,
            not has_experience,
            not has_education,
            not has_employment_type,
            not has_location,
            not has_company_extracted,
        ]
    )

    # Return extracted information and model features
    return {
        "job_title": title,
        "company_name": company,
        "location": location,
        "salary": salary,
        "experience": experience,
        "education": education or [],
        "employment_type": employment_type,
        "skills": skills,
        "job_text_char_length": len(text),
        "job_text_word_count": len(words),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_is_range": salary_is_range,
        "salary_midpoint": (salary_min + salary_max) / 2,
        "salary_range_width": salary_max - salary_min,
        "has_location": has_location,
        "is_remote": is_remote,
        "is_hybrid": is_hybrid,
        "skill_count": len(skills),
        "has_technical_skills": has_technical_skills,
        "has_python": has_python,
        "has_ml_skill": has_ml_skill,
        "has_data_skill": has_data_skill,
        "is_entry_level": is_entry_level,
        "is_internship_experience": is_internship_experience,
        "has_education_requirement": has_education,
        "is_bachelors": int(
            "bachelor" in education_text or "b.tech" in education_text
        ),
        "is_masters": int(
            "master" in education_text or "m.tech" in education_text
        ),
        "is_doctorate": int(
            any(term in education_text for term in ["ph.d", "phd", "doctorate"])
        ),
        "has_salary_extracted": has_salary,
        "has_experience_extracted": has_experience,
        "has_employment_type": has_employment_type,
        "has_company_extracted": has_company_extracted,
        "missing_important_info_count": int(missing_important_info_count),
        "text_section_count": len(sections),
    }
