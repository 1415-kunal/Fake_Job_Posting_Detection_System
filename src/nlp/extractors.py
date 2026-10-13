
import re
import spacy

from .skill_dictionary import SKILL_ALIASES

import os
from gliner import GLiNER

# Temporary workaround for the OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

gliner_model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1"
)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re

from gliner import GLiNER
from .skill_dictionary import SKILL_ALIASES


# Load spaCy for location extraction.
nlp = spacy.load("en_core_web_sm")

gliner_model = GLiNER.from_pretrained(
    "urchade/gliner_small-v2.1"
)

def extract_job_title(text):
    """Extract a job title using GLiNER with rule-based fallback."""

    if not text or not text.strip():
        return None

    # 1. Try GLiNER first
    try:
        entities = gliner_model.predict_entities(
            text,
            ["job title"],
            threshold=0.4,
        )

        titles = [
            entity["text"].strip()
            for entity in entities
            if entity["label"].lower() == "job title"
            and entity["text"].strip()
        ]

        if titles:
            # Prefer the longest detected title
            title = max(titles, key=len)
            return re.sub(r"\s+", " ", title).strip()

    except Exception as error:
        print(f"GLiNER title extraction warning: {error}")

    # 2. Rule-based fallback for explicitly labelled titles
    lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in text.splitlines()
        if line.strip()
    ]

    label_patterns = [
        r"^\s*(?:job\s*)?title\s*[:\-]\s*(.+)$",
        r"^\s*position\s*[:\-]\s*(.+)$",
        r"^\s*role\s*[:\-]\s*(.+)$",
        r"^\s*job\s+role\s*[:\-]\s*(.+)$",
    ]

    for line in lines[:15]:
        for pattern in label_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)

            if match:
                title = match.group(1).strip()
                title = re.split(r"[.!?]", title, maxsplit=1)[0].strip()

                if title:
                    return title

    # 3. Existing candidate-scoring fallback
    title_keywords = [
        "engineer", "developer", "scientist", "analyst",
        "manager", "designer", "architect", "consultant",
        "specialist", "administrator", "intern", "researcher",
        "director", "lead", "executive", "officer",
        "associate", "technician", "accountant", "recruiter",
        "tester", "devops", "software", "machine learning",
        "product",
    ]

    candidates = []

    for index, line in enumerate(lines[:10]):
        lower_line = line.lower()

        if len(line.split()) > 8:
            continue

        if any(
            phrase in lower_line
            for phrase in [
                "is hiring", "are hiring", "we are looking",
                "we're looking", "is looking for",
                "are looking for", "please apply", "apply now",
            ]
        ):
            continue

        score = 0

        if 1 <= len(line.split()) <= 6:
            score += 3

        if any(keyword in lower_line for keyword in title_keywords):
            score += 5

        score += max(0, 3 - index)

        candidates.append((score, index, line))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        score, _, title = candidates[0]

        if score >= 5:
            return title.strip()

    return None


def extract_company(text):
    """Extract a company name using GLiNER with rule-based fallback."""

    if not text or not text.strip():
        return None

    invalid_names = {
        "we", "we are", "we're", "our", "our company",
        "the company", "company", "they", "they are",
        "the team", "our team",
    }

    def clean_company(value):
        if not value:
            return None

        value = value.strip()
        value = re.sub(r'^[\'"“”‘’]+|[\'"“”‘’]+$', "", value)
        value = re.sub(r"[.,;:]+$", "", value).strip()

        if not value or value.casefold() in invalid_names:
            return None

        if len(value.split()) > 10:
            return None

        return value

    # 1. Try GLiNER first
    try:
        entities = gliner_model.predict_entities(
            text,
            ["company name"],
            threshold=0.4,
        )

        companies = [
            entity["text"].strip()
            for entity in entities
            if entity["label"].lower() == "company name"
            and entity["text"].strip()
        ]

        if companies:
            company = clean_company(max(companies, key=len))

            if company:
                return company

    except Exception as error:
        print(f"GLiNER company extraction warning: {error}")

    # 2. Rule-based fallback
    lines = [
        re.sub(r"\s+", " ", line.strip())
        for line in text.splitlines()
        if line.strip()
    ]

    label_patterns = [
        r"^\s*company\s*[:\-]\s*(.+?)\s*$",
        r"^\s*company\s+name\s*[:\-]\s*(.+?)\s*$",
        r"^\s*employer\s*[:\-]\s*(.+?)\s*$",
        r"^\s*organization\s*[:\-]\s*(.+?)\s*$",
        r"^\s*organisation\s*[:\-]\s*(.+?)\s*$",
    ]

    for line in lines[:15]:
        for pattern in label_patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)

            if match:
                company = clean_company(match.group(1))

                if company:
                    return company

    # Company mentioned after "at", e.g. "Senior Engineer at TechNova"
    match = re.search(
        r"\b(?:at|with|join)\s+"
        r"([A-Z][A-Za-z0-9&.'’\-]*(?:\s+"
        r"[A-Z][A-Za-z0-9&.'’\-]*){0,5})",
        text,
    )

    if match:
        company = clean_company(match.group(1))

        if company:
            return company

    # Company as the subject of a hiring sentence
    for line in lines[:15]:
        match = re.search(
            r"^([A-Z][A-Za-z0-9&.,'’\- ]{1,60}?)\s+"
            r"(?:is hiring|is looking for|is seeking|"
            r"is recruiting|provides|offers|develops|builds)\b",
            line,
        )

        if match:
            company = clean_company(match.group(1))

            if company:
                return company

    return None






from transformers import pipeline

# Load once when the module is imported, not on every prediction.
location_ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


def extract_location_ner(text):
    """Detect location entities in job-posting text using BERT NER."""
    if not text or not text.strip():
        return None

    entities = location_ner(text)
    locations = []

    for entity in entities:
        if entity["entity_group"] == "LOC":
            location = entity["word"].strip()

            if location and location.lower() not in {
                item.lower() for item in locations
            }:
                locations.append(location)

    return ", ".join(locations) if locations else None



def extract_location(text):
    """Extract the most relevant work location from a job posting."""

    if not text or not text.strip():
        return None

    # 1. Check explicit location labels first
    explicit_patterns = [
        r"(?im)^\s*location\s*:\s*(.+)$",
        r"(?im)^\s*job location\s*:\s*(.+)$",
        r"(?im)^\s*work location\s*:\s*(.+)$",
        r"(?im)^\s*勤務地\s*:\s*(.+)$",
    ]

    for pattern in explicit_patterns:
        match = re.search(pattern, text)

        if match:
            location = match.group(1).strip()
            location = re.split(r"[.!?;\n]", location, maxsplit=1)[0].strip()

            if location:
                return location

    # 2. Check remote work
    remote_patterns = [
        r"\bfully remote\b",
        r"\b100%\s*remote\b",
        r"\bremote position\b",
        r"\bremote role\b",
        r"\bremote work\b",
        r"\bwork remotely\b",
        r"\bwork from anywhere\b",
        r"\bremote\b",
    ]

    for pattern in remote_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "Remote"

    # 3. Check hybrid work
    hybrid_patterns = [
        r"\bhybrid work\b",
        r"\bhybrid role\b",
        r"\bhybrid position\b",
        r"\bhybrid\b",
    ]

    for pattern in hybrid_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return "Hybrid"

    # 4. Prioritize phrases that describe the actual work location
    work_location_patterns = [
        r"\bthis role is based in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bthe role is based in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\brole is based in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bposition is based in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bthe selected candidate will work in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bthe candidate will work in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bwork location is\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\boffice location is\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bjob is based in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
        r"\bjoin our team in\s+([A-Za-z][A-Za-z\s,.-]{1,60})",
    ]

    for pattern in work_location_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            location = match.group(1).strip()

            # Stop at the end of the sentence
            location = re.split(r"[.!?;\n]", location, maxsplit=1)[0].strip()

            # Remove trailing punctuation
            location = re.sub(r"[,]+$", "", location).strip()

            # Avoid capturing the rest of a sentence
            location = re.split(
                r"\s+(?:and|but|with|for|where|who|while)\s+",
                location,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

            if location:
                return location

    # 5. Use BERT NER only as a fallback
    return extract_location_ner(text)





def extract_salary(text):
    """Extract salary information using regex and contextual validation."""

    if not text or not text.strip():
        return None

    text = re.sub(r"\s+", " ", text).strip()

    # Ignore vague salary statements
    if re.search(
        r"\b(?:salary|compensation|pay)\b.{0,30}"
        r"\b(?:competitive|negotiable|not disclosed)\b",
        text,
        flags=re.IGNORECASE,
    ):
        return None

    amount = r"\d[\d,]*(?:\.\d+)?"
    currency = r"(?:₹|Rs\.?|INR|USD|EUR|GBP|\$|€|£)"
    period = (
        r"(?:per\s+(?:year|annum|month|week|hour)"
        r"|/\s*(?:year|month|week|hour)"
        r"|annually|monthly|weekly|hourly)"
    )

    # 1. Prefer salary ranges expressed in LPA or lakhs per annum.
    # Check this BEFORE general numeric ranges so the unit is preserved.
    lpa_range = re.search(
        rf"(?<!\w){amount}\s*(?:-|–|—|to)\s*{amount}"
        rf"\s*(?:LPA|lakhs?\s+per\s+annum|lakhs?\s+annually)\b",
        text,
        flags=re.IGNORECASE,
    )
    if lpa_range:
        return lpa_range.group(0).strip()

    # 2. Salary ranges explicitly associated with salary/pay/compensation.
    range_pattern = (
        rf"(?<!\w)(?:{currency}\s*)?{amount}"
        rf"\s*(?:-|–|—|to)\s*"
        rf"(?:{currency}\s*)?{amount}"
        rf"(?:\s*{period})?"
    )

    label_pattern = re.search(
        rf"\b(?:salary|compensation|pay)\b"
        rf"(?:\s+in\s+{currency})?\s*:?\s*"
        rf"({range_pattern})",
        text,
        flags=re.IGNORECASE,
    )
    if label_pattern:
        return label_pattern.group(1).strip()

    # 3. Currency-based salary ranges.
    currency_range = re.search(
        rf"{currency}\s*{amount}\s*(?:-|–|—|to)\s*"
        rf"(?:{currency}\s*)?{amount}"
        rf"(?:\s*{period})?",
        text,
        flags=re.IGNORECASE,
    )
    if currency_range:
        return currency_range.group(0).strip()

    # 4. A single salary amount with currency.
    single_salary = re.search(
        rf"{currency}\s*{amount}(?:\s*{period})?",
        text,
        flags=re.IGNORECASE,
    )
    if single_salary:
        return single_salary.group(0).strip()

    return None





def extract_experience(text):
    """Extract experience requirements from a complete job posting."""

    if not text or not text.strip():
        return None

    text = re.sub(r"\s+", " ", text).strip()

    # 1. Explicit experience ranges, e.g. 1-2 years, 3 to 5 years
    range_pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:-|–|—|to)\s*"
        r"\d+(?:\.\d+)?\s*"
        r"(?:years?|yrs?|months?|mos?)"
        r"(?:\s+of)?\s+(?:relevant\s+)?experience\b"
    )

    match = re.search(range_pattern, text, flags=re.IGNORECASE)

    if match:
        return match.group(0).strip()

    # 2. Experience minimums, e.g. 3+ years of experience
    minimum_pattern = (
        r"\b\d+(?:\.\d+)?\s*\+\s*"
        r"(?:years?|yrs?|months?|mos?)"
        r"(?:\s+of)?\s+(?:relevant\s+)?experience\b"
    )

    match = re.search(minimum_pattern, text, flags=re.IGNORECASE)

    if match:
        return match.group(0).strip()

    # 3. Experience requirements, e.g. 2 years experience
    standard_pattern = (
        r"\b(?:at\s+least\s+)?\d+(?:\.\d+)?\s*"
        r"(?:years?|yrs?|months?|mos?)"
        r"(?:\s+of)?\s+(?:relevant\s+)?experience\b"
    )

    match = re.search(standard_pattern, text, flags=re.IGNORECASE)

    if match:
        return match.group(0).strip()

    # 4. Common entry-level and fresher descriptions
    experience_levels = [
        (r"\b(?:freshers?|fresh\s+graduates?|recent\s+graduates?)\b",
         "Fresher"),
        (r"\bentry[\s-]?level\b", "Entry-level"),
        (r"\bno\s+(?:prior\s+)?experience\s+(?:required|needed)\b",
         "No experience required"),
        (r"\b(?:internship|intern)\s+experience\b",
         "Internship experience"),
    ]

    for pattern, label in experience_levels:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label

    return None


def extract_education(text):
    """Extract unique education-related mentions from a job posting."""

    if not text or not text.strip():
        return None

    patterns = [
        # Bachelor's, Master's, and Associate degrees
        (
            r"\b(?:bachelor(?:'s)?|master(?:'s)?|associate(?:'s)?)"
            r"\s+(?:degree|qualification)\b",
            re.IGNORECASE,
        ),

        # PhD, Ph.D., doctorate, and doctoral qualifications
        (
            r"\b(?:Ph\.?\s*D\.?|doctorate|doctoral)"
            r"(?:\s+(?:degree|qualification))?\b",
            re.IGNORECASE,
        ),

        # Degree or diploma in a subject
        (
            r"\b(?:degree|diploma)\s+in\s+"
            r"[A-Za-z][A-Za-z\s,&/-]*?"
            r"(?=\s*(?:[.,;]|and\b|or\b|preferred\b|required\b|"
            r"may\b|can\b|is\b|are\b|$))",
            re.IGNORECASE,
        ),

        # Common degree abbreviations
        (
            r"(?<![A-Za-z])"
            r"(?:B\.Tech|B\.E\.|M\.Tech|M\.E\.|B\.Sc|M\.Sc|"
            r"BCA|MCA|MBA)(?![A-Za-z])",
            re.IGNORECASE,
        ),

        # Diploma
        (
            r"\b(?:advanced\s+)?diploma\b",
            re.IGNORECASE,
        ),

        # High school qualifications
        (
            r"\b(?:high\s+school(?:\s+diploma)?|"
            r"secondary\s+school\s+certificate|12th\s+pass)\b",
            re.IGNORECASE,
        ),

        # Any degree
        (
            r"\bany\s+(?:bachelor(?:'s)?|master(?:'s)?|college)\s+degree\b",
            re.IGNORECASE,
        ),

        # No degree or formal education required
        (
            r"\bno\s+(?:degree|formal\s+education)\s+"
            r"(?:is\s+)?required\b",
            re.IGNORECASE,
        ),

        # Certifications
        (
            r"\b(?:professional|relevant|industry-recognized)\s+"
            r"certification\b|\bcertification\s+required\b",
            re.IGNORECASE,
        ),
    ]

    candidates = []

    for pattern, flags in patterns:
        for match in re.finditer(pattern, text, flags=flags):
            candidates.append({
                "text": match.group(0).strip(),
                "start": match.start(),
                "end": match.end(),
            })

    # Prefer longer matches when patterns overlap
    candidates.sort(
        key=lambda item: (
            item["start"],
            -(item["end"] - item["start"]),
        )
    )

    selected = []

    for candidate in candidates:
        overlaps = any(
            candidate["start"] < existing["end"]
            and candidate["end"] > existing["start"]
            for existing in selected
        )

        if not overlaps:
            selected.append(candidate)

    # Remove duplicate mentions while preserving original wording
    results = []
    seen = set()

    for item in sorted(selected, key=lambda item: item["start"]):
        value = item["text"].strip()
        key = value.casefold()

        if key not in seen:
            results.append(value)
            seen.add(key)

    return results if results else None



def extract_employment_type(text):
    """Extract unique employment types from a complete job posting."""

    if not text or not text.strip():
        return None

    patterns = [
        (r"\bfull[\s-]?time\b", "Full-time"),
        (r"\bpart[\s-]?time\b", "Part-time"),
        (r"\bcontract(?:ual)?\b", "Contract"),
        (r"\btemporary\b", "Temporary"),
        (r"\bfreelance(?:r|rs)?\b", "Freelance"),
        (r"\binternships?\b", "Internship"),
        (r"\bapprenticeships?\b", "Apprenticeship"),
        (r"\bpermanent\b", "Permanent"),
        (r"\bcasual employment\b", "Casual"),
        (r"\bself[\s-]?employed\b", "Self-employed"),
    ]

    results = []
    seen = set()

    for pattern, label in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            key = label.casefold()

            if key not in seen:
                results.append(label)
                seen.add(key)

    return results if results else None


def extract_skills(text, skill_aliases=SKILL_ALIASES):
    """Extract canonical skills from job-posting text."""
    if not text:
        return None

    found_skills = []

    for canonical_skill, aliases in skill_aliases.items():
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(alias) + r"(?!\w)"

            if re.search(pattern, text, flags=re.IGNORECASE):
                found_skills.append(canonical_skill)
                break

    return found_skills if found_skills else None
