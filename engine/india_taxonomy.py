"""
india_taxonomy.py — Indian skill taxonomy additions and text processing tools.
Handles degree variations, local synonyms, Hinglish terms, and Indian PII redaction.
"""
import re

INDIA_SKILL_ADDITIONS = {
    "b.tech": ["b.tech", "btech", "b tech", "bachelor of technology", "b. tech", "be", "b.e", "bachelor of engineering"],
    "m.tech": ["m.tech", "mtech", "m tech", "master of technology", "m. tech", "me", "m.e", "master of engineering"],
    "java": ["java", "core java", "advanced java"],
    "spring boot": ["spring boot", "springboot", "spring"],
    "hibernate": ["hibernate"],
    ".net": [".net", "dotnet", "dot net"],
    "c#": ["c#", "csharp"],
    "asp.net": ["asp.net", "asp.net mvc"],
    "angular": ["angular", "angularjs", "angular.js"],
    "react": ["react", "reactjs", "react.js"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "machine learning": ["ml", "machine learning"],
    "natural language processing": ["nlp", "natural language processing"],
    "python": ["python", "py"],
    "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "sql": ["sql", "mysql", "postgresql", "postgres"],
    "mongodb": ["mongodb", "mongo"],
    "agile": ["agile", "scrum", "kanban"],
    "git": ["git", "github", "gitlab"],
    "flutter": ["flutter"],
    "react native": ["react native", "react-native"],
    "entry level": ["fresher", "entry level", "passout", "graduate"]
}

def normalize_indian_resume_text(text: str) -> str:
    """
    Lowercases the input text and normalizes Hinglish/Indian specific terms:
    - "fresher" -> "entry level"
    - "b tech" -> "b.tech"
    - "dot net" -> ".net"
    - "passout" -> "graduate"
    """
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    replacements = {
        "fresher": "entry level",
        "b tech": "b.tech",
        "dot net": ".net",
        "passout": "graduate"
    }
    for old, new in replacements.items():
        text_lower = text_lower.replace(old, new)
    return text_lower

def redact_indian_pii(text: str) -> str:
    """
    Redacts +91 phone numbers and 12-digit Aadhaar number patterns.
    """
    if not isinstance(text, str):
        return ""
    # Redact +91 phone numbers (e.g. +91 9876543210, +91-9876543210, +919876543210)
    text = re.sub(r'\+91[-\s]?\d{10}', '[REDACTED_PHONE]', text)
    text = re.sub(r'\+91[-\s]?\d{5}[-\s]?\d{5}', '[REDACTED_PHONE]', text)
    
    # Redact Aadhaar: 12-digit numbers (usually grouped in 4-digit blocks like XXXX XXXX XXXX or continuous 12 digits)
    text = re.sub(r'\b\d{4}[-\s]\d{4}[-\s]\d{4}\b', '[REDACTED_AADHAAR]', text)
    text = re.sub(r'\b\d{12}\b', '[REDACTED_AADHAAR]', text)
    return text

INDIAN_TIER1_COLLEGES = [
    "iit", "indian institute of technology",
    "nit", "national institute of technology",
    "iiit", "indian institute of information technology",
    "bits pilani", "birla institute of technology",
    "iim", "indian institute of management",
    "iisc", "indian institute of science",
    "dtu", "delhi technological university",
    "nsut", "netaji subhas university of technology"
]

def has_indian_tier1_college(text: str) -> bool:
    """
    Checks if a candidate's resume contains Tier-1 Indian institutes.
    """
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for college in INDIAN_TIER1_COLLEGES:
        pattern = r"\b" + re.escape(college) + r"\b"
        if re.search(pattern, text_lower):
            return True
    return False

