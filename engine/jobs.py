"""
jobs.py — manages persistent job openings repository (saved inside data/job_openings/).
"""
import os
import json
import re
from typing import Optional


# Setup persistent directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_DIR = os.path.join(BASE_DIR, "data", "job_openings")
SAMPLE_JD_DIR = os.path.join(BASE_DIR, "data", "sample_jd")

os.makedirs(JOBS_DIR, exist_ok=True)


def get_job_slug(title: str) -> str:
    """Helper to generate a clean, filename-safe slug from a job title."""
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower().strip())
    slug = slug.strip("_")
    return slug if slug else "job_opening"


def save_job_opening(
    title: str,
    jd_text: str,
    mandatory_skills: list,
    required_years: float,
    required_degree: int
) -> str:
    """Saves a job opening to the persistent storage directory."""
    slug = get_job_slug(title)
    
    # Avoid blank slugs collapsing on same file
    filepath = os.path.join(JOBS_DIR, f"{slug}.json")
    counter = 1
    while os.path.exists(filepath):
        filepath = os.path.join(JOBS_DIR, f"{slug}_{counter}.json")
        counter += 1

    job_data = {
        "slug": os.path.basename(filepath).replace(".json", ""),
        "title": title,
        "jd_text": jd_text,
        "mandatory_skills": mandatory_skills,
        "required_years": required_years,
        "required_degree": required_degree
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(job_data, f, indent=4, ensure_ascii=False)

    return job_data["slug"]


def delete_job_opening(slug: str) -> bool:
    """Deletes a saved job opening by its slug."""
    filepath = os.path.join(JOBS_DIR, f"{slug}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def bootstrap_default_jobs():
    """Bootstraps default job openings from data/sample_jd if the repository is empty."""
    if not os.path.exists(SAMPLE_JD_DIR):
        return

    # Check if there are already job openings saved
    existing_openings = [f for f in os.listdir(JOBS_DIR) if f.endswith(".json")]
    if existing_openings:
        return  # already bootstrapped or has user openings

    # Import parser helpers to analyze required years, degree, and skills
    from engine.scorer import extract_required_years, extract_highest_degree_level, extract_explicit_jd_skills

    for filename in sorted(os.listdir(SAMPLE_JD_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(SAMPLE_JD_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Format name nicely
                raw_name = filename.replace(".txt", "")
                if raw_name == "backend_python_developer":
                    title = "Backend Python Developer"
                elif raw_name == "frontend_react_developer":
                    title = "Frontend React Developer"
                elif raw_name == "devops_sre_engineer":
                    title = "DevOps SRE Engineer"
                elif raw_name == "data_scientist_ml":
                    title = "Data Scientist (Machine Learning)"
                elif raw_name == "product_manager":
                    title = "Product Manager"
                elif raw_name == "security_engineer":
                    title = "Cybersecurity Analyst / Security Engineer"
                elif raw_name == "mobile_developer":
                    title = "Mobile iOS/Android Developer"
                else:
                    title = raw_name.replace("_", " ").title()

                req_years = extract_required_years(content)
                req_deg = extract_highest_degree_level(content)
                
                # Pick some logical default mandatory skills from explicit skills (e.g. first 2 skills found)
                skills = sorted(list(extract_explicit_jd_skills(content)))
                mandatory = [skills[0]] if skills else []

                save_job_opening(
                    title=title,
                    jd_text=content,
                    mandatory_skills=mandatory,
                    required_years=req_years,
                    required_degree=req_deg
                )
            except Exception:
                pass  # resilient bootstrapping


def load_all_job_openings() -> list[dict]:
    """Loads and returns all active saved job openings, bootstrapping if empty."""
    bootstrap_default_jobs()
    
    openings = []
    if os.path.exists(JOBS_DIR):
        for filename in sorted(os.listdir(JOBS_DIR)):
            if filename.endswith(".json"):
                filepath = os.path.join(JOBS_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        openings.append(data)
                except Exception:
                    pass
    return openings
