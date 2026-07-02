import pytest
from unittest.mock import MagicMock
from engine.scorer import (
    tokenize,
    extract_years_of_experience,
    extract_required_years,
    extract_highest_degree_level,
    normalize_skills_in_text,
    cosine_similarity_tfidf,
    score_resume_against_jd,
    rank_resumes,
    route_resumes_to_open_jobs,
)

def test_tokenize():
    text = "Python, Java, C++, and Go! HTML5 & CSS3."
    tokens = tokenize(text)
    assert "python" in tokens
    assert "java" in tokens
    assert "c++" in tokens
    assert "go" in tokens
    assert "html" in tokens

def test_extract_years_of_experience():
    # Pattern 1
    assert extract_years_of_experience("I have 5 years of experience in Python.") == 5.0
    assert extract_years_of_experience("3.5+ years of software development") == 3.5
    # Ignore extremely high numbers (e.g. company age)
    assert extract_years_of_experience("Established 30 years ago") == 0.0

    # Date ranges
    assert extract_years_of_experience("Developer (2019 - 2023)") == 4.0
    assert extract_years_of_experience("Software Engineer (2020 to Present)") == 6.0  # 2026 - 2020

def test_extract_required_years():
    # Pattern 1
    assert extract_required_years("Requires 3 years of work experience.") == 3.0
    # Range
    assert extract_required_years("Seeking 5-7 years of relevant experience.") == 5.0
    # Fallback pattern
    assert extract_required_years("Minimum 4+ years preferred.") == 4.0
    # Filter high values
    assert extract_required_years("Founded 20 years ago. Requires 3+ years experience.") == 3.0

def test_extract_highest_degree_level():
    assert extract_highest_degree_level("Bachelor of Science in CS") == 3
    assert extract_highest_degree_level("Has a PhD in AI") == 5
    assert extract_highest_degree_level("No degree listed") == 0
    # Regression tests: Verify no false positive substring matches on short degree keys
    assert extract_highest_degree_level("I work on systems databases.") == 0  # 'ms' substring in 'systems' should not trigger Master's (4)
    assert extract_highest_degree_level("I will be working on coding.") == 0  # 'be' substring in 'be' should not trigger Bachelor's (3)
    assert extract_highest_degree_level("We need jobs.") == 0  # 'bs' substring in 'jobs' should not trigger Bachelor's (3)
    # Positive tests for uppercase short abbreviations
    assert extract_highest_degree_level("I have a BE in Computer Science") == 3
    assert extract_highest_degree_level("Got my BS degree") == 3
    assert extract_highest_degree_level("Graduated with MS") == 4

def test_normalize_skills_in_text():
    # Test skill synonym mapping
    text = "We use ReactJS, JS, and Node."
    skills = normalize_skills_in_text(text)
    # ReactJS maps to react, Node maps to node, JS maps to javascript
    assert "react" in skills
    assert "javascript" in skills
    assert "node" in skills

def test_cosine_similarity_tfidf():
    text_a = "Python machine learning data science"
    text_b = "Python data science machine learning"
    sim = cosine_similarity_tfidf(text_a, text_b)
    assert sim > 0.9

    text_c = "Completely unrelated words here"
    sim_unrelated = cosine_similarity_tfidf(text_a, text_c)
    assert sim_unrelated < 0.2

    # Empty inputs
    assert cosine_similarity_tfidf("", "") == 0.0

def test_score_resume_against_jd_empty():
    res = score_resume_against_jd("", "Python Developer")
    assert res["final_score"] == 0.0
    assert res["breakdown"]["skill_score"] == 0.0

    res = score_resume_against_jd("Experienced developer", "")
    assert res["final_score"] == 0.0

def test_score_resume_against_jd_normal():
    resume = "Experienced Software Engineer. 5 years in Python, React, and SQL. Completed Bachelor of Science."
    jd = "Looking for a Python Developer. Must have Bachelor degree. 3 years experience. Knowledge of SQL, React, AWS."
    
    res = score_resume_against_jd(resume, jd)
    assert res["final_score"] > 0.0
    assert res["required_years"] == 3.0
    assert res["candidate_years"] == 5.0
    assert res["candidate_degree_level"] == 3
    assert len(res["matched_skills"]) > 0

def test_score_resume_against_jd_mandatory_penalty():
    resume = "Experienced Software Engineer. Python and SQL."
    jd = "Seeking Python Developer. Skills: SQL, Docker."
    
    # Missing mandatory skill 'docker'
    res = score_resume_against_jd(resume, jd, mandatory_skills=["docker"])
    assert res["mandatory_penalty"] == 10.0
    assert "docker" in res["missing_mandatory_skills"]

def test_score_resume_against_jd_llm_nudge():
    resume = "Experienced Software Engineer. Python and SQL."
    jd = "Seeking Python Developer."
    
    # Mock LLM judge function returning a positive nudge
    def mock_llm_judge(r_txt, jd_txt, current_res):
        return {"score_adjustment": 5, "justification": "Good candidate profile"}
        
    res = score_resume_against_jd(resume, jd, llm_judge_fn=mock_llm_judge)
    assert res["llm_review"]["score_adjustment"] == 5
    assert res["llm_review"]["adjusted_score"] == res["final_score"]

def test_rank_resumes():
    resumes = [
        {"name": "Alice", "text": "Python and SQL expert, Bachelor of Science, 5 years"},
        {"name": "Bob", "text": "Junior developer, no experience, no skills"},
    ]
    jd = "Python Developer. Requires Bachelor degree and SQL."
    
    ranked = rank_resumes(resumes, jd)
    assert len(ranked) == 2
    assert ranked[0]["name"] == "Alice"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["name"] == "Bob"
    assert ranked[1]["rank"] == 2

def test_new_taxonomy_mappings():
    skills = normalize_skills_in_text("I do penetration testing, OWASP, product roadmaps, and swift.")
    assert "cybersecurity" in skills
    assert "product management" in skills
    assert "mobile development" in skills


def test_route_resumes_to_open_jobs():
    resumes = [
        {"name": "Alice", "text": "React Developer. 5 years in JS, HTML, CSS, frontend UI. Bachelor degree."},
        {"name": "Bob", "text": "Python Engineer. 6 years in Django, FastAPI, backend APIs, sql. Master degree."},
        {"name": "Charlie", "text": "Completely unqualified candidate text."},
    ]
    open_jobs = [
        {
            "slug": "react_developer",
            "title": "React Developer",
            "jd_text": "Need a Frontend React Developer. Required: JS, React, HTML, CSS. Bachelor degree. 3 years exp.",
            "mandatory_skills": ["react"]
        },
        {
            "slug": "python_developer",
            "title": "Python Developer",
            "jd_text": "Need a Backend Python Developer. Required: Python, Django, FastAPI, SQL. Master degree. 5 years exp.",
            "mandatory_skills": ["python"]
        }
    ]
    
    routed = route_resumes_to_open_jobs(resumes, open_jobs)
    assert len(routed["eligible"]) == 2
    assert len(routed["ineligible"]) == 1
    
    # Alice should be matched to React Developer
    alice_match = next(c for c in routed["eligible"] if c["name"] == "Alice")
    assert alice_match["best_fit_job_slug"] == "react_developer"
    assert alice_match["best_fit_job_title"] == "React Developer"
    assert alice_match["final_score"] >= 70.0
    
    # Bob should be matched to Python Developer
    bob_match = next(c for c in routed["eligible"] if c["name"] == "Bob")
    assert bob_match["best_fit_job_slug"] == "python_developer"
    assert bob_match["final_score"] >= 70.0
    
    # Charlie should be matched to closest but ineligible
    charlie_match = next(c for c in routed["ineligible"] if c["name"] == "Charlie")
    assert charlie_match["final_score"] < 70.0

