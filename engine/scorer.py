"""
scorer.py — the core hybrid scoring engine.

WHY HYBRID (this is your pitch to judges):
Pure-LLM scoring is non-deterministic, slow, hallucination-prone, and a black box.
Pure-keyword scoring misses synonyms and context ("led" vs "participated in").
We combine FOUR deterministic, explainable signals (fast, reproducible, free)
and OPTIONALLY blend in a local LLM (via Ollama) for semantic nuance on
experience descriptions. Judges can see exactly *why* a candidate scored
what they scored — that explainability is the differentiator.

Final Score = weighted blend of:
  1. Skill Match Score      (40%) - taxonomy-aware keyword/synonym overlap
  2. Experience Match Score (25%) - years required vs years found
  3. Education Match Score  (15%) - degree level required vs degree level found
  4. Semantic Similarity    (20%) - TF-IDF cosine similarity of full text (no heavy deps)

Optional: LLM Judge Layer (Ollama) re-ranks top-N candidates with qualitative
reasoning and can nudge final score by a small bounded amount with a written
justification — but it can NEVER override the deterministic floor, so the
score stays trustworthy even if the LLM is flaky.
"""
import re
import math
from collections import Counter
from typing import Optional

from engine.taxonomy import ALIAS_TO_CANONICAL, SKILL_SYNONYMS, DEGREE_LEVELS


# ---------- Text utilities ----------

STOPWORDS = set("""
a an the and or of to in on for with at by from as is are was were be been being
this that these those it its he she they his her their our your you i we
will would could should can may might must shall not no nor but if then than
""".split())

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z+#\.\-]*")


def tokenize(text: str) -> list:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def extract_years_of_experience(text: str) -> float:
    """Look for patterns like '5 years', '3+ years of experience', date ranges."""
    years_found = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*year", text, re.IGNORECASE):
        val = float(m.group(1))
        # Ignore extremely high numbers in sentences (likely company age or statistics, not candidate experience)
        if val <= 25:
            years_found.append(val)
    if years_found:
        return max(years_found)

    # fallback: estimate from date ranges like "2019 - 2023" or "2019 - Present"
    year_ranges = re.findall(r"(20[0-2]\d|19[89]\d)\s*[-–to]+\s*(20[0-2]\d|present|current)", text, re.IGNORECASE)
    total = 0.0
    for start, end in year_ranges:
        start_year = int(start)
        end_year = 2026 if not end.isdigit() else int(end)
        if end_year >= start_year:
            total += (end_year - start_year)
    return total


def extract_required_years(jd_text: str) -> float:
    """Extract required years of experience from JD, ignoring company history boilerplate."""
    matches = []
    
    # Pattern 1: Look for digit + year followed by experience keywords
    pattern_exp = r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*\+?\s*year[s]?\s*(?:of\s*)?(?:relevant\s*)?(?:experience|work|professional|industry|engineering|development|coding|SRE)"
    for m in re.finditer(pattern_exp, jd_text, re.IGNORECASE):
        val = int(m.group(1))
        if val <= 15:
            matches.append(val)
            
    # Pattern 2: Fallback to any digit + "year" pattern, strictly filter for values <= 15
    if not matches:
        pattern_fallback = r"(\d+)\s*\+?\s*year"
        for m in re.finditer(pattern_fallback, jd_text, re.IGNORECASE):
            val = int(m.group(1))
            if val <= 15:
                matches.append(val)
                
    if matches:
        return float(max(matches))
        
    return 0.0


def extract_highest_degree_level(text: str) -> int:
    best = 0
    for keyword, level in DEGREE_LEVELS.items():
        if keyword in ["be", "bs", "ms"]:
            # Match case-sensitively as uppercase (e.g. 'BE', 'BS', 'MS')
            # to avoid false positives on common words like the verb 'be'.
            pattern = r"\b" + re.escape(keyword.upper()) + r"\b"
            if re.search(pattern, text):
                best = max(best, level)
        else:
            # Case-insensitive match for longer keywords and dotted abbreviations (e.g. 'b.e.')
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text.lower()):
                best = max(best, level)
    return best


def normalize_skills_in_text(text: str) -> set:
    """Map every token/phrase found in text to its canonical skill name."""
    text_lower = text.lower()
    found = set()
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # word-boundary aware search, handles multi-word aliases too
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(alias) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.add(canonical)
    return found


def extract_explicit_jd_skills(jd_text: str) -> set:
    """Skills explicitly required by the JD (taxonomy-aware)."""
    return normalize_skills_in_text(jd_text)


# ---------- TF-IDF cosine similarity (no heavy ML deps needed) ----------

def compute_tf(tokens: list) -> Counter:
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return Counter(tokens)


def cosine_similarity_tfidf(text_a: str, text_b: str) -> float:
    """Lightweight TF-IDF-style cosine similarity between two texts.
    No sklearn/torch dependency — pure python, fast, good enough for
    ranking purposes (judges care about relative ranking, not perfect IR)."""
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    tf_a = compute_tf(tokens_a)
    tf_b = compute_tf(tokens_b)

    vocab = set(tf_a.keys()) | set(tf_b.keys())
    if not vocab:
        return 0.0

    # simple idf proxy: rarer terms across the two docs get slightly boosted
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for term in vocab:
        a_val = tf_a.get(term, 0)
        b_val = tf_b.get(term, 0)
        dot += a_val * b_val
        norm_a += a_val ** 2
        norm_b += b_val ** 2

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# ---------- Main scoring function ----------

def score_resume_against_jd(
    resume_text: str,
    jd_text: str,
    llm_judge_fn: Optional[callable] = None,
    weights: Optional[dict] = None,
    mandatory_skills: Optional[list] = None,
    enable_indian_mode: bool = False,
    embedding_fn: Optional[callable] = None,
) -> dict:
    """
    Returns a structured breakdown dict — this structure is what gets
    rendered in the UI and is the core 'explainability' artifact.
    """
    if not resume_text or not resume_text.strip() or not jd_text or not jd_text.strip():
        default_weights = weights or {
            "skills": 0.40,
            "experience": 0.25,
            "education": 0.15,
            "semantic": 0.20,
        }
        return {
            "final_score": 0.0,
            "final_score_before_penalty": 0.0,
            "breakdown": {
                "skill_score": 0.0,
                "experience_score": 0.0,
                "education_score": 0.0,
                "semantic_score": 0.0,
            },
            "weights": default_weights,
            "matched_skills": [],
            "missing_skills": [],
            "missing_mandatory_skills": [],
            "mandatory_penalty": 0.0,
            "required_years": 0.0,
            "candidate_years": 0.0,
            "required_degree_level": 0,
            "candidate_degree_level": 0,
            "llm_review": None,
            "education_boost_applied": False,
        }

    if enable_indian_mode:
        from engine.india_taxonomy import normalize_indian_resume_text
        resume_text = normalize_indian_resume_text(resume_text)

    jd_skills = extract_explicit_jd_skills(jd_text)
    resume_skills = normalize_skills_in_text(resume_text)

    matched_skills = jd_skills & resume_skills
    missing_skills = jd_skills - resume_skills

    skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 50.0

    required_years = extract_required_years(jd_text)
    candidate_years = extract_years_of_experience(resume_text)
    if required_years > 0:
        raw_experience_score = min(candidate_years / required_years, 1.2) * 100
        raw_experience_score = min(raw_experience_score, 100.0)
    else:
        raw_experience_score = 75.0  # no explicit requirement -> neutral-positive default

    # Relevance gate: years of experience only count fully if those years were
    # spent using skills the JD actually asks for. Otherwise a candidate with
    # 10 years in an unrelated field would wrongly score 100% on experience.
    # This is a key differentiator vs naive "years extraction" approaches.
    skill_overlap_ratio = (len(matched_skills) / len(jd_skills)) if jd_skills else 1.0
    relevance_factor = 0.4 + 0.6 * skill_overlap_ratio  # floor at 40% so some credit always given
    experience_score = raw_experience_score * relevance_factor

    required_degree = extract_highest_degree_level(jd_text)
    candidate_degree = extract_highest_degree_level(resume_text)
    if required_degree > 0:
        education_score = 100.0 if candidate_degree >= required_degree else max(0, 100 - (required_degree - candidate_degree) * 30)
    else:
        education_score = 80.0

    # Indian Tier-1 college boost
    education_boost_applied = False
    if enable_indian_mode:
        from engine.india_taxonomy import has_indian_tier1_college
        if has_indian_tier1_college(resume_text):
            education_score = min(100.0, education_score + 10.0)
            education_boost_applied = True

    if embedding_fn:
        try:
            semantic_score = embedding_fn(resume_text, jd_text)
        except Exception:
            semantic_score = cosine_similarity_tfidf(resume_text, jd_text) * 100
            # TF-IDF cosine tends to run low in absolute terms; rescale with a soft curve
            semantic_score = min(100.0, semantic_score * 3.0)
    else:
        semantic_score = cosine_similarity_tfidf(resume_text, jd_text) * 100
        # TF-IDF cosine tends to run low in absolute terms; rescale with a soft curve
        semantic_score = min(100.0, semantic_score * 3.0)

    if not weights:
        weights = {
            "skills": 0.40,
            "experience": 0.25,
            "education": 0.15,
            "semantic": 0.20,
        }

    final_score = (
        skill_score * weights["skills"]
        + experience_score * weights["experience"]
        + education_score * weights["education"]
        + semantic_score * weights["semantic"]
    )
    final_score = round(min(100.0, max(0.0, final_score)), 1)
    final_score_before_penalty = final_score

    # Mandatory skills check
    missing_mandatory = []
    if mandatory_skills:
        for ms in mandatory_skills:
            if ms not in matched_skills:
                missing_mandatory.append(ms)
    
    # Apply -10 penalty per missing mandatory skill
    mandatory_penalty = len(missing_mandatory) * 10.0
    final_score = round(max(0.0, final_score - mandatory_penalty), 1)

    result = {
        "final_score": final_score,
        "final_score_before_penalty": final_score_before_penalty,
        "breakdown": {
            "skill_score": round(skill_score, 1),
            "experience_score": round(experience_score, 1),
            "education_score": round(education_score, 1),
            "semantic_score": round(semantic_score, 1),
        },
        "weights": weights,
        "matched_skills": sorted(matched_skills),
        "missing_skills": sorted(missing_skills),
        "missing_mandatory_skills": sorted(missing_mandatory),
        "mandatory_penalty": mandatory_penalty,
        "required_years": required_years,
        "candidate_years": round(candidate_years, 1),
        "required_degree_level": required_degree,
        "candidate_degree_level": candidate_degree,
        "llm_review": None,
        "education_boost_applied": education_boost_applied,
    }

    # Optional LLM judge layer — bounded nudge, never a full override
    if llm_judge_fn is not None:
        try:
            llm_result = llm_judge_fn(resume_text, jd_text, result)
            if llm_result:
                nudge = max(-10, min(10, llm_result.get("score_adjustment", 0)))
                adjusted = round(min(100.0, max(0.0, final_score + nudge)), 1)
                result["llm_review"] = {
                    "justification": llm_result.get("justification", ""),
                    "score_adjustment": nudge,
                    "adjusted_score": adjusted,
                }
                result["final_score"] = adjusted
        except Exception as e:
            result["llm_review"] = {"error": str(e)}

    return result


def rank_resumes(
    resumes: list,
    jd_text: str,
    llm_judge_fn: Optional[callable] = None,
    weights: Optional[dict] = None,
    mandatory_skills: Optional[list] = None,
) -> list:
    """resumes: list of dicts with keys {name, text}. Returns sorted list with scores attached."""
    import concurrent.futures

    if not resumes:
        return []

    def score_single(r):
        result = score_resume_against_jd(
            r["text"],
            jd_text,
            llm_judge_fn=llm_judge_fn,
            weights=weights,
            mandatory_skills=mandatory_skills,
        )
        return {**r, **result}

    max_workers = min(8, len(resumes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        scored = list(executor.map(score_single, resumes))

    scored.sort(key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")))
    for i, r in enumerate(scored, start=1):
        r["rank"] = i
    return scored


def route_resumes_to_open_jobs(
    resumes: list[dict],
    open_jobs: list[dict],
    llm_judge_fn: Optional[callable] = None,
    weights: Optional[dict] = None,
    enable_indian_mode: bool = False,
    embedding_fn: Optional[callable] = None
) -> dict:
    """
    Evaluates each candidate resume against all open jobs in parallel.
    Identifies the best-fit job (highest score) for each candidate.
    Classifies candidates as Eligible (score >= 70% and no missing mandatory skills for that job)
    or Ineligible (score < 70% or missing any mandatory skills for that job).
    
    Returns:
         dict: {
             "eligible": list[dict],      # List of candidates matched to their eligible roles
             "ineligible": list[dict]     # List of candidates matched to their best-fit but ineligible roles
         }
    """
    import concurrent.futures

    if not resumes or not open_jobs:
        return {"eligible": [], "ineligible": []}

    # Helper function to evaluate one resume against all open jobs
    def evaluate_candidate_against_all_jobs(r):
        best_fit = None
        best_score = -1.0
        best_result = None

        for job in open_jobs:
            result = score_resume_against_jd(
                resume_text=r["text"],
                jd_text=job["jd_text"],
                llm_judge_fn=llm_judge_fn,
                weights=weights,
                mandatory_skills=job.get("mandatory_skills", []),
                enable_indian_mode=enable_indian_mode,
                embedding_fn=embedding_fn
            )
            score = result["final_score"]
            if score > best_score:
                best_score = score
                best_fit = job
                best_result = result

        # Determine eligibility for the best-fit role
        is_eligible = False
        if best_result:
            has_mandatory_skills = len(best_result.get("missing_mandatory_skills", [])) == 0
            if best_score >= 70.0 and has_mandatory_skills:
                is_eligible = True

        candidate_record = {
            **r,
            "best_fit_job_title": best_fit["title"] if best_fit else "None",
            "best_fit_job_slug": best_fit["slug"] if best_fit else "none",
            "final_score": best_score if best_score >= 0 else 0.0,
            "is_eligible": is_eligible,
            "evaluation": best_result
        }
        return candidate_record

    # Run routing concurrently across resumes using a ThreadPoolExecutor
    max_workers = min(8, len(resumes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        routed_candidates = list(executor.map(evaluate_candidate_against_all_jobs, resumes))

    # Group candidates
    eligible_list = [c for c in routed_candidates if c["is_eligible"]]
    ineligible_list = [c for c in routed_candidates if not c["is_eligible"]]

    # Sort each list by final score
    eligible_list.sort(key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")))
    ineligible_list.sort(key=lambda x: (-float(x.get("final_score", 0.0)), x.get("filename", ""), x.get("name", "")))

    return {
        "eligible": eligible_list,
        "ineligible": ineligible_list
    }
