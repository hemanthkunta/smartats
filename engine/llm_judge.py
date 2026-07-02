"""
llm_judge.py — semantic judge layer using Ollama, Gemini, or an Offline Simulation model.

This module provides qualitative re-ranking of candidates (nudge score by ±10 points),
determines candidates' overall eligibility, and generates a professional summary.
"""
import json
import requests
import re
from typing import Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:1b"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"


def safe_int(val, default=0) -> int:
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        # Extract digits and minus sign
        cleaned = re.sub(r"[^\d-]", "", str(val))
        return int(cleaned) if cleaned else default
    except Exception:
        return default

JUDGE_PROMPT_TEMPLATE = """You are an expert HR evaluator. You will see a JOB DESCRIPTION and a RESUME (already pre-scored by an algorithmic system). 
Your job is to judge qualitative nuance the algorithm may have missed — e.g. leadership signal, depth of project ownership, quality of achievements (quantified impact vs vague claims).

Based on the candidate's skills, experience, education, and credentials against the JD requirements, categorize the candidate as:
- "eligible" (meets core requirements, strong skills, and ownership)
- "potentially_eligible" (has relevant skills but some experience or nuance gaps)
- "ineligible" (significant skill, experience, or mandatory requirement mismatches)

Respond with ONLY a JSON object, no other text, in this exact format:
{{
  "eligibility": "<eligible|potentially_eligible|ineligible>", 
  "score_adjustment": <integer between -10 and 10>, 
  "summary": "<a 2-3 sentence professional summary of the candidate's background, main strengths, and key gaps>",
  "justification": "<one or two sentences explaining eligibility and score change>"
}}

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}

ALGORITHMIC SCORE SO FAR: {algo_score}/100

JSON response:"""


def is_ollama_available(ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Checks if Ollama is running at the specified URL."""
    try:
        url = ollama_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def get_ollama_models(ollama_url: str = DEFAULT_OLLAMA_URL) -> list:
    """Fetches list of downloaded Ollama models."""
    try:
        url = ollama_url.rstrip("/") + "/api/tags"
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


def judge_with_ollama(
    resume_text: str,
    jd_text: str,
    algo_result: dict,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
) -> dict:
    """Calls a local Ollama model and returns {eligibility, score_adjustment, summary, justification}."""
    resume_snippet = resume_text[:3000]
    jd_snippet = jd_text[:2000]

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        jd_text=jd_snippet,
        resume_text=resume_snippet,
        algo_score=algo_result["final_score"],
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }

    url = ollama_url.rstrip("/") + "/api/generate"
    resp = requests.post(url, json=payload, timeout=180)
    resp.raise_for_status()
    raw_output = resp.json().get("response", "{}").strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        # best-effort JSON extraction
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw_output[start : end + 1])
        else:
            raise ValueError(f"Could not parse Ollama response: {raw_output}")

    elig = parsed.get("eligibility", "potentially_eligible").strip().lower().replace(" ", "_")
    if elig not in ["eligible", "potentially_eligible", "ineligible"]:
        elig = "potentially_eligible"

    return {
        "eligibility": elig,
        "score_adjustment": safe_int(parsed.get("score_adjustment", 0)),
        "summary": parsed.get("summary", ""),
        "justification": parsed.get("justification", ""),
    }


def judge_with_gemini(
    resume_text: str,
    jd_text: str,
    algo_result: dict,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
) -> dict:
    """Calls Google Gemini API via raw REST requests to judge resume quality."""
    if not api_key:
        raise ValueError("Gemini API key is required.")

    # Gemini has a larger context window
    resume_snippet = resume_text[:6000]
    jd_snippet = jd_text[:3000]

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        jd_text=jd_snippet,
        resume_text=resume_snippet,
        algo_score=algo_result["final_score"],
    )

    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}

    # Handle standard API keys (starting with AIzaSy or AQ) vs OAuth/Bearer access tokens (starting with ya29)
    if api_key.startswith("ya29."):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        url = f"{url}?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    try:
        raw_output = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError):
        raise ValueError(f"Unexpected response format from Gemini: {data}")

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw_output[start : end + 1])
        else:
            raise ValueError(f"Could not parse Gemini JSON: {raw_output}")

    elig = parsed.get("eligibility", "potentially_eligible").strip().lower().replace(" ", "_")
    if elig not in ["eligible", "potentially_eligible", "ineligible"]:
        elig = "potentially_eligible"

    return {
        "eligibility": elig,
        "score_adjustment": safe_int(parsed.get("score_adjustment", 0)),
        "summary": parsed.get("summary", ""),
        "justification": parsed.get("justification", ""),
    }


def judge_with_simulation(resume_text: str, jd_text: str, algo_result: dict) -> dict:
    """
    Simulates a high-quality qualitative LLM judge offline using pattern matching.
    Analyzes quantified achievements, active ownership verbs, passive verbs, and role seniority.
    """
    score_nudge = 0
    reasons = []
    res_lower = resume_text.lower()

    # 1. Look for quantified achievements
    quant_matches = re.findall(
        r"\b\d+(?:\.\d+)?%\b|\b\$\d+(?:[kKmMbB]|\s*million|\s*billion)?\b|\b\d+\s*(?:times|x|fold)\b",
        resume_text,
    )
    if len(quant_matches) >= 3:
        score_nudge += 3
        reasons.append(
            f"Demonstrates strong quantified business impact and metrics (found: {', '.join(quant_matches[:2])})."
        )
    elif len(quant_matches) > 0:
        score_nudge += 1
        reasons.append("Includes some quantified metrics in project results.")

    # 2. Look for strong leadership & technical ownership verbs
    ownership_verbs = [
        "led",
        "managed",
        "founded",
        "established",
        "architected",
        "headed",
        "spearheaded",
        "designed",
        "optimized",
        "built",
    ]
    found_ownership = [v for v in ownership_verbs if re.search(r"\b" + v + r"\b", res_lower)]
    if len(found_ownership) >= 3:
        score_nudge += 3
        reasons.append(
            f"High technical ownership and leadership indicators (actions like: {', '.join(found_ownership[:3])})."
        )
    elif len(found_ownership) > 0:
        score_nudge += 1
        reasons.append(
            f"Demonstrates component-level ownership (using verbs like: {', '.join(found_ownership[:2])})."
        )

    # 3. Look for passive/support verbs
    passive_verbs = ["assisted", "helped", "participated", "supported", "contributed"]
    found_passive = [v for v in passive_verbs if re.search(r"\b" + v + r"\b", res_lower)]
    if len(found_passive) >= 3 and len(found_ownership) < 2:
        score_nudge -= 2
        reasons.append(
            "Contains multiple passive descriptors ('assisted', 'participated'), suggesting support-level rather than lead-level contribution."
        )

    # 4. Seniority match
    is_jd_senior = any(
        x in jd_text.lower() for x in ["senior", "lead", "architect", "sr.", "principal", "manager"]
    )
    is_candidate_senior = any(
        x in res_lower
        for x in ["senior", "lead", "architect", "principal", "manager", "head", "director"]
    )

    if is_jd_senior and not is_candidate_senior:
        score_nudge -= 3
        reasons.append(
            "The role requires senior/lead level responsibility, but the candidate's profile is primarily individual contributor level."
        )
    elif not is_jd_senior and is_candidate_senior:
        score_nudge += 2
        reasons.append(
            "Candidate displays senior expertise for a mid-level role, offering extra technical capability."
        )

    # Bound the nudge to ±10 points
    nudge = max(-10, min(10, score_nudge))

    # Evaluate eligibility based on final score & missing mandatory gates & qualitative nudge
    score = algo_result["final_score"]
    missing_mandatory = algo_result.get("missing_mandatory_skills", [])
    
    if len(missing_mandatory) > 0:
        eligibility = "ineligible"
    elif score >= 70 and nudge >= -2:
        eligibility = "eligible"
    elif score >= 40 and nudge >= -5:
        eligibility = "potentially_eligible"
    else:
        eligibility = "ineligible"

    label = eligibility.replace("_", " ").title()
    justification = f"[{label}] " + (
        " ".join(reasons)
        if reasons
        else "Algorithmic score matches candidate profile; no significant qualitative deviations found."
    )

    # Dynamic summary generation
    skills_matched_str = ", ".join(algo_result["matched_skills"]) if algo_result["matched_skills"] else "general industry skills"
    deg_map = {5: "PhD", 4: "Master's", 3: "Bachelor's", 2: "Diploma", 1: "High School", 0: "No degree"}
    deg_str = deg_map.get(algo_result['candidate_degree_level'], "undergraduate level")
    
    summary = f"A {deg_str} holder with {algo_result['candidate_years']} years of relevant experience. They demonstrate technical alignment with core JD requirements, showing hands-on familiarity with tools such as {skills_matched_str}. "
    if nudge > 0:
        summary += "Their resume reflects strong ownership, active project leadership, and quantifiable business achievements, making them a very promising prospect."
    elif nudge < 0:
        summary += "While their technical matches are good, their background lacks sufficient active leadership metrics or senior-level responsibility as required by the role description."
    else:
        summary += "They present a solid technical profile with standard project involvement that fits the baseline expectations for this role."

    return {
        "eligibility": eligibility,
        "score_adjustment": nudge,
        "summary": summary,
        "justification": justification
    }
