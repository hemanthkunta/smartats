import pytest
import json
from unittest.mock import patch, MagicMock
from engine.llm_judge import (
    is_ollama_available,
    get_ollama_models,
    judge_with_ollama,
    judge_with_gemini,
    judge_with_simulation,
    safe_int,
)

def test_is_ollama_available():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        assert is_ollama_available() is True

    with patch("requests.get", side_effect=Exception("Connection refused")):
        assert is_ollama_available() is False

def test_get_ollama_models():
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [{"name": "llama3.2:1b"}, {"name": "mistral"}]
        }
        models = get_ollama_models()
        assert "llama3.2:1b" in models
        assert "mistral" in models

    with patch("requests.get", side_effect=Exception("Error")):
        assert get_ollama_models() == []

def test_judge_with_ollama_success():
    mock_resp = {
        "response": json.dumps({
            "eligibility": "eligible",
            "score_adjustment": 5,
            "summary": "Great match.",
            "justification": "Has strong Python background."
        })
    }
    algo_result = {"final_score": 75.0}

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_resp
        
        result = judge_with_ollama("Resume content", "JD content", algo_result)
        assert result["eligibility"] == "eligible"
        assert result["score_adjustment"] == 5
        assert result["summary"] == "Great match."

def test_judge_with_ollama_json_fallback():
    # Ollama returns extra conversational wrapper, but JSON is inside
    mock_resp = {
        "response": "Here is the response: " + json.dumps({
            "eligibility": "potentially_eligible",
            "score_adjustment": -2,
            "summary": "No leadership.",
            "justification": "Needs more experience."
        })
    }
    algo_result = {"final_score": 50.0}

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_resp
        
        result = judge_with_ollama("Resume content", "JD content", algo_result)
        assert result["eligibility"] == "potentially_eligible"
        assert result["score_adjustment"] == -2

def test_judge_with_gemini_success():
    mock_resp = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "eligibility": "eligible",
                                "score_adjustment": 3,
                                "summary": "Good credentials.",
                                "justification": "Fits the team."
                            })
                        }
                    ]
                }
            }
        ]
    }
    algo_result = {"final_score": 80.0}

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_resp
        
        # Test with standard API key
        result = judge_with_gemini("Resume text", "JD text", algo_result, api_key="AIzaSyTestKey")
        assert result["eligibility"] == "eligible"
        assert result["score_adjustment"] == 3

        # Test with bearer OAuth token (starts with ya29.)
        result_oauth = judge_with_gemini("Resume text", "JD text", algo_result, api_key="ya29.OAuthToken")
        assert result_oauth["eligibility"] == "eligible"

def test_judge_with_gemini_error():
    algo_result = {"final_score": 80.0}
    
    # Missing API key error
    with pytest.raises(ValueError, match="Gemini API key is required"):
        judge_with_gemini("Resume text", "JD text", algo_result, api_key="")

    # Invalid payload format returned from Gemini
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"error": "Quota exceeded"}
        with pytest.raises(ValueError, match="Unexpected response format from Gemini"):
            judge_with_gemini("Resume text", "JD text", algo_result, api_key="key")

def test_judge_with_simulation():
    # Case 1: Strong candidate, quantified achievements, active ownership
    resume_good = """
    John Doe
    Architected and led a team of 5 engineers to build a data pipeline.
    Optimized PostgreSQL queries resulting in a 40% performance gain.
    Spearheaded the integration of Docker containers, saving $50k annually.
    """
    jd_senior = "Looking for a senior architect and lead developer."
    algo_result_good = {
        "final_score": 85.0,
        "matched_skills": ["python", "postgresql", "docker"],
        "candidate_degree_level": 4,
        "candidate_years": 8.0,
        "missing_mandatory_skills": []
    }
    
    res = judge_with_simulation(resume_good, jd_senior, algo_result_good)
    assert res["eligibility"] == "eligible"
    assert res["score_adjustment"] > 0
    assert "ownership" in res["justification"].lower()

    # Case 2: Missing mandatory skill -> Ineligible
    algo_result_missing = {
        "final_score": 50.0,
        "matched_skills": ["python"],
        "candidate_degree_level": 3,
        "candidate_years": 2.0,
        "missing_mandatory_skills": ["kubernetes"]
    }
    res_missing = judge_with_simulation("Resume content", jd_senior, algo_result_missing)
    assert res_missing["eligibility"] == "ineligible"

    # Case 3: Passive verbs, JD is senior but candidate is junior
    resume_junior = """
    Assisted the team with backend coding.
    Helped support existing React applications.
    Contributed to unit tests.
    """
    algo_result_junior = {
        "final_score": 45.0,
        "matched_skills": ["react"],
        "candidate_degree_level": 3,
        "candidate_years": 1.0,
        "missing_mandatory_skills": []
    }
    res_junior = judge_with_simulation(resume_junior, jd_senior, algo_result_junior)
    assert res_junior["eligibility"] == "ineligible" or res_junior["score_adjustment"] < 0

def test_safe_int():
    assert safe_int(5) == 5
    assert safe_int("5") == 5
    assert safe_int("+5") == 5
    assert safe_int("-3") == -3
    assert safe_int("  10  ") == 10
    assert safe_int("none", default=0) == 0
    assert safe_int(None, default=2) == 2
    assert safe_int(3.7) == 3
    assert safe_int("score is 5") == 5
