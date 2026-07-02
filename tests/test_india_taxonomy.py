import pytest
from engine.india_taxonomy import normalize_indian_resume_text, redact_indian_pii, INDIA_SKILL_ADDITIONS
from engine.taxonomy import SKILL_SYNONYMS, ALIAS_TO_CANONICAL

def test_normalize_indian_resume_text():
    assert normalize_indian_resume_text("I am a Fresher with B Tech degree") == "i am a entry level with b.tech degree"
    assert normalize_indian_resume_text("Specialist in Dot Net development") == "specialist in .net development"
    assert normalize_indian_resume_text("2026 passout graduate") == "2026 graduate graduate"

def test_redact_indian_pii():
    text = "Contact: +91 9876543210 or Aadhaar 1234-5678-9012"
    redacted = redact_indian_pii(text)
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_AADHAAR]" in redacted
    
    text2 = "Call +919876543210 or check Aadhaar 123456789012"
    redacted2 = redact_indian_pii(text2)
    assert "[REDACTED_PHONE]" in redacted2
    assert "[REDACTED_AADHAAR]" in redacted2

def test_india_taxonomy_merging():
    # Verify that Indian skill synonyms have been successfully merged into taxonomy
    assert "b.tech" in SKILL_SYNONYMS
    assert "m.tech" in SKILL_SYNONYMS
    assert "hibernate" in SKILL_SYNONYMS
    assert "flutter" in SKILL_SYNONYMS
    
    # Check alias mapping works correctly
    assert ALIAS_TO_CANONICAL["btech"] == "b.tech"
    assert ALIAS_TO_CANONICAL["fresher"] == "entry level"
    assert ALIAS_TO_CANONICAL["hibernate"] == "hibernate"
    assert ALIAS_TO_CANONICAL["flutter"] == "flutter"

from engine.india_taxonomy import has_indian_tier1_college
from engine.scorer import score_resume_against_jd

def test_has_indian_tier1_college():
    assert has_indian_tier1_college("I graduated from IIT Madras") is True
    assert has_indian_tier1_college("Graduated from BITS Pilani") is True
    assert has_indian_tier1_college("Studied at DTU Delhi") is True
    assert has_indian_tier1_college("Some other college") is False

def test_score_resume_against_jd_indian_mode():
    jd = "Need a Python Developer with a Bachelor's Degree."
    # Resume with "b tech" and "IIT" - "b tech" will normalize to "b.tech" which satisfies the bachelor requirement,
    # and "IIT" will trigger the +10 boost to education score.
    resume = "Python Developer. I have a b tech degree from IIT Bombay."
    
    # Without Indian mode
    result_normal = score_resume_against_jd(resume, jd, enable_indian_mode=False)
    # With Indian mode
    result_indian = score_resume_against_jd(resume, jd, enable_indian_mode=True)
    
    assert result_indian["education_boost_applied"] is True
    assert result_indian["breakdown"]["education_score"] > result_normal["breakdown"]["education_score"]

