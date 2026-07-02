import json

from engine.evaluation import (
    build_submission_matrix,
    detect_trap_anomalies,
    load_validated_candidates,
    stream_jsonl_candidates,
    validate_candidate_record,
)


def test_stream_jsonl_candidates_reads_row_by_row(tmp_path, monkeypatch):
    data_path = tmp_path / "candidates.jsonl"
    data_path.write_text(
        json.dumps({"name": "Alice", "resume_text": "Python"}) + "\n"
        + "not-json\n"
        + json.dumps({"name": "Bob", "resume_text": "SQL"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    rows = list(stream_jsonl_candidates())
    assert [row["name"] for row in rows] == ["Alice", "Bob"]


def test_validate_candidate_record_and_trap_detection():
    schema = {
        "type": "object",
        "required": ["name", "skills"],
        "properties": {
            "name": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
    }
    record = {"name": "Alice", "skills": ["Python", "Python", "Python"], "start_year": 2024, "end_year": 2020}
    assert validate_candidate_record(record, schema) == []
    anomalies = detect_trap_anomalies(record)
    assert any("Chronology" in message for message in anomalies)
    assert any("skill" in message.lower() for message in anomalies)


def test_load_validated_candidates_honors_schema(tmp_path, monkeypatch):
    (tmp_path / "candidate_schema.json").write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["name", "resume_text"],
                "properties": {
                    "name": {"type": "string"},
                    "resume_text": {"type": "string"},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "candidates.jsonl").write_text(
        json.dumps({"name": "Alice", "resume_text": "Python developer"}) + "\n"
        + json.dumps({"name": 42, "resume_text": "bad"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    records = load_validated_candidates()
    assert len(records) == 1
    assert records[0]["name"] == "Alice"
    assert records[0]["text"] == "Python developer"


def test_build_submission_matrix_aligns_to_sample_submission(tmp_path):
    sample_path = tmp_path / "sample_submission.csv"
    sample_path.write_text(
        "Rank,Name,Final Score (%),Matched Role,Skill Match,Experience Match,Education Match,Semantic Similarity,Years Experience,Status,Email\n",
        encoding="utf-8",
    )
    candidates = [
        {
            "rank": 1,
            "name": "Alice",
            "final_score": 91.2,
            "best_fit_job_title": "Data Scientist",
            "breakdown": {
                "skill_score": 95,
                "experience_score": 88,
                "education_score": 90,
                "semantic_score": 85,
            },
            "candidate_years": 5,
            "status": "Eligible",
            "contact": {"email": "alice@example.com"},
        }
    ]

    df = build_submission_matrix(candidates, spec_path=str(sample_path))
    assert list(df.columns) == [
        "Rank",
        "Name",
        "Final Score (%)",
        "Matched Role",
        "Skill Match",
        "Experience Match",
        "Education Match",
        "Semantic Similarity",
        "Years Experience",
        "Status",
        "Email",
    ]
    assert df.iloc[0]["Final Score (%)"] == "91.2%"
    assert df.iloc[0]["Email"] == "alice@example.com"
