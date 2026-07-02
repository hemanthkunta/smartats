import json

from evaluation import build_submission_rows, load_hackathon_candidate_pool


def test_load_hackathon_candidate_pool_streams_jsonl(tmp_path):
    schema_path = tmp_path / "candidate_schema.json"
    jsonl_path = tmp_path / "candidates.jsonl"

    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "required": ["candidate_id", "text"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "text": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "additionalProperties": True,
            }
        ),
        encoding="utf-8",
    )
    jsonl_path.write_text(
        '{"candidate_id": "c1", "name": "Alice", "text": "Python dev", "email": "a@example.com"}\n',
        encoding="utf-8",
    )

    candidates = load_hackathon_candidate_pool(jsonl_path=jsonl_path, schema_path=schema_path)
    assert len(candidates) == 1
    assert candidates[0]["name"] == "Alice"
    assert candidates[0]["contact"]["email"] == "a@example.com"


def test_build_submission_rows_matches_columns():
    columns = ["candidate_id", "rank", "final_score", "best_fit_job_title"]
    rows = build_submission_rows(
        [
            {"candidate_id": "c1", "rank": 1, "final_score": 91.2, "best_fit_job_title": "Python Developer"},
        ],
        column_keys=columns,
    )

    assert list(rows[0].keys()) == columns
    assert rows[0]["candidate_id"] == "c1"
    assert rows[0]["rank"] == 1