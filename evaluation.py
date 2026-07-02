"""Hackathon evaluation helpers for candidate JSONL ingestion and submission export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, Optional


BASE_DIR = Path(__file__).resolve().parent


def _candidate_search_paths(filename: str) -> list[Path]:
    return [
        BASE_DIR / filename,
        BASE_DIR / "data" / filename,
        BASE_DIR / "data" / "hackathon" / filename,
        BASE_DIR / "data" / "job_openings" / filename,
    ]


def _resolve_existing_file(filename: str) -> Optional[Path]:
    for candidate_path in _candidate_search_paths(filename):
        if candidate_path.exists():
            return candidate_path
    return None


def load_candidate_schema(schema_path: str | Path | None = None) -> dict:
    resolved_path = Path(schema_path) if schema_path else _resolve_existing_file("candidate_schema.json")
    if not resolved_path or not resolved_path.exists():
        raise FileNotFoundError("candidate_schema.json was not found in the workspace")

    with resolved_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)

    if not isinstance(schema, dict):
        raise ValueError("candidate_schema.json must contain a JSON object")
    return schema


def stream_candidates_jsonl(jsonl_path: str | Path | None = None, schema_path: str | Path | None = None) -> Iterator[dict]:
    resolved_jsonl = Path(jsonl_path) if jsonl_path else _resolve_existing_file("candidates.jsonl")
    if not resolved_jsonl or not resolved_jsonl.exists():
        raise FileNotFoundError("candidates.jsonl was not found in the workspace")

    schema = load_candidate_schema(schema_path)
    required_fields = set(schema.get("required", []))
    allowed_fields = set(schema.get("properties", {}).keys()) if isinstance(schema.get("properties"), dict) else set()
    additional_allowed = schema.get("additionalProperties", True)

    with resolved_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in candidates.jsonl must be a JSON object")

            missing_fields = sorted(field for field in required_fields if field not in row)
            if missing_fields:
                raise ValueError(f"Line {line_number} is missing required fields: {', '.join(missing_fields)}")

            if allowed_fields and not additional_allowed:
                unexpected_fields = sorted(field for field in row if field not in allowed_fields)
                if unexpected_fields:
                    raise ValueError(
                        f"Line {line_number} contains unexpected fields: {', '.join(unexpected_fields)}"
                    )

            yield row


def load_hackathon_candidate_pool(jsonl_path: str | Path | None = None, schema_path: str | Path | None = None) -> list[dict]:
    candidates = []
    for index, row in enumerate(stream_candidates_jsonl(jsonl_path=jsonl_path, schema_path=schema_path), start=1):
        filename = row.get("filename") or row.get("file_name") or row.get("candidate_id") or f"candidate_{index}"
        text = row.get("text") or row.get("resume_text") or row.get("content") or ""
        name = row.get("name") or row.get("candidate_name") or row.get("full_name") or filename
        contact = row.get("contact") if isinstance(row.get("contact"), dict) else {}
        if not contact:
            contact = {
                "email": row.get("email", "Not Provided"),
                "phone": row.get("phone", "Not Provided"),
            }

        candidates.append(
            {
                "name": name,
                "filename": filename,
                "text": text,
                "contact": contact,
                "candidate_id": row.get("candidate_id", filename),
                "source": "candidates.jsonl",
                "raw_record": row,
            }
        )
    return candidates


def load_sample_submission_columns(sample_submission_path: str | Path | None = None) -> list[str]:
    resolved_path = Path(sample_submission_path) if sample_submission_path else _resolve_existing_file("sample_submission.csv")
    if not resolved_path or not resolved_path.exists():
        return ["candidate_id", "rank", "final_score", "best_fit_job_title"]

    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def build_submission_rows(candidates: Iterable[dict], column_keys: Optional[list[str]] = None) -> list[dict]:
    keys = column_keys or load_sample_submission_columns()
    rows = []
    for candidate in candidates:
        record = candidate.get("raw_record", {}) if isinstance(candidate.get("raw_record"), dict) else {}
        row = {}
        for key in keys:
            value = (
                candidate.get(key)
                or record.get(key)
                or candidate.get(key.replace(" ", "_"))
                or record.get(key.replace(" ", "_"))
            )
            if value is None:
                if key.lower() == "rank":
                    value = candidate.get("rank", "")
                elif key.lower() in {"score", "final_score", "match_score"}:
                    value = candidate.get("final_score", "")
                else:
                    value = ""
            row[key] = value
        rows.append(row)
    return rows