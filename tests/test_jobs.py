import os
import shutil
import pytest
from engine.jobs import save_job_opening, load_all_job_openings, delete_job_opening, JOBS_DIR


@pytest.fixture(autouse=True)
def clean_jobs_dir():
    # Back up existing files if any
    backup_dir = JOBS_DIR + "_backup"
    if os.path.exists(JOBS_DIR):
        shutil.move(JOBS_DIR, backup_dir)
    os.makedirs(JOBS_DIR, exist_ok=True)
    
    yield
    
    # Restore backup
    if os.path.exists(JOBS_DIR):
        shutil.rmtree(JOBS_DIR)
    if os.path.exists(backup_dir):
        shutil.move(backup_dir, JOBS_DIR)


def test_save_and_load_job_opening():
    slug = save_job_opening(
        title="Test Job Opening",
        jd_text="Need a Python developer with 3 years experience.",
        mandatory_skills=["python"],
        required_years=3.0,
        required_degree=3
    )
    assert slug == "test_job_opening"
    
    jobs = load_all_job_openings()
    # Check that at least our test job is present
    assert any(j["title"] == "Test Job Opening" for j in jobs)
    test_job = next(j for j in jobs if j["title"] == "Test Job Opening")
    assert test_job["required_years"] == 3.0
    assert test_job["mandatory_skills"] == ["python"]


def test_delete_job_opening():
    slug = save_job_opening(
        title="Delete Me",
        jd_text="Text content",
        mandatory_skills=[],
        required_years=0.0,
        required_degree=0
    )
    
    jobs = load_all_job_openings()
    assert any(j["slug"] == slug for j in jobs)
    
    deleted = delete_job_opening(slug)
    assert deleted is True
    
    jobs_after = load_all_job_openings()
    assert not any(j["slug"] == slug for j in jobs_after)


def test_delete_nonexistent_job():
    assert delete_job_opening("fake_slug") is False
