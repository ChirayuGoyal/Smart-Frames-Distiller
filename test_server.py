"""
test_server.py — Verification tests for the FastAPI server layer (Phase 4).
"""
from __future__ import annotations

import time
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from faces.store import InMemoryFaceStore
from server.app import create_app
from server.settings import Settings


@pytest.fixture
def app_settings(tmp_path: Path) -> Settings:
    """Provide isolated Settings for testing."""
    work_dir = tmp_path / "sfd_test_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Create synthetic video path for allowed_input_dirs
    repo_root = Path(__file__).resolve().parent
    test_data = repo_root / "test_data"
    
    settings = Settings(
        work_dir=work_dir,
        allowed_input_dirs=[test_data, work_dir],
        warm_models=False,
    )
    settings.kafka.enabled = False
    return settings


@pytest.fixture
def client(app_settings: Settings) -> TestClient:
    app = create_app(app_settings)
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client: TestClient) -> None:
    # Basic liveness
    r1 = client.get("/v1/health")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok"}

    # Readiness probe
    r2 = client.get("/v1/health/ready")
    assert r2.status_code == 200
    data = r2.json()
    assert "status" in data
    assert "dependencies" in data
    assert "models" in data["dependencies"]
    assert "gpu" in data["dependencies"]


def test_job_submission_and_polling(client: TestClient, app_settings: Settings) -> None:
    repo_root = Path(__file__).resolve().parent
    video_path = repo_root / "test_data" / "synthetic_action_aware.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    # 1. Submit job via JSON server_path
    payload = {
        "server_path": str(video_path),
        "options": {
            "filter_enabled": True,
            "run_id": "test_job_run_1",
            "device": "cpu",
        },
    }
    resp = client.post("/v1/jobs", json=payload)
    assert resp.status_code == 201, resp.text
    job_record = resp.json()
    job_id = job_record["job_id"]
    assert job_record["state"] in ("queued", "running", "succeeded")
    assert job_record["type"] == "pipeline"

    # 2. Poll until finished (or timeout after 15s)
    t0 = time.time()
    while time.time() - t0 < 15:
        r_poll = client.get(f"/v1/jobs/{job_id}")
        assert r_poll.status_code == 200
        state = r_poll.json()["state"]
        if state in ("succeeded", "failed", "cancelled"):
            break
        time.sleep(0.5)

    final_job = client.get(f"/v1/jobs/{job_id}").json()
    assert final_job["state"] == "succeeded", f"Job failed: {final_job.get('error')}"
    assert final_job["report"] is not None

    # 3. List jobs
    r_list = client.get("/v1/jobs")
    assert r_list.status_code == 200
    assert any(j["job_id"] == job_id for j in r_list.json())

    # 4. Get report explicitly
    r_rep = client.get(f"/v1/jobs/{job_id}/report")
    assert r_rep.status_code == 200
    assert "filter" in r_rep.json()

    # 5. List artifacts
    r_arts = client.get(f"/v1/jobs/{job_id}/artifacts")
    assert r_arts.status_code == 200
    artifacts = r_arts.json()
    assert len(artifacts) > 0

    # 6. Path traversal security check
    r_trav = client.get(f"/v1/jobs/{job_id}/artifacts/..%2F..%2Fjob.json")
    assert r_trav.status_code in (403, 404)


def test_job_validation_and_cancellation(client: TestClient) -> None:
    # 1. Validation error when no stages enabled
    payload = {
        "server_path": __file__,
        "options": {
            "filter_enabled": False,
            "detection_enabled": False,
            "chunk_enabled": False,
        },
    }
    r = client.post("/v1/jobs", json=payload)
    assert r.status_code == 422  # Pydantic validator rejects no stage selected

    # 2. Submit a dummy job and cancel it while queued/starting
    repo_root = Path(__file__).resolve().parent
    video_path = repo_root / "test_data" / "synthetic_action_aware.mp4"
    if video_path.exists():
        r_sub = client.post("/v1/jobs", json={
            "server_path": str(video_path),
            "options": {"filter_enabled": True, "device": "cpu"}
        })
        if r_sub.status_code == 201:
            jid = r_sub.json()["job_id"]
            r_canc = client.post(f"/v1/jobs/{jid}/cancel")
            assert r_canc.status_code in (200, 409)  # 409 if already finished super fast


def test_streams_endpoints(client: TestClient) -> None:
    repo_root = Path(__file__).resolve().parent
    video_path = repo_root / "test_data" / "synthetic_action_aware.mp4"
    if not video_path.exists():
        pytest.skip(f"Test video not found: {video_path}")

    # Start stream (cv2.VideoCapture opens local file paths seamlessly)
    payload = {
        "rtsp_url": str(video_path),
        "site_id": "test_site",
        "camera_id": "cam_1",
        "options": {"enabled": False},  # disable kafka chunks
    }
    r_start = client.post("/v1/streams", json=payload)
    assert r_start.status_code == 201, r_start.text
    stream_id = r_start.json()["stream_id"]
    assert r_start.json()["state"] in ("connecting", "running", "stopped")

    # List streams
    r_list = client.get("/v1/streams")
    assert r_list.status_code == 200
    assert any(s["stream_id"] == stream_id for s in r_list.json())

    # Get stream status
    r_get = client.get(f"/v1/streams/{stream_id}")
    assert r_get.status_code == 200
    assert "stats" in r_get.json()

    # Stop stream
    r_stop = client.delete(f"/v1/streams/{stream_id}")
    assert r_stop.status_code == 200
    assert r_stop.json()["state"] in ("stopped", "failed")


def test_faces_endpoints_with_in_memory_store(client: TestClient, monkeypatch) -> None:
    import numpy as np
    # Inject InMemoryFaceStore into model_cache and seed an identity
    mem_store = InMemoryFaceStore()
    mem_store.upsert_row(
        uid="test-uuid-100",
        person_id="test-uuid-100",
        embedding=np.zeros(512, dtype=np.float32),
        name="Alice Old",
        role="Junior Eng",
        department="R&D",
        notes="",
        site_id="hq_1",
        camera_id="cam_1",
    )
    monkeypatch.setattr("server.model_cache.get_face_store", lambda settings, **kw: mem_store)
    monkeypatch.setattr("server.routers.faces.get_face_store", lambda settings, **kw: mem_store)

    # 1. Tag identity
    tag_payload = {
        "uid": "test-uuid-100",
        "name": "Alice Smith",
        "role": "Engineer",
        "department": "R&D",
        "site_id": "hq_1",
    }
    r_tag = client.post("/v1/faces/identities", json=tag_payload)
    assert r_tag.status_code == 200
    assert r_tag.json()["name"] == "Alice Smith"

    # 2. List identities
    r_list = client.get("/v1/faces/identities?site_id=hq_1")
    assert r_list.status_code == 200
    identities = r_list.json()
    assert len(identities) == 1
    assert identities[0]["name"] == "Alice Smith"

    # 3. Patch identity
    r_patch = client.patch("/v1/faces/identities/test-uuid-100", json={
        "name": "Alice Jones",
        "role": "Lead Engineer",
        "department": "R&D",
        "site_id": "hq_1",
    })
    assert r_patch.status_code == 200
    assert r_patch.json()["name"] == "Alice Jones"

    # 4. Delete identity
    r_del = client.delete("/v1/faces/identities/test-uuid-100")
    assert r_del.status_code == 200
    assert r_del.json() == {"deleted": True}
