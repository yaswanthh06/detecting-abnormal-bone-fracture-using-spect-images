import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_model_status_endpoint():
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "checkpoint_loaded" in data
    assert "classes" in data
    # When checkpoint is not present, checkpoint_loaded is False and ready_for_inference is False
    assert isinstance(data["checkpoint_loaded"], bool)

def test_evaluation_summary_endpoint():
    response = client.get("/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "has_evaluation_data" in data

def test_invalid_upload_extension():
    response = client.post(
        "/screen",
        files={"file": ("test.txt", b"dummy text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "unsupported file format" in response.json()["detail"].lower()
