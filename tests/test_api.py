"""Local smoke tests. Run after the existing model artifact is present."""
from fastapi.testclient import TestClient
from api.main import app

SAMPLE = {
    "age": 50,"bp": 80,"bgr": 100.0,"bu": 36.0,"sc": 1.1,
    "sod": 140.0,"pot": 4.3,"hemo": 14.0,"pcv": 42.0,
    "wbcc": 7500.0,"rbcc": 4.8,"sg": "1.020","al": "0","su": "0",
    "rbc": "normal","pc": "normal","pcc": "notpresent","ba": "notpresent",
    "htn": "no","dm": "no","cad": "no","appet": "good","pe": "no","ane": "no",
}

def test_full_workflow():
    # Context manager invokes FastAPI lifespan, including model loading.
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["status"] == "healthy"
        assert client.get("/ready").json()["ready"] is True
        info = client.get("/model-info").json()
        assert info["number_of_features"] == 24
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["prediction"] in {"CKD","Not CKD"}
        assert abs(sum(body["probabilities"].values()) - 1) < 1e-8
        assert client.post("/predict", json={}).status_code == 422
        assert client.post("/predict", json={**SAMPLE, "age": -1}).status_code == 422
        assert client.post("/predict", json={**SAMPLE, "sg": "1.030"}).status_code == 422
        assert client.post("/predict", json={**SAMPLE, "bgr": "banana"}).status_code == 422
        metrics = client.get("/metrics").json()
        assert metrics["successful_predictions"] >= 1
        assert metrics["failed_requests"] >= 4
        assert client.get("/dashboard").status_code == 200
        assert client.get("/dashboard-assets/app.js").status_code == 200
        assert client.get("/dashboard-assets/styles.css").status_code == 200
