from fastapi.testclient import TestClient

from src import api


class FakeModel:
    def predict(self, features):
        assert len(features) == 1
        return [1]

    def predict_proba(self, features):
        assert len(features) == 1
        return [[0.2, 0.8]]


def test_health_and_prediction_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "model", None)

    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
        response = client.post("/predict", json={"age": 50})

    assert response.status_code == 503


def test_prediction_response(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(api, "model", None)
    payload = {
        "age": 57,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 236,
        "fbs": 0,
        "restecg": 0,
        "thalach": 174,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 2,
    }

    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "model", FakeModel())
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["label"] == "disease_present"
    assert body["confidence"] == 0.8
    assert body["probability_disease_present"] == 0.8
    assert "Educational demonstration only" in body["disclaimer"]
