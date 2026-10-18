"""FastAPI inference service. Expects a JSON object with original feature names."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning(
            "Model artifact missing at %s; train the model before prediction",
            MODEL_PATH,
        )
    yield


app = FastAPI(
    title="Heart Disease Risk Classifier",
    version="1.0.0",
    description="Educational ML demo; not a medical device or diagnostic service.",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = datetime.now(UTC)
    response = await call_next(request)
    elapsed_ms = (datetime.now(UTC) - start).total_seconds() * 1000
    logger.info(
        "method=%s path=%s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    # Lightweight operational status payload; this is not Prometheus exposition text.
    return {
        "service": "heart-disease-api",
        "model_loaded": model is not None,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


@app.post("/predict")
def predict(payload: dict[str, Any]):
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Train model first."
        )
    if not payload:
        raise HTTPException(
            status_code=422, detail="Provide a non-empty JSON object of feature values"
        )
    try:
        row = pd.DataFrame([payload])
        prediction = int(model.predict(row)[0])
        probabilities = model.predict_proba(row)[0]
        confidence = float(max(probabilities))
        positive_probability = float(probabilities[1])
    except Exception as exc:
        # Do not include submitted values or exception details in service logs.
        logger.warning("Prediction rejected: error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=422, detail=f"Invalid feature payload: {exc}"
        ) from exc
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_detected",
        "confidence": round(confidence, 4),
        "probability_disease_present": round(positive_probability, 4),
        "disclaimer": "Educational demonstration only; not medical advice or diagnosis.",
    }
