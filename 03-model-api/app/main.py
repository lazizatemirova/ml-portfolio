"""Project 3 - serving both models from projects 1 and 2 through a REST API (FastAPI).

Endpoints
  GET  /health            liveness + which models are loaded
  GET  /tumor/features    the 30 feature names, in the order /predict/tumor expects
  POST /predict/tumor     {"features": [30 floats]} -> label + probability
  POST /predict/digit     {"pixels": [64 floats in 0..16]} -> digit + class probabilities

Interactive docs (Swagger UI) are generated automatically at /docs.
Run:  uvicorn app.main:app --reload     (from the 03-model-api folder)

The tumor endpoint is an educational demo and is NOT medical advice.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .cnn import SmallCNN

ROOT = Path(__file__).resolve().parents[2]
TUMOR_PATH = Path(os.getenv("TUMOR_MODEL_PATH", ROOT / "01-tabular-ml" / "artifacts" / "tumor_model.joblib"))
DIGIT_PATH = Path(os.getenv("DIGIT_MODEL_PATH", ROOT / "02-image-cnn" / "artifacts" / "digit_cnn.pt"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at start-up, not on every request."""
    app.state.tumor = joblib.load(TUMOR_PATH) if TUMOR_PATH.exists() else None
    app.state.digit = None
    if DIGIT_PATH.exists():
        cnn = SmallCNN()
        cnn.load_state_dict(torch.load(DIGIT_PATH, map_location="cpu"))
        cnn.eval()
        app.state.digit = cnn
    yield


app = FastAPI(title="ML Portfolio API", version="1.0.0", lifespan=lifespan)


class TumorRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30,
                                  description="30 numeric features, order from GET /tumor/features")


class TumorResponse(BaseModel):
    label: str
    probability_malignant: float
    model: str


class DigitRequest(BaseModel):
    pixels: list[float] = Field(..., min_length=64, max_length=64,
                                description="8x8 image flattened row by row, values 0..16")

    @field_validator("pixels")
    @classmethod
    def in_range(cls, v: list[float]) -> list[float]:
        if any(p < 0 or p > 16 for p in v):
            raise ValueError("pixel values must be between 0 and 16")
        return v


class DigitResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


def _require(request: Request, name: str):
    model = getattr(request.app.state, name)
    if model is None:
        raise HTTPException(status_code=503, detail=f"{name} model is not loaded - run the training script first")
    return model


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok",
            "tumor_model_loaded": request.app.state.tumor is not None,
            "digit_model_loaded": request.app.state.digit is not None}


@app.get("/tumor/features")
def tumor_features(request: Request) -> dict:
    return {"features": _require(request, "tumor")["feature_names"]}


@app.post("/predict/tumor", response_model=TumorResponse)
def predict_tumor(body: TumorRequest, request: Request) -> TumorResponse:
    bundle = _require(request, "tumor")
    row = pd.DataFrame([body.features], columns=bundle["feature_names"])  # keep column names the model was trained with
    proba = float(bundle["model"].predict_proba(row)[0, 1])
    return TumorResponse(label="malignant" if proba >= 0.5 else "benign",
                         probability_malignant=round(proba, 4), model=bundle["model_name"])


@app.post("/predict/digit", response_model=DigitResponse)
def predict_digit(body: DigitRequest, request: Request) -> DigitResponse:
    cnn = _require(request, "digit")
    x = torch.tensor(body.pixels, dtype=torch.float32).view(1, 1, 8, 8) / 16.0
    with torch.no_grad():
        probs = torch.softmax(cnn(x), dim=1)[0]
    return DigitResponse(digit=int(probs.argmax()), confidence=round(float(probs.max()), 4),
                         probabilities=[round(float(p), 4) for p in probs])
