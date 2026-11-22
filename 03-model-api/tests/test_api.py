"""API tests. Run from 03-model-api:  pytest -q
(needs the artifacts created by 01-tabular-ml/train.py and 02-image-cnn/train_cnn.py)"""
import joblib
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer, load_digits

from app.cnn import SmallCNN
from app.main import DIGIT_PATH, TUMOR_PATH, app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the `with` block runs the lifespan (loads the models)
        yield c


def test_health_reports_loaded_models(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "tumor_model_loaded": True, "digit_model_loaded": True}


def test_tumor_features_has_30_names(client):
    names = client.get("/tumor/features").json()["features"]
    assert len(names) == 30 and names[0] == "mean radius"


def test_tumor_prediction_matches_the_model_directly(client):
    data = load_breast_cancer()
    sample = data.data[0]
    bundle = joblib.load(TUMOR_PATH)
    expected = float(bundle["model"].predict_proba(pd.DataFrame([sample], columns=bundle["feature_names"]))[0, 1])

    res = client.post("/predict/tumor", json={"features": sample.tolist()})
    assert res.status_code == 200
    body = res.json()
    assert body["label"] in {"malignant", "benign"}
    assert body["probability_malignant"] == pytest.approx(expected, abs=1e-4)


def test_tumor_rejects_wrong_number_of_features(client):
    assert client.post("/predict/tumor", json={"features": [1.0, 2.0]}).status_code == 422


def test_digit_prediction_is_a_valid_distribution(client):
    pixels = load_digits().images[0].flatten().tolist()
    res = client.post("/predict/digit", json={"pixels": pixels})
    assert res.status_code == 200
    body = res.json()
    assert 0 <= body["digit"] <= 9
    assert sum(body["probabilities"]) == pytest.approx(1.0, abs=1e-2)


def test_digit_api_agrees_with_model_and_is_accurate(client):
    digits = load_digits()
    cnn = SmallCNN()
    cnn.load_state_dict(torch.load(DIGIT_PATH, map_location="cpu"))
    cnn.eval()
    correct = 0
    for img, label in zip(digits.images[:100], digits.target[:100]):
        res = client.post("/predict/digit", json={"pixels": img.flatten().tolist()}).json()
        with torch.no_grad():
            direct = int(cnn(torch.tensor(img, dtype=torch.float32).view(1, 1, 8, 8) / 16.0).argmax())
        assert res["digit"] == direct
        correct += res["digit"] == int(label)
    assert correct >= 90  # first 100 images include training data, so this is a sanity check only


@pytest.mark.parametrize("bad", [[-1.0] + [0.0] * 63, [17.0] + [0.0] * 63, [0.0] * 10])
def test_digit_rejects_invalid_pixels(client, bad):
    assert client.post("/predict/digit", json={"pixels": bad}).status_code == 422
