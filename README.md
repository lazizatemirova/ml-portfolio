# ML Portfolio – three small, reproducible machine-learning projects

Python · Pandas · NumPy · scikit-learn · PyTorch · FastAPI · pytest

| # | Project | Tools | Headline result |
|---|---------|-------|-----------------|
| 1 | [Tumor classification on tabular data](01-tabular-ml) | Pandas, NumPy, scikit-learn | ROC AUC **0.996**, accuracy **96.5 %**, malignant recall **92.9 %** (hold-out test, n = 114) |
| 2 | [Image classification: CNN and transfer learning](02-image-cnn) | PyTorch, torchvision, scikit-learn | CNN test accuracy **98.1 %** (13.7k parameters) vs. RBF-SVM baseline 99.6 % |
| 3 | [Serving the models through a REST API](03-model-api) | FastAPI, Pydantic, pytest | 2 prediction endpoints, input validation, 9 passing tests |

All datasets are bundled with scikit-learn, so nothing has to be downloaded and every number
below can be reproduced with a fixed seed (`42`). These are learning projects on small, clean
benchmark data – not clinical or production systems.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python 01-tabular-ml/train.py            # ~15 s  -> 01-tabular-ml/results + artifacts
python 02-image-cnn/train_cnn.py         # ~10 s  -> 02-image-cnn/results + artifacts
cd 03-model-api && pytest -q             # 9 tests
uvicorn app.main:app --reload            # open http://127.0.0.1:8000/docs
```

## 1 · Tumor classification (Pandas / NumPy / scikit-learn)

*Question:* can 30 numeric measurements of a cell nucleus separate malignant from benign tumors?

- **Data audit with Pandas/NumPy:** 569 rows, 30 features, 0 missing values, 0 duplicates, 37.3 % malignant.
  30.1 % of rows contain at least one IQR outlier – they are *kept*, because in medical data a large
  tumour is a real observation, not noise. 15 feature pairs have |r| > 0.95 (e.g. radius / perimeter / area).
- **Positive class = malignant**, so precision and recall mean what a clinician expects.
- **Four model families** (logistic regression, RBF-SVM, random forest, gradient boosting), each with a
  scaler in a `Pipeline` (no leakage) and tuned with `GridSearchCV` on stratified 5-fold CV, scored by ROC AUC.
- **Test set used once**, for the single best model (selected by CV ROC AUC).

| Model (best params) | CV accuracy | CV recall | CV F1 | CV ROC AUC |
|---|---|---|---|---|
| Logistic regression (C=1) | 0.974 | 0.953 | 0.964 | **0.996** |
| SVC (C=10, gamma=0.01) | 0.976 | 0.953 | 0.967 | 0.995 |
| Gradient boosting | 0.969 | 0.947 | 0.958 | 0.991 |
| Random forest | 0.965 | 0.941 | 0.952 | 0.989 |

**Hold-out test (n = 114), logistic regression:** accuracy 0.965 · precision 0.975 · recall 0.929 ·
F1 0.951 · ROC AUC 0.996 · confusion matrix: 71 TN, 1 FP, 3 FN, 39 TP.

What it shows / limits:
- The simple linear model won. On clean, well-engineered tabular data with ~450 training rows,
  complex models had nothing extra to learn.
- 3 of 42 malignant cases were missed. In a real screening setting recall would matter most, and the
  0.5 threshold would be lowered and the trade-off studied.
- Permutation importances are small (< 0.01 AUC) because many features are highly correlated:
  shuffling one is compensated by its twin. They show *which* groups matter, not exact weights.
- CV scores come from the same folds used for tuning, so they are slightly optimistic; the test numbers are the honest ones.
- Figures: `results/confusion_matrix.png`, `results/feature_importance.png`.

## 2 · Image classification (PyTorch)

*Data:* 1,797 handwritten-digit images (8×8 pixels, 10 classes), stratified 70/15/15 split.

**2a · CNN from scratch** (`train_cnn.py`): two conv blocks + a small classifier head, Adam, 40 epochs;
the checkpoint with the best *validation* accuracy is kept and the test set is touched once.

| Model | Test accuracy |
|---|---|
| CNN (13,706 parameters) | **0.9815** (macro-F1 0.981) |
| Logistic regression on raw pixels | 0.9741 |
| RBF-SVM on raw pixels | **0.9963** |

The honest finding: **a classic SVM beat the small CNN** on this tiny 8×8 dataset. CNNs earn their
keep on larger, higher-resolution images where local patterns matter; here there is too little data
and too little structure. Always benchmark against a simple baseline before reaching for deep learning.
(Test set = 270 images, so one image ≈ 0.4 percentage points – differences this small are partly noise.)
Figures: `results/training_curve.png`, `results/confusion_matrix_cnn.png`.

**2b · Fine-tuning a pre-trained ResNet-18** (`finetune_resnet.py`): ImageNet weights, new 10-class head,
two stages (head only → everything unfrozen with a smaller learning rate).
Compare `python finetune_resnet.py` with `python finetune_resnet.py --no-pretrained` to measure
what pre-training actually buys on this task.

> Results: _not filled in yet._ The script needs internet access to download the ImageNet weights, so
> run both commands once on your own machine and paste the two `test_accuracy` values from
> `results/metrics_resnet18_pretrained.json` and `results/metrics_resnet18_scratch.json` here.

## 3 · Model API (FastAPI)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness and which models are loaded |
| `GET /tumor/features` | names/order of the 30 input features |
| `POST /predict/tumor` | `{"features": [30 floats]}` → label, `probability_malignant`, model name |
| `POST /predict/digit` | `{"pixels": [64 floats, 0..16]}` → digit, confidence, class probabilities |

- Models are loaded once at start-up (FastAPI lifespan), not per request.
- Pydantic validates inputs (exact length, pixel range) and returns `422` for bad requests;
  `503` if a model file is missing.
- 9 `pytest` tests cover health, feature list, API-vs-model consistency, validation errors and a
  sanity accuracy check through the HTTP layer.
- Swagger UI is generated automatically at `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/predict/digit \
     -H "Content-Type: application/json" \
     -d '{"pixels": [0,0,5,13,9,1,0,0, ... 64 values ...]}'
```

## Not covered (yet)

Prompt engineering, LangChain/LangGraph, n8n workflows and speech models (STT/TTS) are not part of
this repo. Deployment (Docker, cloud) and experiment tracking (MLflow) are natural next steps.
