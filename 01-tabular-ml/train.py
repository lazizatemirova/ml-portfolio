"""Project 1 - Breast tumor classification on tabular data.

Pandas/NumPy for the data audit, scikit-learn for modelling:
  * data audit (missing values, duplicates, outliers, correlated features)
  * 4 model families, each tuned with GridSearchCV on a stratified 5-fold split
  * the best model (by cross-validated ROC AUC) is evaluated ONCE on a held-out test set
  * confusion matrix + permutation feature importance plots, metrics saved as JSON

Dataset: Wisconsin Diagnostic Breast Cancer, bundled with scikit-learn (no download needed).
Educational project only - this is not a medical device.

Run:  python train.py
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42

# Newer scikit-learn versions deprecate SVC(probability=True) in favour of
# CalibratedClassifierCV. It still works and the argument exists in older versions
# too, so we keep it and silence only that one warning.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*probability.*")


def audit(df: pd.DataFrame, features: list[str]) -> dict:
    """Basic data-quality report. Outliers are only counted, not removed: in medical
    data a large tumour is a real observation, not noise."""
    X = df[features]
    q1, q3 = X.quantile(0.25), X.quantile(0.75)
    iqr = q3 - q1
    outliers = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
    corr = X.corr().abs().to_numpy()
    upper = np.triu_indices_from(corr, k=1)
    return {
        "rows": int(len(df)),
        "features": len(features),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "malignant_share": round(float(df["malignant"].mean()), 3),
        "share_of_rows_with_any_iqr_outlier": round(float(outliers.any(axis=1).mean()), 3),
        "feature_pairs_with_abs_corr_above_0.95": int((corr[upper] > 0.95).sum()),
    }


def candidates() -> dict:
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000, random_state=SEED),
            {"C": [0.01, 0.1, 1, 10]},
        ),
        "SVC": (
            SVC(probability=True, random_state=SEED),
            {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]},
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 5]},
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results_dir, artifacts_dir = args.out / "results", args.out / "artifacts"
    results_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn encodes malignant as 0; flip it so "positive" = malignant, which makes
    # precision/recall below mean what a clinician would expect.
    df["malignant"] = (df["target"] == 0).astype(int)
    df = df.drop(columns="target")
    features = [str(name) for name in data.feature_names]  # plain str, not numpy.str_
    X, y = df[features], df["malignant"]

    audit_info = audit(df, features)
    print("Data audit:", json.dumps(audit_info, indent=2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, (estimator, grid) in candidates().items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", estimator)])
        search = GridSearchCV(
            pipe, {f"clf__{k}": v for k, v in grid.items()}, scoring="roc_auc", cv=skf, n_jobs=2
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        cv = cross_validate(
            search.best_estimator_, X_train, y_train, cv=skf,
            scoring=["accuracy", "recall", "f1", "roc_auc"],
        )
        rows.append({
            "model": name,
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_accuracy": round(cv["test_accuracy"].mean(), 4),
            "cv_recall": round(cv["test_recall"].mean(), 4),
            "cv_f1": round(cv["test_f1"].mean(), 4),
            "cv_roc_auc": round(cv["test_roc_auc"].mean(), 4),
        })
        print(f"{name:20s} cv_auc={rows[-1]['cv_roc_auc']:.4f} params={rows[-1]['best_params']}")

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best = fitted[best_name]

    # Single, final look at the untouched test set.
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_malignant": round(precision_score(y_test, pred), 4),
        "recall_malignant": round(recall_score(y_test, pred), 4),
        "f1_malignant": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix [[TN, FP], [FN, TP]]": confusion_matrix(y_test, pred).tolist(),
        "n_test": int(len(y_test)),
    }
    print(f"\nBest model: {best_name}\nTest metrics: {json.dumps(test_metrics, indent=2)}")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["benign", "malignant"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"{best_name} - test set")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    imp = permutation_importance(
        best, X_test, y_test, n_repeats=20, random_state=SEED, scoring="roc_auc"
    )
    top = pd.Series(imp.importances_mean, index=features).nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    top.plot.barh(ax=ax, color="#1D4ED8")
    ax.set_xlabel("Drop in ROC AUC when feature is shuffled")
    ax.set_title("Top 10 features (permutation importance, test set)")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump(
        {"model": best, "feature_names": features, "model_name": best_name,
         "positive_class": "malignant"},
        artifacts_dir / "tumor_model.joblib",
    )
    (results_dir / "metrics.json").write_text(json.dumps(
        {"data_audit": audit_info, "selected_model": best_name,
         "selected_params": comparison.iloc[0]["best_params"],
         "test_metrics": test_metrics, "seed": SEED}, indent=2, default=str))


if __name__ == "__main__":
    main()
