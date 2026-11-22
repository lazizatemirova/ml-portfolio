"""Project 2a - CNN image classifier in PyTorch, trained from scratch.

Data: the 8x8 handwritten-digit images bundled with scikit-learn (1,797 images, 10 classes).
  * stratified 70 / 15 / 15 train / validation / test split
  * the checkpoint with the best VALIDATION accuracy is kept; test set is used once at the end
  * the same split is given to two classic scikit-learn baselines (SVM, logistic regression)
    so the CNN result has something honest to be compared with

Run:  python train_cnn.py
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from model import SmallCNN

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split():
    digits = load_digits()
    images, labels = digits.images.astype("float32") / 16.0, digits.target
    x_train, x_rest, y_train, y_rest = train_test_split(
        images, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED
    )
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def to_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(torch.tensor(x).unsqueeze(1), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds = []
    for xb, _ in loader:
        preds.append(model(xb).argmax(dim=1))
    return torch.cat(preds).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results_dir, artifacts_dir = args.out / "results", args.out / "artifacts"
    results_dir.mkdir(exist_ok=True)
    artifacts_dir.mkdir(exist_ok=True)

    set_seed(SEED)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_split()
    train_loader = to_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = to_loader(x_val, y_val, 256, shuffle=False)
    test_loader = to_loader(x_test, y_test, 256, shuffle=False)

    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())

    history, best_val, best_state = [], 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        val_acc = accuracy_score(y_val, predict(model, val_loader))
        history.append({"epoch": epoch, "train_loss": running / len(y_train), "val_acc": val_acc})
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train_loss={history[-1]['train_loss']:.4f}  val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    test_pred = predict(model, test_loader)
    cnn_acc = accuracy_score(y_test, test_pred)

    # Baselines on identical data (flattened pixels).
    flat = lambda a: a.reshape(len(a), -1)
    baselines = {}
    for name, clf in {"SVC (rbf)": SVC(), "LogisticRegression": LogisticRegression(max_iter=5000)}.items():
        clf.fit(flat(x_train), y_train)
        baselines[name] = round(accuracy_score(y_test, clf.predict(flat(x_test))), 4)

    metrics = {
        "cnn_test_accuracy": round(cnn_acc, 4),
        "cnn_test_macro_f1": round(f1_score(y_test, test_pred, average="macro"), 4),
        "cnn_best_val_accuracy": round(best_val, 4),
        "baseline_test_accuracy": baselines,
        "parameters": n_params,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "epochs": args.epochs, "seed": SEED,
    }
    print(json.dumps(metrics, indent=2))
    (results_dir / "metrics_cnn.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, test_pred, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"CNN confusion matrix (test accuracy {cnn_acc:.3f})")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix_cnn.png", dpi=150)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax1.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], color="#1D4ED8")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="#1D4ED8")
    ax2 = ax1.twinx()
    ax2.plot([h["epoch"] for h in history], [h["val_acc"] for h in history], color="#DC2626")
    ax2.set_ylabel("validation accuracy", color="#DC2626")
    fig.tight_layout()
    fig.savefig(results_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), artifacts_dir / "digit_cnn.pt")


if __name__ == "__main__":
    main()
