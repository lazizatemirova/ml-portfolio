"""Project 2b - Fine-tuning a pre-trained model (transfer learning).

Takes torchvision's ResNet-18 (pre-trained on ImageNet), swaps its last layer for a
10-class head and trains it in two stages:
  1. "head" stage  - backbone frozen, only the new classifier layer is trained
  2. "full" stage  - everything unfrozen, trained with a much smaller learning rate

Run it twice and compare - that is the whole point of the experiment:
    python finetune_resnet.py                  # starts from ImageNet weights
    python finetune_resnet.py --no-pretrained  # same network, random start

The ImageNet weights are downloaded by torchvision the first time (needs internet).
Uses the same 8x8 digits split as train_cnn.py, upscaled to 32x32 RGB for ResNet.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

from train_cnn import SEED, make_split

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare(x: np.ndarray) -> torch.Tensor:
    t = torch.tensor(x).unsqueeze(1)                              # (N, 1, 8, 8) in [0, 1]
    t = F.interpolate(t, size=32, mode="bilinear", align_corners=False)
    t = t.repeat(1, 3, 1, 1)                                      # grayscale -> 3 channels
    return (t - IMAGENET_MEAN) / IMAGENET_STD


def loader(x, y, batch_size, shuffle):
    return DataLoader(TensorDataset(prepare(x), torch.tensor(y, dtype=torch.long)),
                      batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    return torch.cat([model(xb).argmax(1) for xb, _ in dl]).numpy()


def run_stage(model, params, epochs, lr, train_dl, val_dl, y_val, state):
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()
        val_acc = accuracy_score(y_val, predict(model, val_dl))
        state["history"].append(round(val_acc, 4))
        if val_acc > state["best_val"]:
            state["best_val"] = val_acc
            state["best_weights"] = {k: v.clone() for k, v in model.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--full-epochs", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results_dir = args.out / "results"
    results_dir.mkdir(exist_ok=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = make_split()
    train_dl = loader(x_train, y_train, 64, True)
    val_dl = loader(x_val, y_val, 256, False)
    test_dl = loader(x_test, y_test, 256, False)

    model = resnet18(weights=None if args.no_pretrained else ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state = {"best_val": 0.0, "best_weights": None, "history": []}

    # Stage 1: only the new head learns.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    run_stage(model, model.fc.parameters(), args.head_epochs, 1e-3, train_dl, val_dl, y_val, state)

    # Stage 2: unfreeze everything, small learning rate.
    for p in model.parameters():
        p.requires_grad = True
    run_stage(model, model.parameters(), args.full_epochs, 1e-4, train_dl, val_dl, y_val, state)

    model.load_state_dict(state["best_weights"])
    pred = predict(model, test_dl)
    tag = "scratch" if args.no_pretrained else "pretrained"
    metrics = {
        "init": tag,
        "test_accuracy": round(accuracy_score(y_test, pred), 4),
        "test_macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "best_val_accuracy": round(state["best_val"], 4),
        "val_accuracy_per_epoch": state["history"],
        "head_epochs": args.head_epochs, "full_epochs": args.full_epochs, "seed": SEED,
    }
    print(json.dumps(metrics, indent=2))
    (results_dir / f"metrics_resnet18_{tag}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
