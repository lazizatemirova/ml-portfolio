"""Small CNN for 8x8 grayscale digit images. Shared by train_cnn.py.
(03-model-api/app/cnn.py holds an identical copy so the API stays self-contained.)"""
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 8x8 -> 4x4
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 4x4 -> 2x2
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(32 * 2 * 2, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
