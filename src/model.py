"""
Simple CNN for CIFAR-10 classification.

In here it's where I have the CNN architecture, and where i make the changes in it
"""

import torch
import torch.nn as nn


class ResidualLayers(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_inputs, n_outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_outputs),
            nn.ReLU(),

            nn.Conv2d(n_outputs, n_outputs, 3, padding=1, bias=False),
            nn.BatchNorm2d(n_outputs),
        )
        self.relu = nn.ReLU()

        if n_inputs == n_outputs:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(n_inputs, n_outputs, 1)

    def forward(self, x):
        gx = self.skip(x)
        out = self.features(x) + gx

        return self.relu(out)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ResidualLayers(3, 64),

            nn.MaxPool2d(2, 2),

            ResidualLayers(64, 128),

            nn.MaxPool2d(2, 2),

            ResidualLayers(128, 256),
            ResidualLayers(256, 512),

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(512, 10)
        )
    def forward(self, x):
        logits = self.features(x)
        return logits


if __name__ == "__main__":
    # Quick sanity check: run `python src/model.py`
    model = CNN()
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expected: torch.Size([4, 10])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
