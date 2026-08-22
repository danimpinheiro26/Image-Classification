"""
Training script for SimpleCNN on CIFAR-10.

To try different settings, just edit the hyperparameters below and run:
    python src/train.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleCNN

# ---- Hyperparameters ----
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SCHEDULER_PATIENCE = 10
# -------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"


def get_dataloaders():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode = 'max',
        patience = SCHEDULER_PATIENCE,
    )

    CKPT_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, seen = 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen += images.size(0)

        train_loss = running_loss / seen
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "best_model.pt")
        scheduler.step(val_acc)

    print(f"Done. Best val_acc={best_acc:.4f}")

    # Save a summary of this run so you can compare it to future ones.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    summary = {
        "run_id": run_id,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "best_val_acc": best_acc,
        "history": history,
    }
    with open(EXPERIMENTS_DIR / f"{run_id}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary to experiments/{run_id}.json")


if __name__ == "__main__":
    main()