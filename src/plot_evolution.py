"""
Plot best validation accuracy across every run in experiments/, in order.

This is the "evolution" chart: run it after a few training sessions to see
your progress over time.

Usage:
    python src/plot_evolution.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_runs():
    runs = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        runs.append(data)
    return runs


def main():
    runs = load_runs()
    if not runs:
        print("No experiment files found in experiments/. Run src/train.py first.")
        return

    run_ids = [r["run_id"] for r in runs]
    best_accs = [r["best_val_acc"] for r in runs]

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(runs)), best_accs, marker="o")
    plt.xticks(range(len(runs)), run_ids, rotation=45, ha="right")
    plt.ylabel("Best validation accuracy")
    plt.title("Model evolution over training runs")
    plt.tight_layout()
    plt.savefig(EXPERIMENTS_DIR / "evolution.png")
    print(f"Saved plot to {EXPERIMENTS_DIR / 'evolution.png'}")


if __name__ == "__main__":
    main()
