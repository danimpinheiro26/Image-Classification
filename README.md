# cnn-evolution

This repo is my learning log for machine learning and PyTorch, built around
one small, evolving project: a CNN trained on CIFAR-10.

I'm not trying to build a state-of-the-art model here. The point is to have
a real, working piece of code that I keep coming back to and improving as I
learn — so I (and anyone else) can see how my understanding of deep learning
progressed over time, not just look at a finished result.

## What's here

- `src/model.py` — the CNN architecture
- `src/train.py` — training loop, checkpointing, per-run result logging
- `src/plot_evolution.py` — plots accuracy across all past runs
- `notebooks/` — messier exploration (trying ideas, visualizing things)
- `experiments/` — a JSON summary of every training run, kept in git so the
  history of results survives even as the code changes

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```powershell
python src\train.py --epochs 15 --batch-size 128 --lr 1e-3
```

CIFAR-10 downloads automatically the first time. Each run saves a summary
to `experiments/<timestamp>.json`.

## How I'm tracking progress

Every change I make — a bug fix, a new layer, a different optimizer — is
its own git commit with a message explaining what changed and why. That
commit history *is* the point of this repo: it's a record of what I tried,
what worked, and what I learned along the way.

## Current state

A small CNN (3 conv blocks + 2 fully connected layers, ~1.2M params).
Nothing fancy yet — this is the baseline I'm building on.

## What I'm learning next

- [ ] Understanding data augmentation better (comparing with/without)
- [ ] Residual connections
- [ ] Reading my own `experiments/*.json` results critically instead of just collecting them
