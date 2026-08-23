# Image Classification

A hands-on learning project for tracking my progress in machine learning and PyTorch through image classification.

My goal isn't to build the best possible model, but to track my progress and thinking process.

The code, experiments, and git history are all part of the learning process.

## What's Inside

```text
src/
├── model.py          # Model architecture
├── train.py          # Training loop, checkpointing, and result logging
└── plot_evolution.py # Visualize progress across previous runs

notebooks/            # Messy exploration
experiments/          # JSON summaries of training runs
```

## Getting Started

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

Run the current model with:

```powershell
python src\train.py
```

The dataset is downloaded automatically the first time it is needed.

Each training run saves a summary to:

```text
experiments/<timestamp>.json
```

This allows me to keep track of how different changes affect the results over time.


## Current State

The project currently features 8 convolutional layers, with residual connections at every 2 layers, followed by a final fully connected layer.
It got a 88% accuracy on the validation set.
