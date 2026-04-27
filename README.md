# linear-ar-kf

This repository contains the experimental code for the paper _"Two-Layer Linear Auto-Regressive Models Estimate Latent States"_ submitted to ICML 2026.

## What is in the repo

- `exp/`: reusable experiment code.
- `notebooks/`: experiment notebooks and notebook-generated figures.

## Setup

Create an environment with the core dependencies and install the repo in editable mode:

```bash
pip install -e .
```

That makes the experiment modules importable from notebooks anywhere in the repo:

```python
from exp.define_system import define_system
from exp.train_test import train_test, run_sweep
```

## Quick start

The current workflow is:

1. Define a linear dynamical system with `define_system`.
2. Simulate trajectories with `simulate`.
3. Train a two-layer linear autoregressive model with `train_test` or `train_only_AR`.
4. Analyze reconstruction quality or sample complexity in a notebook.

## Recommended layout for future experiments

Keep the reusable code in `exp/` and put notebook-specific analysis in `notebooks/`.

- If a notebook generates figures, save them into a notebook-specific folder rather than mixing them into the source tree.
- Prefer importing functions from `exp.*` instead of copying training code into each notebook.
- If an experiment grows beyond one notebook, promote the shared logic into a Python module under `exp/`.

## Main modules

- `exp/define_system.py`: system generation and controllability/observability checks.
- `exp/simulate.py`: trajectory simulation under different input and noise distributions.
- `exp/steady_state_KF.py`: steady-state Kalman gain and filtering utilities.
- `exp/dataloader.py`: autoregressive window construction and PyTorch dataloaders.
- `exp/train_test.py`: training, evaluation, and sample-complexity utilities.
- `exp/architecture_search.py`: hidden-dimension search for the autoregressive model.
