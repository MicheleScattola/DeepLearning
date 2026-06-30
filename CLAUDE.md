# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics ML research project studying tau lepton polarization at the FCC-ee collider. Two unsupervised anomaly-detection approaches are compared for distinguishing pion signal events (τ→πν) from rho background (τ→ρν→ππ⁰ν):

1. **Autoencoder** (TensorFlow/Keras) — trained only on pion signal; high reconstruction error flags background.
2. **RTBM** (Riemann-Theta Boltzmann Machine) — learns P(v) over the pion phase space; low log-likelihood flags anomalies.

## Environment Setup

Python 3.12 is required (enforced by `.python-version`). The project uses `uv` for dependency management.

```bash
uv sync            # install all dependencies into .venv
source .venv/bin/activate
```

The `rtbm_workspace/theta` directory is a git submodule (RiemannAI/theta). If it is missing or empty:
```bash
git submodule update --init --recursive
```
The theta Cython extensions must be compiled before `training.py` in `rtbm_workspace/` can run.

## Running the Training Scripts

Scripts must be run **from within their respective workspace directory** because they resolve dataset paths as `../datasets/`.

**Autoencoder:**
```bash
cd autoencoder_workspace
python training.py [--save] [-b BOTTLENECK] [-e EPOCHS] [-p PATIENCE] [-o OUTFILE]
python opt_training.py      # Bayesian hyperparameter search (hyperopt)
python evaluation.py        # ROC/anomaly score plots on saved model
```

**RTBM:**
```bash
cd rtbm_workspace
python training.py [--save] [--optimize] [-nh N_HIDDEN] [--maxiter N] [--ncores N] [--param_bound B]
python sweep.py [--nh ...] [--pb ...] [--seeds ...]   # performance sweep; see rtbm_workspace/CLAUDE.md
python plot_performance.py
```

Both training scripts create a `training/<outfile>/` directory with plots and an `args.txt` record.

**Data scraping** (requires ROOT files at `/mnt/data/physics_data/`):
```bash
cd data_scraping
python scrape.py            # extracts per-event 4-vectors from .root files → datasets/*.npy
python build_datasets.py    # mixes LH/RH samples at SM polarization P_tau = -0.147
```

## Data Pipeline

```
MadGraph/Pythia8/Delphes simulation
    → ROOT files (pi_LH/RH.root, rho_LH/RH.root)
    → data_scraping/scrape.py (uproot)
    → datasets/pi_LH.npy, pi_RH.npy, rho_LH.npy, rho_RH.npy
    → data_scraping/build_datasets.py
    → datasets/pi.npy (~100k events), rho.npy (background)
    → autoencoder_workspace/training.py or rtbm_workspace/training.py
```

The 4D feature vector per event is `[x_vis, I_gamma, f_had, eta]`. Both workspaces share this base preprocessing:
- Column 1 (`I_gamma`) clipped to `[0, 1]`
- Column 3 (`eta`) linearly rescaled: `(η − 2.5) / 5`

RTBM additionally z-score standardizes with training-set statistics, and applies a logit transform to column 0 (`x_vis`) before standardization — see `rtbm_workspace/math.md` Section 8 for why only `x_vis` (not `eta` or `f_had`) benefits from this.

## Architecture Notes

### Autoencoder (`autoencoder_workspace/`)
- Input/output: 4D → Dense(32, elu) → Dense(16, elu) → Dense(bottleneck, linear) → Dense(16, elu) → Dense(32, elu) → 4D
- Trained on pion signal only (80/20 train/val split); anomaly score = MSE reconstruction error
- TensorFlow parallelism is capped at `physical_cores / 2` to avoid over-subscription

### RTBM (`rtbm_workspace/`)
See **`rtbm_workspace/CLAUDE.md`** for the file layout, numerical gotchas, and performance-sweep tooling. Summary:
- Model: `RTBM(N_VISIBLE=4, N_HIDDEN, diagonal_T=False, mode=LogProbability)` from the `theta` submodule, constructed via `rtbmlib.make_rtbm` (handles initialisation and CMA bounds — do not call the bare `RTBM(...)` constructor directly)
- Optimizer: CMA-ES via the `cma` package, parallelized with `multiprocessing.Pool`; invalid (non-positive-definite) candidates get a finite `NAN_PENALTY` rather than being discarded and re-sampled
- Anomaly score: `−log P(v)`; model is trained only on pion signal
- **Critical shape convention**: the theta library requires data as `(N_features, N_events)` — the transpose of standard ML convention. Always transpose before passing to the model: `X_tr = train_data.T`
- Parameter validity: `set_parameters()` returns `False` if the positive-definiteness constraints (T > 0, Q > 0, Schur complement Q − WᵀT⁻¹W > 0) are violated
- Optional Bayesian hyperparameter search (`--optimize`) via `hyperopt` searches `n_hidden ∈ {2,3,4}` and `param_bound ∈ [1, 8]`
- `rtbm_workspace/math.md` documents every numerical bug fixed in this model — read it before touching `rtbmlib.make_rtbm` or `train_rtbm`

### Theta Submodule (`rtbm_workspace/theta/`)
The Riemann theta function is a Cython/C implementation. Key files:
- `theta/rtbm.py` — RTBM class, parameter get/set, mode dispatch
- `theta/mathtools.py` — `rtbm_probability()`, positivity checks
- `theta/minimizer.py` — `worker_initialize`, `worker_compute` (used by Pool)
- `theta/costfunctions.py` — `logarithmic` cost = −Σ log P
- `theta/riemann_theta/` — Cython wrappers + C finite-sum inner loop
