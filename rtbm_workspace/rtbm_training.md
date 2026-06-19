# RTBM Training — How It All Works

This document traces the full execution path of `training.py`, explaining every layer from high-level training logic down to the Riemann theta function at the mathematical core.

---

## 1. What Is an RTBM?

An **RTBM (Riemann-Theta Boltzmann Machine)** is a generative density model. Given a dataset of events (here: τ→πν pion decay tracks), it learns a probability distribution P(v) over the visible feature space. After training on signal, anomalous events (τ→ρν) will have a lower learned probability — making −log P(v) a natural anomaly score.

The RTBM is an extension of the classical Restricted Boltzmann Machine (RBM). In a standard RBM the hidden units are binary. The RTBM replaces the discrete marginalisation sum over hidden states with a **Riemann theta function** — an infinite sum over the integer lattice Z^g — which allows the hidden units to be continuous and gives the model more expressive power.

---

## 2. The Physical Feature Space

Each event is a 4-dimensional vector:

| Index | Variable | Meaning |
|-------|----------|---------|
| 0 | x_vis | Track momentum fraction |
| 1 | Iso = ΣE_ph/E_track | Photon isolation (clipped 0–1) |
| 2 | f_had = E_HCAL/E_tot | Hadronic energy fraction |
| 3 | η (scaled) | Pseudorapidity, shifted by ETA_MAX=2.5 and divided by 5 |

Preprocessing (done inside `load_datasets()`):
- Column 1 is clipped to [0, 1].
- Column 3 is linearly rescaled: `(η − 2.5) / 5`.
- All columns are then **z-score standardised** (zero mean, unit variance) using training-set statistics, before being fed to the model.

The `standardize()` function returns transformed train/val/test/rho arrays and the (μ, σ) pair saved with the model for later inference.

---

## 3. RTBM Parameters

The model `RTBM(N_VISIBLE=4, N_HIDDEN=nh)` from `theta/theta/rtbm.py` has five matrix parameters:

| Symbol | Shape | Role |
|--------|-------|------|
| **Bv** | (Nv, 1) | Visible bias vector |
| **Bh** | (Nh, 1) | Hidden bias vector |
| **T** | (Nv, Nv) | Positive-definite precision matrix on visible units (symmetric) |
| **W** | (Nv, Nh) | Visible–hidden coupling weights |
| **Q** | (Nh, Nh) | Positive-definite matrix on hidden units (symmetric) |

For Nv=4, Nh=2 (default), the total parameter count is:

```
size = Nv + Nh + Nv*Nh + (Nv²+Nv)/2 + (Nh²+Nh)/2
     = 4   + 2  +  8    +    10       +    3
     = 27 parameters
```

CMA-ES searches this 27-dimensional space.

---

## 4. The Probability Formula

The model assigns a probability to each visible vector v ∈ ℝ^Nv via:

```
P(v) = sqrt(det(T) / (2π)^Nv)
       × exp(−½ vᵀTv − Bvᵀv − ½ Bvᵀ T⁻¹ Bv)
       × θ(vᵀW + Bhᵀ | −Q) / θ(Bhᵀ − Bvᵀ T⁻¹ W | −Q + Wᵀ T⁻¹ W)
```

where `θ(z | Ω)` is the Riemann theta function (see Section 6).

Implemented in `mathtools.py → rtbm_probability()`.

**Intuition:**
- The prefactor `sqrt(det(T)/(2π)^Nv) × exp(−½ vᵀTv − Bvᵀv − …)` is a **Gaussian** envelope over visible space, controlled by T and Bv.
- The **theta ratio** is the non-Gaussian correction that captures higher-order correlations through the hidden units. It is a data-dependent modulation: the numerator θ depends on each event v (through `vᵀW`), while the denominator θ is the normalisation constant (no v dependence).

---

## 5. Schur-Complement Initialisation

Before training starts, `RTBM.random_init(bound)` creates parameters that **already satisfy the positivity constraint** needed for a valid probability distribution.

The requirement is that both T and Q must be positive definite, and additionally the Schur complement:

```
Q − Wᵀ T⁻¹ W > 0   (positive definite)
```

This is checked by `check_normalization_consistency(T, Q, W)` in `mathtools.py`.

The trick: draw a random `((Nv+Nh) × (Nv+Nh))` matrix X, compute `A = Xᵀ X` (always positive semi-definite), and carve:
```
Q = A[:Nh, :Nh]
T = A[Nh:, Nh:]
W = A[Nh:, :Nh]
```
Because A is PSD, the Schur complement of Q in A is automatically non-negative, satisfying the constraint from the start.

---

## 6. The Riemann Theta Function

The Riemann theta function is defined as:

```
θ(z, Ω) = Σ_{n ∈ Z^g}  exp(2πi (½ nᵀΩn + nᵀz))
```

where the sum runs over **all integer lattice points** n ∈ Z^g and Ω is a complex symmetric matrix with positive-definite imaginary part.

### How it is called

In `mathtools.py`, the theta function is always called with purely imaginary arguments in Phase I mode:

```python
# Numerator theta (event-dependent):
z1 = (vᵀW + Bhᵀ) / (2πi)      # shape (N_events, Nh)
Ω1 = −Q / (2πi)                 # shape (Nh, Nh)

# Denominator theta (normalisation, no v):
z2 = (Bhᵀ − Bvᵀ T⁻¹ W) / (2πi)
Ω2 = (−Q + Wᵀ T⁻¹ W) / (2πi)
```

The `/2πi` re-parameterisation turns the imaginary exponent into a real Gaussian-like decay, making the sum converge.

### Numerical computation (`theta/riemann_theta/`)

The `RiemannTheta` object in `riemann_theta.pyx` is a compiled Cython wrapper around C routines. The computation proceeds in two steps:

1. **Radius determination** (`radius.pyx`): computes the truncation radius R such that all lattice points beyond R contribute less than ε = 1e-8 to the sum.

2. **Integer points enumeration** (`integer_points.pyx`): enumerates all n ∈ Z^g with ‖T n‖ ≤ R, where T is the Cholesky factor of Im(Ω).

3. **Finite sum** (`finite_sum.c`): evaluates the exponential sum over those N lattice points in C for speed.

`parts_eval()` returns separately the **exponential part** (u) and **oscillatory part** (v) of theta to avoid numerical overflow:

```
θ(z, Ω) = exp(u) × v
```

The probability then uses `exp(u_num − u_denom) × (v_num / v_denom)` — computing in log-space first to stay numerically stable across many events.

### Derivative calls

For the `backprop()` method (gradient-based training, not used here), the theta function is also evaluated with first and second directional derivatives, passing a `derivs` list:

```python
# First derivative w.r.t. z in direction e_i:
derivs=[[0,...,1,...,0]]   # 1 in position i

# Second derivative (Hessian):
derivs=[[0,...,1,...,0], [0,...,1,...,0]]
```

These are only needed for SGD/BFGS training. The CMA-ES path used in `training.py` is **gradient-free**.

---

## 7. The Cost Function: Logarithmic (Negative Log-Likelihood)

`theta/theta/costfunctions.py → class logarithmic`:

```python
cost(x) = -sum(log(x))
```

where `x = P(v_i)` for each training event. This is the standard **maximum likelihood** objective: minimising `−Σ log P(v_i)` is equivalent to maximising the likelihood of the training data under the model.

The `worker_compute()` function in `minimizer.py` performs one fitness evaluation:
1. Try `model.set_parameters(params)` — returns False if the positivity constraint is violated.
2. If invalid: return `NaN` (candidate is discarded by CMA-ES).
3. Otherwise: call `model(x_data)` to get P(v) for all training events.
4. Return `logarithmic.cost(P)` = −Σ log P(v_i).

---

## 8. The CMA-ES Optimiser

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** is an evolutionary algorithm for black-box optimisation. It is used because:
- The parameter space has hard constraints (positivity) that make gradient methods fragile.
- The landscape may be multi-modal.
- No analytic gradient is needed.

### How it works (iteration of `train_rtbm()`)

```
1. Initialise population: sample ~popsize candidate parameter vectors
   around initsol with step-size sigma = max_bound × 0.1

2. Each generation:
   a. Ask CMA-ES for popsize candidate solutions
   b. For each candidate:
      - worker_compute(candidate) → fitness value
      - If NaN (constraint violated): discard and re-ask
   c. Tell CMA-ES the valid (solution, fitness) pairs
   d. CMA-ES updates its internal mean and covariance matrix
      to model the shape of the loss landscape

3. Stop when:
   - cost function change < tolfun (default 1e-5), OR
   - maxiter reached (default 200)

4. model.set_parameters(es.result[0])  ← best solution wins
```

### Parallelisation

When `ncores > 1`, a `multiprocessing.Pool` is created. Workers are initialised with `worker_initialize()`, which copies the model and data into each worker's global `resource` object. Fitness evaluations are then dispatched with `pool.map_async()`. This speeds up each generation by evaluating candidates in parallel.

The `with closing(pool)` + `pool.terminate()` pattern ensures clean shutdown even if the loop exits early.

---

## 9. `training.py` End-to-End Flow

```
load_datasets()
│
│  pi.npy  → (100 000, 4) raw signal events
│  rho.npy → background events
│  clip column 1, rescale column 3
│
standardize(train_pi, val_pi, test_pi, rho_data)
│
│  mu, std computed from train_pi only
│  all arrays transformed: (x - mu) / std
│
Transpose: shape (N_features=4, N_events)   ← theta requires this layout
│
[optional] Bayesian hyperparameter search (--optimize flag)
│   hyperopt.fmin + tpe.suggest
│   search space: n_hidden ∈ {1,2,3}, param_bound ∈ [5, 50] log-uniform
│   each trial: run fast CMA-ES (80 iters) → return val NLL
│   pick best (n_hidden, param_bound)
│
RTBM(N_VISIBLE=4, n_hidden, init_max_param_bound=param_bound, random_bound=1)
│   random_init() ← Schur-complement initialisation
│
train_rtbm(model, X_tr, ncores, maxiter, tolfun)
│   CMA-ES loop (see Section 8)
│   returns (best_params, history list of per-iteration best cost)
│
model.set_parameters(best_params)
│
Evaluation
│   anomaly_scores(model, X_test) = −log P(v) for test pions
│   anomaly_scores(model, X_rho)  = −log P(v) for rho background
│
Plots
│   history.png       — CMA-ES convergence curve (log scale)
│   density_check.png — P(x)-weighted histograms vs raw validation data
│   anomaly_scores.png— score distributions + 95% efficiency threshold
│   roc_curve.png     — ROC curve + AUC
│
[optional --save] pickle model parameters + normalisation stats
```

---

## 10. Anomaly Detection Logic

The RTBM is trained **only on pion signal**. After training:

```python
anomaly_score(v) = −log P(v)
```

- **Low score**: the model assigns high probability → likely looks like training signal.
- **High score**: the model assigns low probability → anomalous (background-like).

At inference time for background rejection:
1. Compute scores for all test pions → set threshold at the 95th percentile (keeps 95% signal efficiency).
2. Apply threshold to rho events → `bkg_rejection = fraction of rho > threshold`.

The ROC curve sweeps all thresholds and AUC quantifies overall discrimination power.

---

## 11. Key Gotchas

**Shape convention**: The theta library requires data as `(N_features, N_events)` — the transpose of the usual ML convention. This is done explicitly before calling the model:
```python
X_tr = tr_std.T   # shape (4, N_train)
```

**Positivity constraint**: If a CMA-ES candidate violates T > 0, Q > 0, or Q − WᵀT⁻¹W > 0, `set_parameters()` returns False and `worker_compute()` returns NaN. CMA-ES keeps asking for replacements until a full valid population is assembled. This is why the inner while-loop in `train_rtbm()` uses `while len(solutions) < es.popsize` rather than a fixed iterate.

**Complex arithmetic**: The theta function is inherently complex-valued; `np.real()` is called on the model output to discard vanishingly-small imaginary parts (numerical noise). Probabilities must be positive real numbers by construction if the positivity constraints hold.

**Numerical stability**: `rtbm_probability()` computes the theta ratio using `parts_eval()` which returns (u, v) = (log exponential part, oscillatory part) separately, then assembles `exp(u_num − u_denom) × (v_num / v_denom)`. This avoids computing `exp(u_num)` and `exp(u_denom)` individually, which would overflow for large inputs.

---

## 12. File Map

```
rtbm_workspace/
├── training.py                   ← main script (this document)
├── main.py                       ← earlier simpler prototype
└── theta/                        ← git submodule
    └── theta/
        ├── rtbm.py               ← RTBM class, set/get_parameters, mode dispatch
        ├── mathtools.py          ← rtbm_probability(), rtbm_log_probability(),
        │                            hidden_expectations(), check_pos_def()
        ├── costfunctions.py      ← logarithmic cost = −Σ log P
        ├── minimizer.py          ← CMA class, worker_initialize, worker_compute
        └── riemann_theta/
            ├── riemann_theta.pyx ← RiemannTheta object, eval/log_eval/parts_eval
            ├── finite_sum.c      ← C inner loop over integer lattice points
            ├── radius.pyx        ← truncation radius R for the lattice sum
            └── integer_points.pyx← enumeration of Z^g lattice within radius R
```
