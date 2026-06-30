# RTBM Performance Study

## 1. What to Measure and Why

A complete performance study tracks two layers: the **inner loop** (how well CMA-ES optimises a single model) and the **outer loop** (how model quality depends on hyperparameters).

### Primary metrics

| Metric | What it measures | Where it lives |
|--------|-----------------|----------------|
| Validation NLL / event | Density model quality (lower = better fit) | `mean_nll(model, X_val)` after training |
| AUC | Physics separation power (higher = better anomaly detection) | `roc_curve` + `auc` in `plot_roc` |
| Background rejection @ 95% eff. | Operational physics figure of merit | `plot_anomaly_scores` |
| CMA convergence curve | Speed and stability of inner optimisation | `history` list in `train_rtbm` |

### Secondary diagnostics

| Metric | What it reveals |
|--------|----------------|
| Fraction of valid candidates per generation | How hard the Schur constraint is for a given (Nh, param_bound) pair |
| Fevals to reach target NLL | Fair comparison of convergence speed across Nh (see Section 2) |
| Time per CMA iteration | Runtime scaling with Nh and n_train |

---

## 2. Controlling for Computational Budget: Fevals, not Iterations

Comparing models across $N_h \in \{2, 3, 4\}$ at fixed `SEARCH_MAXITER` is **biased**: CMA-ES popsize grows with the number of parameters ($\approx 4 + 3\ln N$), so larger models get more function evaluations per iteration.

| $N_h$ | Params | Popsize | Fevals at 150 iters |
|--------|--------|---------|---------------------|
| 2 | 27 | ≈ 14 | ≈ 2100 |
| 3 | 35 | ≈ 15 | ≈ 2250 |
| 4 | 44 | ≈ 16 | ≈ 2400 |

The correct unit for inner-loop budget is **function evaluations** (`es.countevals` in the cma library), not iterations. For scaling plots, fix the total feval budget (e.g. 2000) across all $N_h$ values. In practice this means using a smaller `maxiter` for larger $N_h$.

---

## 3. Should Each Hyperopt Trial Run to Completion?

**Yes — and the reason is specific to the NAN_PENALTY strategy.**

With `tolfun=0` and `tolflatfitness=maxiter`, trials already run to `SEARCH_MAXITER` by default. The question is whether stuck trials (many Schur failures → all candidates receive NAN_PENALTY) should be truncated.

They should not. Keeping NAN_PENALTY constant and running the full budget is correct because:

1. **The loss landscape is identical for all configurations.** Truncating a badly-initialised trial after fewer fevals gives it an artificially low compute cost, making cross-config comparison meaningless.
2. **Stuck trials are informative.** A configuration that spends 150 iterations at NAN_PENALTY naturally receives validation NLL = 1e9 and ranks last — which is the correct outcome. Truncating it would hide whether it ever recovers.
3. **Adaptive penalty changes the landscape.** If you reduce the penalty over time to "help" stuck configurations, you modify the fitness surface mid-run and comparisons between configs that needed help vs those that did not become invalid.

The fraction of iterations spent at full penalty is itself a useful diagnostic (see Section 5).

---

## 4. What is Already Trackable in `training.py`

### Automatic (zero extra work)

- **`args.txt`**: all CLI hyperparameters written per run — provides the full experimental record.
- **`history.png` / `history` list**: CMA best NLL per iteration — convergence curve.
- **`roc_curve.png`**, **`anomaly_scores.png`**: physics performance metrics with numerical values printed to terminal.
- **`hyperopt_trials.pkl`**: the full hyperopt `Trials` object. Deserialise with `pickle.load` to access every trial's `{'loss': val_nll, 'tid': trial_id, 'misc': {'vals': {param: value}}}`. This gives a complete (N_hidden, param_bound, val_NLL) table for free.

### One-line additions

- **Training time**: wrap the main block with `time.time()` (already done in `mytraining.py`; not in `training.py`).
- **Fevals**: read `es.countevals` after the CMA loop and append to `history` or print it.

### Requires modifying `train_rtbm`

- **Valid candidate fraction per generation**: count how many entries in `fits` are finite vs NAN_PENALTY before the replacement step. Log the ratio each iteration. This reveals how aggressively the Schur constraint limits a given (Nh, param_bound) config.

```python
n_valid = sum(np.isfinite(v) for v in fits)
valid_fraction_history.append(n_valid / len(fits))
```

- **AUC per hyperopt trial**: the current `make_objective` only returns `val_NLL`. Adding AUC requires calling `anomaly_scores` and `roc_curve` inside the objective. It roughly doubles the evaluation cost (scoring the full test set) and is usually not worth it during the search — compute AUC only for the winning configuration.

---

## 5. What Needs Manual Search Infrastructure

### Scaling curves (NLL or AUC vs N_hidden, vs param_bound)

These require a **grid sweep**, not a Bayesian search. Hyperopt explores the space adaptively and does not provide an evenly-sampled grid for 2D plots. You need a bash loop:

```bash
for nh in 2 3 4; do
  for pb in 1.0 2.0 3.0 5.0 8.0; do
    python training.py --save -nh $nh --param_bound $pb \
      --maxiter 150 -o sweep_nh${nh}_pb${pb}
  done
done
```

Then a small post-processing script reads each `args.txt` + terminal output (or saved `model.pkl`) and assembles the heatmap.

### Multiple random seeds

The RTBM has three sources of randomness:

1. **`np.random.seed`** — controls data shuffling and `RTBM.random_init` (which uses `np.random.uniform` internally).
2. **CMA-ES** — uses numpy's random state by default; the `seed` key in `cma_opts` overrides it: `cma_opts['seed'] = seed`.
3. **Theta Cython C extensions** — the finite-sum computation is deterministic given the parameters; no random state.

So full reproducibility per seed requires only two lines:

```python
np.random.seed(seed)
cma_opts['seed'] = seed
```

The multiprocessing pool does **not** introduce seed-dependence: `worker_compute` contains no random operations (it only evaluates the log-probability of fixed data under fixed parameters).

For a seed sweep, expose `--seed` via argparse (replacing the hardcoded `np.random.seed(42)`) and run:

```bash
for seed in 1 2 3 4 5; do
  python training.py --save -nh 2 --seed $seed -o run_seed_${seed}
done
```

A post-processing script then aggregates validation NLL and AUC across seeds to produce mean ± std error bars.

---

## 6. Suggested Plot Catalogue

| Plot | x-axis | y-axis | Fixed variables |
|------|--------|--------|-----------------|
| Convergence curves | CMA iteration (or fevals) | Best NLL | N_hidden, one param_bound |
| Scaling with $N_h$ | N_hidden | Val NLL / AUC (mean ± std over seeds) | param_bound fixed at hyperopt best |
| Hyperparameter landscape | param_bound | Val NLL | N_hidden fixed, heatmap or line plot |
| Valid-candidate fraction | CMA iteration | Fraction of valid candidates | One (N_hidden, param_bound) combo |
| Physics performance | FPR | TPR (AUC = area) | Best (N_hidden, param_bound) |

The most informative single plot for this physics problem is **AUC vs N_hidden** with error bars from multiple seeds: it directly answers whether more hidden structure improves the $\tau \to \pi\nu$ vs $\tau \to \rho\nu$ separation beyond what the full $T$ matrix already captures via Gaussian correlations.
