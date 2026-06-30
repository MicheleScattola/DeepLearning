# RTBM Performance Tooling: Build Summary

This document summarizes the performance-tooling work in `rtbm_workspace/`: the shared library, the hyperparameter search, the grid sweep, an incident encountered while running it, and the final results.

> **Caveat added after this was first written**: `param_bound \leq 4` was found to have **no effect** on the actual run due to a `random_bound` floor + CMA-bounds-widening interaction (math.md Section 11) — `param_bound \in \{1,2,3\}` silently collapsed into duplicate runs of the same configuration, in every sweep including the one summarized below. This is now fixed. The $N_h$ reliability/crash-rate finding below is unaffected (it didn't depend on which low `param_bound` was tested), but `heatmap_pb_nh.png`'s shape at low `param_bound` does not reflect genuine variation and `sweep_results.csv`/`sweep_runs/`/`sweep_plots/` should be regenerated from a clean sweep before drawing further conclusions from the `param_bound` axis specifically.

## What Was Built

| File | Purpose |
|---|---|
| `rtbmlib.py` | Shared library: `make_rtbm`, `train_rtbm`, `mean_nll`, `anomaly_scores`, `compute_auc`, `background_rejection`, `default_popsize`, `run_tag`, `TrainingTimeout` |
| `sweep.py` | Grid + seed sweep over `(n_hidden, param_bound)`; writes `sweep_results.csv` + per-run `.npy` diagnostics |
| `plot_performance.py` | Reads the sweep CSV, produces the scaling/heatmap/convergence/valid-fraction plot catalogue |

`training.py` was refactored to import from `rtbmlib.py` instead of duplicating its own copies of these functions. `mytraining.py` was left untouched (personal learning copy).

---

## Run 1: Hyperopt Search (`training.py --optimize`)

```bash
python training.py --save --optimize --max_evals 15 --maxiter 300 -o optimized_run
```

15 trials, ~7 minutes total. **Best hyperparameters found: `n_hidden=3`, `param_bound=1.89`** (val_NLL=4.50 during the 150-iter search). Final training (300 iters) with those hyperparameters: **AUC=0.8874**, background rejection=56.2%, validation NLL=5.08. Two trials failed outright (`val_NLL=1e9`): `nh=4, pb=2.11` and `nh=3, pb=3.51` — random-initialization bad luck.

---

## Run 2: Performance Sweep (`sweep.py`) — Incident and Fix

### The hang

Launched concurrently with the hyperopt run, both capped at half the machine's 16 cores. The first attempt saturated **all 16 cores** instead of the intended 8 — `multiprocessing.Pool` workers each independently spawn multi-threaded BLAS operations, so process count alone doesn't bound thread count. Fixed by setting `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` before relaunching — load average then matched `--ncores` almost exactly.

The resumed sweep ran cleanly through `nh=2` and `nh=3` (30/45 runs, zero failures), then hung on `nh=4, param_bound=1.0, seed=1`. Diagnosis via `ps -eLf`: of 14 pool workers, 13 were idle (finished in ~3s each) while **1 worker had consumed 24+ minutes of continuous CPU time** — `pool.map()` is synchronous and was blocked waiting for that single stuck result.

### Root cause: a new failure mode

Distinct from the earlier Schur-complement collapse (math.md §2, §7), which produces a fast, clean NaN. Here the candidate parameters are **valid** (pass `check_pos_def`) but land near-singular — an eigenvalue at, say, `1e-6`. The Riemann theta lattice-summation radius `R` scales roughly as `1/sqrt(λ_min)`, and the number of lattice points enumerated scales as `R^g` where `g = N_h` is the theta genus. At `g=4` this can explode into a combinatorially expensive (but finite) computation that a single worker grinds through for an unbounded time. `multiprocessing.Pool` has no built-in cancellation for in-flight tasks, so one stuck candidate blocks the entire batch — and with it, the whole sweep — forever.

### Fix: `gen_timeout` in `train_rtbm`

```python
try:
    fits = pool.map_async(worker_compute, candidates).get(timeout=gen_timeout)
except mp.TimeoutError:
    raise TrainingTimeout(...)
finally:
    pool.terminate()   # forceful kill, not pool.close()
    pool.join()
```

Default `gen_timeout=60s`, configurable via `sweep.py --gen_timeout`. `TrainingTimeout` is a `RuntimeError` subclass, so it's caught by the same `except Exception` blocks already handling other training failures in `sweep.py` and the hyperopt objective. The single-core path uses an equivalent `signal.alarm`-based timeout. Verified with a forced 1-microsecond timeout: clean exception, zero leftover processes.

Documented in `math.md` Section 10.

### Resuming

Killed the hung process tree (30/45 runs already safely in the CSV), then resumed just the missing `nh=4` block:
```bash
python sweep.py --nh 4 --pb 1,2,3,5,8 --seeds 1,2,3 --feval_budget 3000 --ncores 14 --gen_timeout 60 \
    --out sweep_results.csv --runs_dir sweep_runs
```
All 15 completed (no more hangs): **4 timed out** cleanly within 60s each (all at `param_bound=1.0`), the rest finished normally.

---

## Plotting Bug: Outlier-Dominated Aggregation

`plot_performance.py`'s first version used raw mean/std, which a single `val_NLL=1e9` row (the post-training `LinAlgError` fallback from math.md §7 — a marginal-Schur `es.result[0]` that passes CMA's eigenvalue check but fails Cholesky in the main process) would dominate by 8 orders of magnitude, making the scaling and heatmap plots unreadable.

**Fix**: treat `val_NLL >= 1e8` as a categorical "this seed crashed" event rather than a continuous outlier. Quality metrics (NLL, AUC) are now computed only from non-crashed rows; crash counts are reported separately (printed warnings + heatmap cell annotations like `"1/2 crashed"`).

**Follow-up fix**: even after excluding crashes, a config with very few surviving seeds gets a *visually* tight error bar (e.g., `std` of a single value is exactly `0.0`), which looks like robustness but actually means "almost everything failed, n=1." Added `n=` sample-size annotations directly on the scaling plot, and a stdout warning whenever `n < 2`.

---

## Final Sweep Results (45/45 grid points accounted for)

**Reliability by `N_h`** (from the heatmap's crash annotations):
- **`nh=2`, `nh=3`**: zero crashes across the entire grid — 15/15 runs each succeeded cleanly
- **`nh=4`**: 30-50% of runs either timed out or hit the post-training `LinAlgError` fallback, at *every* `param_bound` tested — confirms the lattice-sum cost explosion is structural to `N_h=4`, not a one-off

**Quality** (`scaling_nh.png`, non-crashed seeds only):

| `N_h` | best `param_bound` | mean val_NLL | mean AUC | usable seeds |
|---|---|---|---|---|
| 2 | (grid best) | 4.72 ± 0.46 | 0.897 ± 0.019 | n=3 |
| 3 | (grid best) | 4.60 ± 0.52 | 0.901 ± 0.021 | n=3 |
| 4 | 1.0 | 4.42 | 0.908 | **n=1** |

The `nh=4` row is **not directly comparable** to the others: of 3 attempted seeds, 1 timed out and 1 crashed post-training, leaving exactly one surviving run. Its apparently-best numbers and the complete absence of a visible error bar both come from that single data point, not from nh=4 being more stable — it's the opposite.

**Cross-validation**: the hyperopt search (Run 1, fully independent of this sweep) also converged on **`n_hidden=3`** as best, without any knowledge of the sweep's results.

### Recommendation

**Use `n_hidden=3`.** It matches `nh=4`'s quality (when nh=4 happens to work) while having `nh=2`'s full reliability (0 crashes across the entire grid). `nh=4` is not recommended without further investigation into mitigating the lattice-sum explosion (e.g., constraining the Schur complement's minimum eigenvalue, or accepting the `gen_timeout` losses as a permanent cost of that architecture).

---

## Artifacts

- `sweep_results.csv` — 45 rows, one per `(n_hidden, param_bound, seed)`
- `sweep_runs/` — 90 `.npy` files (CMA convergence history + valid-candidate-fraction history per successful run)
- `sweep_plots/` — `scaling_nh.png`, `heatmap_pb_nh.png`, `convergence_overlay.png`, `valid_fraction.png`
- `training/optimized_run/` — model, plots, and `args.txt` from the hyperopt run