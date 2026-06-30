# CLAUDE.md (rtbm_workspace)

Directory-specific guidance for the RTBM half of the project. See the root `CLAUDE.md` for the overall project, data pipeline, and the autoencoder.

## File Layout

- **`rtbmlib.py`** — shared library: model construction (`make_rtbm`), the CMA-ES training loop (`train_rtbm`), evaluation helpers (`mean_nll`, `anomaly_scores`, `compute_auc`, `background_rejection`), plus `default_popsize`/`run_tag` used by the sweep tooling. `training.py` and the performance scripts both import from here — don't duplicate these functions elsewhere.
- **`training.py`** — production CLI: single training run or hyperopt search, full plot suite (`history.png`, `density_check.png`, `anomaly_scores.png`, `roc_curve.png`). Imports `rtbmlib`.
- **`mytraining.py`** — a personal rewrite kept separate for learning. Do not refactor it to use `rtbmlib.py` or otherwise "clean it up" unless explicitly asked — it's intentionally a standalone exercise, with its own `args.txt` convention (records both launch args and hyperopt-found best params).
- **`sweep.py`** — grid + seed sweep over `(n_hidden, param_bound)` for performance studies. Writes a structured CSV (`sweep_results.csv`) and per-run `.npy` diagnostics, independent of either training script's `args.txt`.
- **`plot_performance.py`** — turns `sweep.py`'s CSV into the scaling/heatmap/convergence/valid-fraction plot catalogue described in `performance.md`.
- **`math.md`** — derivation of every numerical bug encountered and its fix: Schur complement collapse at init, `tolfun`/`tolflatfitness` CMA stopping criteria, logit preprocessing, full-T vs diagonal-T, CMA bounds widening for full T. Read the relevant section before changing anything in `make_rtbm` or `train_rtbm`.
- **`performance.md`** — methodology for performance studies: why function evaluations (not iterations) is the fair comparison unit across `N_hidden`, why `NAN_PENALTY` must stay constant for the full trial duration, how seeds are separated from the data split.

## Key Facts Not Obvious From the Code

- **`diagonal_T=False`, not `True`.** `make_rtbm` uses a full T matrix with W manually zeroed post-init (not the theta library's `diagonal_T=True` flag), so the Gaussian envelope can capture correlations between features (e.g. `f_had` vs `Iso`, both probing the same missing π⁰). `diagonal_T=True` was tried first and rejected — see math.md Section 3.
- **CMA-ES never discards invalid candidates.** An earlier version spun in a `while len(solutions) < popsize` loop re-asking until enough valid candidates appeared; this could hang indefinitely. The current `train_rtbm` always evaluates a full population every generation and substitutes `NAN_PENALTY = 1e9` for any non-finite cost (Schur/positivity failures), then tells CMA-ES the full population.
- **`tolfun=0`, not the cma default.** With `NAN_PENALTY`, an all-invalid generation has zero within-population fitness range, which triggers CMA's default `tolfun` stopping criterion after a single iteration — even though the run hasn't actually converged. `tolfun=0` disables that check; convergence is instead detected via `tolflatfitness=maxiter` (stop only after `maxiter` consecutive generations with no improvement in the *best* fitness). See math.md Section 6.
- **CMA bounds are computed from the actual initial parameters, not from `param_bound` directly.** With full T, the Schur-init diagonal entries are sums of squares and routinely exceed `param_bound`; `make_rtbm` widens the bounds to `max(param_bound, max_abs_init) * 1.2` after construction. See math.md Section 9.3 for why this doesn't reintroduce the Schur-collapse problem it's meant to avoid.
- **`x_vis` gets a logit transform; `eta` and `f_had` do not.** `x_vis` is nearly flat on `[0,1]` (logit → logistic distribution, Gaussian-like). `eta`'s distribution piles up at one detector-acceptance boundary (logit would create a hard spike at -9.2 there). `f_had` is bimodal at exactly 0 and 1 (logit scatters those spikes further apart, making it worse). See math.md Section 8.
- **`make_rtbm` raises `RuntimeError`** (not a silent bad return) if no valid initialisation is found within `max_tries` attempts. Callers (including `sweep.py`) must catch this.
- **Performance sweeps fix the data split, vary only the model/CMA seed.** `sweep.py` shuffles/splits data once under `--data_seed` (default 42) and reuses it for every `(n_hidden, param_bound, seed)` combination; only `np.random.seed` (for `random_init`) and `cma_opts['seed']` vary with `--seeds`. This isolates optimisation variance from which events ended up in train/val/test.

## Common Commands

```bash
cd rtbm_workspace

# single training run
python training.py --save -nh 2 --maxiter 300

# hyperopt search over n_hidden and param_bound
python training.py --save --optimize --max_evals 20

# performance sweep (grid x seeds), then plot the results
python sweep.py --nh 2,3,4 --pb 1,2,3,5,8 --seeds 1,2,3,4,5 --feval_budget 3000 --ncores 4
python plot_performance.py
```
