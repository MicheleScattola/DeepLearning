"""Test whether random_bound / init_max_param_bound scaling affects density fit quality.

Hypothesis: our random_bound=sqrt(param_bound) initializes T_ii >> 1 for standardized
data, which may hurt convergence. The theta default (random_bound=1) starts T_ii near
its natural scale (~1-3) and might give better fits.

Run from rtbm_workspace/:
    .venv/bin/python3 test_init_scaling.py
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.rtbm import RTBM
from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    mean_nll, anomaly_scores, compute_auc,
)


def make_rtbm_custom(nv, nh, random_bound, init_max_param_bound,
                     sigma_schur, max_tries=500):
    """Like make_rtbm but with independently controllable random_bound and init_max_param_bound."""
    tries = 0
    for _ in range(max_tries):
        tries += 1
        m = RTBM(nv, nh, init_max_param_bound=init_max_param_bound,
                 random_bound=random_bound,
                 diagonal_T=False, mode=RTBM.Mode.LogProbability)
        if np.all(np.diag(m.t) > sigma_schur) and np.all(np.diag(m.q) > sigma_schur):
            params = np.real(m.get_parameters()).copy()
            params[nv + nh : nv + nh + nv * nh] = 0.0  # W=0 at init
            if m.set_parameters(params):
                actual_max = float(np.max(np.abs(params)))
                m.set_bounds(max(init_max_param_bound, actual_max) * 1.2)
                t_ii_init = float(np.mean(np.diag(m.t)))
                return m, tries, t_ii_init
    return None, tries, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nh',      type=int,   default=2)
    p.add_argument('--n_train', type=int,   default=20000)
    p.add_argument('--maxiter', type=int,   default=250)
    p.add_argument('--n_runs',  type=int,   default=3)
    p.add_argument('--ncores',  type=int,   default=PARALLEL_CORES)
    p.add_argument('--out',     default='test_init_scaling.csv')
    args = p.parse_args()

    configs = [
        # small scale
        (1.0,  2.0,  'rb=1  pb=2   (theta default)'),
        (1.0,  3.0,  'rb=1  pb=3'),
        (1.0,  5.0,  'rb=1  pb=5'),
        # medium
        (2.0,  5.0,  'rb=2  pb=5'),
        (2.0,  8.0,  'rb=2  pb=8'),
        # current best
        (np.sqrt(5.0),   5.0,  'rb=sqrt(pb) pb=5   (current scheme)'),
        (np.sqrt(11.0), 11.0,  'rb=sqrt(pb) pb=11  (current scheme, best sweep)'),
        # decoupled
        (1.0, 11.0,  'rb=1  pb=11  (small init, wide box)'),
        (1.0, 20.0,  'rb=1  pb=20  (theta example style)'),
    ]

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.seed(42)
    np.random.shuffle(pi_data)

    n_tr = int(0.8 * args.n_train)
    tr_std, [val_std, rho_std], _ = standardize(pi_data[:n_tr],
                                                  pi_data[n_tr:args.n_train],
                                                  rho_data)
    X_tr  = tr_std.T
    X_val = val_std.T
    X_rho = rho_std.T
    print(f"[INFO] train={X_tr.shape[1]}  val={X_val.shape[1]}  rho={X_rho.shape[1]}")

    fieldnames = ['label', 'random_bound', 'init_max_pb', 'run',
                  'init_tries', 't_ii_init', 't_ii_final',
                  'val_nll', 'auc', 'time_sec', 'status']

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rb, pb, label in configs:
            sigma_schur = pb * 0.1
            for run in range(1, args.n_runs + 1):
                print(f"\n[RUN] {label}  run={run}/{args.n_runs}")
                row = dict(label=label, random_bound=round(rb, 4),
                           init_max_pb=pb, run=run,
                           init_tries=None, t_ii_init=None, t_ii_final=None,
                           val_nll=None, auc=None, time_sec=None, status='failed')
                try:
                    model, tries, t_ii_init = make_rtbm_custom(
                        N_VISIBLE, args.nh, rb, pb, sigma_schur)
                    row['init_tries'] = tries
                    row['t_ii_init']  = round(t_ii_init, 4) if t_ii_init else None

                    if model is None:
                        print(f"  [FAIL] Could not find valid init in {tries} tries")
                        writer.writerow(row); f.flush()
                        continue

                    print(f"  init ok: tries={tries}  T_ii_init={t_ii_init:.3f}")

                    t0 = time.perf_counter()
                    CMA(parallel=args.ncores > 1, ncores=args.ncores).train(
                        log_nll_cost, model, X_tr, maxiter=args.maxiter)
                    elapsed = time.perf_counter() - t0

                    t_ii_final = float(np.mean(np.diag(model.t)))
                    nll = mean_nll(model, X_val)
                    auc = compute_auc(anomaly_scores(model, X_val),
                                      anomaly_scores(model, X_rho))

                    row.update(t_ii_final=round(t_ii_final, 4),
                               val_nll=round(nll, 4), auc=round(auc, 4),
                               time_sec=round(elapsed, 1), status='ok')
                    print(f"  T_ii_final={t_ii_final:.3f}  NLL={nll:.4f}  AUC={auc:.4f}  t={elapsed:.0f}s")

                except Exception as e:
                    print(f"  [FAIL] {e}")

                writer.writerow(row)
                f.flush()

    print(f"\n[DONE] Results saved to '{args.out}'")
    _print_summary(args.out)


def _print_summary(path):
    import pandas as pd
    df = pd.read_csv(path)
    ok = df[df['status'] == 'ok']
    if ok.empty:
        print("No successful runs.")
        return
    summary = ok.groupby('label').agg(
        mean_nll=('val_nll', 'mean'),
        std_nll=('val_nll', 'std'),
        mean_auc=('auc', 'mean'),
        mean_T_init=('t_ii_init', 'mean'),
        mean_T_final=('t_ii_final', 'mean'),
        mean_tries=('init_tries', 'mean'),
    ).round(4)
    print("\n" + "="*80)
    print(summary.to_string())
    print("="*80)


if __name__ == '__main__':
    main()