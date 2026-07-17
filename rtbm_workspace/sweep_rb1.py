"""Sweep over n_hidden and param_bound with fixed random_bound=1.

Uses a fixed random_bound=1 (theta default, natural scale for z-scored data).
sigma_schur is fixed at 0.1 (just positivity check) instead of pb*0.1 —
with rb=1 E[T_ii]~2, so the pb-scaled threshold would fail for pb>20.

Run from rtbm_workspace/:
    .venv/bin/python3 sweep_rb1.py
    .venv/bin/python3 sweep_rb1.py --nh 2 3 --pb 1 5 10 15 20 25 30 --n_runs 5
"""
import os
import sys
import time
import csv
import argparse
import threading
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.rtbm import RTBM
from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    mean_nll, anomaly_scores, compute_auc,
)

RANDOM_BOUND = 1.0


def make_rtbm_rb1(nv, nh, param_bound):
    """Init RTBM with random_bound=1 and W=0.

    No positivity retry loop — theta's CMA assigns NAN_PENALTY to invalid
    candidates and moves away naturally, so the starting point doesn't need
    to be valid.
    """
    m = RTBM(nv, nh, init_max_param_bound=param_bound,
             random_bound=RANDOM_BOUND,
             diagonal_T=False, mode=RTBM.Mode.LogProbability)
    params = np.real(m.get_parameters()).copy()
    params[nv + nh : nv + nh + nv * nh] = 0.0  # W=0
    m.set_parameters(params)  # may return False if invalid — CMA handles it
    # Widen box if init params exceed param_bound (can happen when rb >= pb)
    actual_max = float(np.max(np.abs(params)))
    m.set_bounds(max(param_bound, actual_max) * 1.2)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nh',      type=int,   nargs='+', default=[1,2,3])
    p.add_argument('--pb',      type=float, nargs='+', default=[20,25])
    p.add_argument('--n_runs',  type=int,   default=5)
    p.add_argument('--n_train', type=int,   default=10000)
    p.add_argument('--maxiter', type=int,   default=250)
    p.add_argument('--ncores',  type=int,   default=min(PARALLEL_CORES, mp.cpu_count()))
    p.add_argument('--out',     default='sweep_rb1.csv')
    args = p.parse_args()

    print("[INFO] Loading datasets...")
    pi, rho = load_datasets()
    np.random.seed(42)
    np.random.shuffle(pi)

    n_tr = int(0.8 * args.n_train)
    tr_std, [val_std, rho_std], _ = standardize(pi[:n_tr], pi[n_tr:args.n_train], rho)
    X_tr  = tr_std.T
    X_val = val_std.T
    X_rho = rho_std.T
    print(f"[INFO] train={X_tr.shape[1]}  val={X_val.shape[1]}  rho={X_rho.shape[1]}")
    print(f"[INFO] random_bound={RANDOM_BOUND}")

    fieldnames = ['nh', 'param_bound', 'run', 'val_nll', 'auc', 'time_sec', 'status']
    file_exists = os.path.exists(args.out)
    out = open(args.out, 'a', newline='')
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    done = set()
    if file_exists:
        import pandas as pd
        df_done = pd.read_csv(args.out)
        done = set(zip(df_done['nh'], df_done['param_bound'], df_done['run']))
        print(f"[INFO] Skipping {len(done)} already completed runs")

    for nh in args.nh:
        for pb in args.pb:
            for run in range(1, args.n_runs + 1):
                if (nh, pb, run) in done:
                    print(f"[SKIP] nh={nh}  pb={pb}  run={run}")
                    continue

                tag = f"nh={nh}  pb={pb}  run={run}/{args.n_runs}"
                print(f"\n[START] {tag}")
                row = dict(nh=nh, param_bound=pb, run=run,
                           val_nll=None, auc=None,
                           time_sec=None, status='failed')
                try:
                    model = make_rtbm_rb1(N_VISIBLE, nh, pb)

                    t0 = time.perf_counter()
                    _result = {}
                    def _train():
                        try:
                            CMA(parallel=args.ncores > 1, ncores=args.ncores).train(
                                log_nll_cost, model, X_tr, maxiter=args.maxiter, tolfun=1e-6)
                            _result['ok'] = True
                        except Exception as e:
                            _result['error'] = e
                    _t = threading.Thread(target=_train, daemon=True)
                    _t.start()
                    _t.join(timeout=900)
                    if _t.is_alive():
                        raise TimeoutError("CMA timed out after 900s")
                    if 'error' in _result:
                        raise _result['error']
                    elapsed = time.perf_counter() - t0

                    nll = mean_nll(model, X_val)
                    auc_val = compute_auc(anomaly_scores(model, X_val),
                                          anomaly_scores(model, X_rho))
                    row.update(val_nll=round(nll, 4), auc=round(auc_val, 4),
                               time_sec=round(elapsed, 1), status='ok')
                    print(f"[OK]   {tag}  NLL={nll:.3f}  AUC={auc_val:.4f}  t={elapsed:.0f}s")

                except Exception as e:
                    print(f"[FAIL] {tag}: {e}")

                writer.writerow(row)
                out.flush()

    out.close()
    print(f"\n[DONE] '{args.out}'")
    _print_summary(args.out)


def _print_summary(path):
    import pandas as pd
    df = pd.read_csv(path)
    ok = df[df['status'] == 'ok']
    if ok.empty:
        print("No successful runs.")
        return
    summary = ok.groupby(['nh', 'param_bound']).agg(
        mean_nll=('val_nll', 'mean'),
        std_nll=('val_nll', 'std'),
        mean_auc=('auc', 'mean'),
        n=('val_nll', 'count'),
    ).round(4)
    print("\n" + "=" * 70)
    print(summary.to_string())
    print("=" * 70)


if __name__ == '__main__':
    main()