"""Simplified sweep using the original theta CMA training function.

No fixed seeds — each run lets CMA use its default (time-based) seed,
sampling the natural run-to-run variance of the optimizer.

Run from rtbm_workspace/:
    .venv/bin/python3 sweep_simple.py [--nh 2 3] [--pb 3.0 4.3 6.0]
                                      [--n_runs 5] [--n_train 10000]
                                      [--maxiter 300] [--out sweep_simple.csv]
"""
import os
import sys
import time
import csv
import argparse
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost
from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    make_rtbm, mean_nll, anomaly_scores, compute_auc,
)


def train_once(model, X_tr, ncores, maxiter):
    CMA(parallel=ncores > 1, ncores=ncores).train(
        log_nll_cost, model, X_tr, maxiter=maxiter)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nh',      type=int,   nargs='+', default=[1,2,3])
    p.add_argument('--pb',      type=int,   nargs=2,   default=[11, 15],
                   metavar=('MIN', 'MAX'), help='integer param_bound range (inclusive)')
    p.add_argument('--n_runs',  type=int,   default=5)
    p.add_argument('--n_train', type=int,   default=10000)
    p.add_argument('--maxiter', type=int,   default=250)
    p.add_argument('--ncores',  type=int,   default=min(PARALLEL_CORES, mp.cpu_count()))
    p.add_argument('--out',     default='sweep_simple.csv')
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

    fieldnames = ['nh', 'param_bound', 'run', 'val_nll', 'auc', 'time_sec', 'status']
    file_exists = os.path.exists(args.out)
    out = open(args.out, 'a', newline='')
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    # load already completed (nh, param_bound, run) triples to skip them
    done = set()
    if file_exists:
        import pandas as pd
        df_done = pd.read_csv(args.out)
        done = set(zip(df_done['nh'], df_done['param_bound'], df_done['run']))
        print(f"[INFO] Skipping {len(done)} already completed runs")

    pb_values = list(range(args.pb[0], args.pb[1] + 1))
    print(f"[INFO] param_bound values: {pb_values}")

    for nh in args.nh:
        for pb in pb_values:
            for run in range(1, args.n_runs + 1):
                if (nh, pb, run) in done:
                    print(f"[SKIP] nh={nh}  pb={pb}  run={run}/{args.n_runs}")
                    continue
                tag = f"nh={nh}  pb={pb}  run={run}/{args.n_runs}"
                print(f"\n[START] {tag}")
                row = dict(nh=nh, param_bound=pb, run=run,
                           val_nll=None, auc=None, time_sec=None, status='failed')
                try:
                    model = make_rtbm(N_VISIBLE, nh, pb)
                    t0 = time.perf_counter()
                    train_once(model, X_tr, args.ncores, args.maxiter)
                    elapsed = time.perf_counter() - t0

                    nll = mean_nll(model, X_val)
                    auc = compute_auc(anomaly_scores(model, X_val),
                                      anomaly_scores(model, X_rho))
                    row.update(val_nll=round(nll, 4), auc=round(auc, 4),
                               time_sec=round(elapsed, 1), status='ok')
                    print(f"[OK]   {tag}  NLL={nll:.3f}  AUC={auc:.4f}  t={elapsed:.0f}s")
                except Exception as e:
                    print(f"[FAIL] {tag}: {e}")

                writer.writerow(row)
                out.flush()

    out.close()
    print(f"\n[DONE] '{args.out}'")


if __name__ == '__main__':
    main()