"""K-fold cross-validation for RTBM AUC estimation using the theta CMA.train() directly.

Run from rtbm_workspace/:
    .venv/bin/python3 kfold_simple.py [-nh 3] [--param_bound 4.3] [-k 10]
"""
import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    make_rtbm, anomaly_scores, compute_roc,
)


def train_once(model, X_tr, ncores, maxiter):
    CMA(parallel=ncores > 1, ncores=ncores).train(
        log_nll_cost, model, X_tr, maxiter=maxiter)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-nh', '--n_hidden',  type=int,   default=2)
    p.add_argument('--param_bound',      type=float, default=11.3160)
    p.add_argument('-k',                 type=int,   default=5)
    p.add_argument('--maxiter',          type=int,   default=300)
    p.add_argument('--ncores',           type=int,   default=PARALLEL_CORES)
    p.add_argument('--out',              default=None,
                   help='Output file (default: auc_simple_nh<N>.txt)')
    args = p.parse_args()

    N_HIDDEN    = args.n_hidden
    PARAM_BOUND = args.param_bound
    K           = args.k
    outfile     = args.out or f'auc_simple_nh{N_HIDDEN}.txt'

    np.random.seed(42)
    ncores = min(args.ncores, mp.cpu_count())

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    kf     = KFold(n_splits=K, shuffle=False)
    aucs   = []
    curves = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(pi_data), start=1):
        print(f"\n[FOLD {fold}/{K}]  pool={len(train_idx)}  test={len(test_idx)}")

        train_pool = pi_data[train_idx]
        test_fold  = pi_data[test_idx]

        tr_std, [test_std, rho_std], _ = standardize(
            train_pool, test_fold, rho_data)
        X_tr   = tr_std.T
        X_test = test_std.T
        X_rho  = rho_std.T

        print(f"[FOLD {fold}/{K}]  CMA train={len(train_pool)}")

        try:
            model = make_rtbm(N_VISIBLE, N_HIDDEN, PARAM_BOUND)
            train_once(model, X_tr, ncores, args.maxiter)
        except Exception as e:
            print(f"[FOLD {fold}/{K}]  FAILED: {e} — skipping fold")
            continue

        sc_pi  = anomaly_scores(model, X_test)
        sc_rho = anomaly_scores(model, X_rho)
        fpr, tpr, fold_auc = compute_roc(sc_pi, sc_rho)
        aucs.append(fold_auc)
        curves.append((fpr, tpr))
        print(f"[FOLD {fold}/{K}]  AUC = {fold_auc:.4f}")

    if not aucs:
        print("\n[ERROR] All folds failed.")
    else:
        aucs = np.array(aucs)
        lines = [
            "\n=======================================================",
            f"K-Fold Cross-Validation (k={K})",
            f"N_hidden     : {N_HIDDEN}",
            f"param_bound  : {PARAM_BOUND}",
            f"AUC per fold : {' '.join(f'{a:.4f}' for a in aucs)}",
            f"Mean AUC     : {aucs.mean():.4f}",
            f"Std AUC      : {aucs.std():.4f}",
            "=======================================================",
        ]
        for line in lines:
            print(line)
        with open(outfile, 'w') as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] Saved results to '{outfile}'")

        fpr_grid    = np.linspace(0, 1, 300)
        tprs_interp = np.array([np.interp(fpr_grid, fpr, tpr) for fpr, tpr in curves])
        mean_tpr    = tprs_interp.mean(axis=0)

        _, ax = plt.subplots(figsize=(6, 6))
        for i, (fpr, tpr) in enumerate(curves):
            ax.plot(fpr, tpr, color='steelblue', alpha=0.3, lw=1,
                    label='Folds' if i == 0 else None)
        ax.plot(fpr_grid, mean_tpr, color='steelblue', lw=2,
                label=f'Mean AUC = {aucs.mean():.3f} ± {aucs.std():.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'RTBM ROC — $N_h$={N_HIDDEN}, {K}-fold', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_file = outfile.replace('.txt', '_roc.pdf')
        plt.savefig(roc_file, dpi=300)
        plt.close()
        print(f"[PLOT] Saved '{roc_file}'")