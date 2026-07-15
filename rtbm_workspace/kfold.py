"""K-fold cross-validation for RTBM AUC estimation.

Uses the best hyperparameters found by training.py --optimize.
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

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    make_rtbm, train_rtbm, anomaly_scores, compute_roc,
)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-nh', '--n_hidden',    type=int,   default=3)
    p.add_argument('--param_bound',        type=float, default=4.3)
    p.add_argument('-k',                   type=int,   default=10)
    p.add_argument('--n_train',            type=int,   default=30000)
    p.add_argument('--maxiter',            type=int,   default=300)
    p.add_argument('--out',                default=None,
                   help='Output file (default: auc_nh<N>.txt)')
    args = p.parse_args()

    N_HIDDEN    = args.n_hidden
    PARAM_BOUND = args.param_bound
    K           = args.k
    N_TRAIN     = args.n_train
    MAXITER     = args.maxiter
    outfile     = args.out or f'auc_nh{N_HIDDEN}.txt'

    np.random.seed(42)
    ncores = min(PARALLEL_CORES, mp.cpu_count())

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    kf     = KFold(n_splits=K, shuffle=False)
    aucs   = []
    curves = []   # (fpr, tpr) per fold

    for fold, (train_idx, test_idx) in enumerate(kf.split(pi_data), start=1):
        print(f"\n[FOLD {fold}/{K}]  pool={len(train_idx)}  test={len(test_idx)}")

        train_pool = pi_data[train_idx]
        test_fold  = pi_data[test_idx]

        # Split N_TRAIN events 80/20 for CMA-ES train/val
        n_cma_train = int(0.8 * N_TRAIN)
        train_cma   = train_pool[:n_cma_train]
        val_cma     = train_pool[n_cma_train:N_TRAIN]

        tr_std, [val_std, test_std, rho_std], _ = standardize(
            train_cma, val_cma, test_fold, rho_data
        )
        X_tr   = tr_std.T
        X_test = test_std.T
        X_rho  = rho_std.T

        print(f"[FOLD {fold}/{K}]  CMA train={len(train_cma)}  CMA val={len(val_cma)}")

        try:
            model = make_rtbm(N_VISIBLE, N_HIDDEN, PARAM_BOUND)
            train_rtbm(model, X_tr, ncores=ncores, maxiter=MAXITER,
                       tolfun=0.0, init_sigma=PARAM_BOUND * 0.1)
        except RuntimeError as e:
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
        with open(outfile, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[INFO] Saved results to '{outfile}'")

        # ROC overlay plot
        fpr_grid = np.linspace(0, 1, 300)
        tprs_interp = np.array([np.interp(fpr_grid, fpr, tpr) for fpr, tpr in curves])
        mean_tpr = tprs_interp.mean(axis=0)

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
        roc_file = outfile.replace('.txt', '_roc.png')
        plt.savefig(roc_file, dpi=300)
        plt.close()
        print(f"[PLOT] Saved '{roc_file}'")