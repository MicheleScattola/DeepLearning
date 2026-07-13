"""K-fold cross-validation for RTBM AUC estimation.

Uses the best hyperparameters found by training.py --optimize.
"""
import os
import sys
import numpy as np
import multiprocessing as mp
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize,
    make_rtbm, train_rtbm, anomaly_scores, compute_auc,
)

# hyperparameters -- TO BE UPDATED ON NEW RUN
K           = 10
N_HIDDEN    = 3
PARAM_BOUND = 4.3
N_TRAIN     = 30000   # CMA events for each fold
MAXITER     = 300


if __name__ == '__main__':
    np.random.seed(42)
    ncores = min(PARALLEL_CORES, mp.cpu_count())

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    kf   = KFold(n_splits=K, shuffle=False)
    aucs = []

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
        fold_auc = compute_auc(sc_pi, sc_rho)
        aucs.append(fold_auc)
        print(f"[FOLD {fold}/{K}]  AUC = {fold_auc:.4f}")

    if not aucs:
        print("\n[ERROR] All folds failed.")
    else:
        aucs = np.array(aucs)
        lines = [
        "\n=======================================================",
        f"K-Fold Cross-Validation (k={K})",
        f"AUC per fold : {' '.join(f'{a:.4f}' for a in aucs)}",
        f"Mean AUC     : {aucs.mean():.4f}",
        f"Std AUC      : {aucs.std():.4f}",
        "=======================================================",
        ]
        for line in lines:
            print(line)
        with open("auc.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        print("[INFO] Saved results to 'auc_nh3s.txt'")