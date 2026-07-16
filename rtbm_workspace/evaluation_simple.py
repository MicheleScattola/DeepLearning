"""Evaluate a saved RTBM model: density check, anomaly scores, ROC curve.

Usage:
    python evaluation_simple.py simple_training
    python evaluation_simple.py --folder simple_training
"""
import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from rtbmlib import (
    N_VISIBLE,
    load_datasets, train_val_test_split,
    make_rtbm, anomaly_scores, compute_auc, compute_roc, background_rejection,
)


def load_model_and_data(folder):
    pkl_path = os.path.join('training', folder, 'model.pkl')
    with open(pkl_path, 'rb') as fh:
        payload = pickle.load(fh)

    mu      = payload['mu']
    std     = payload['std']
    n_train = payload.get('n_train', 20000)

    model = make_rtbm(N_VISIBLE, payload['n_hidden'], payload['param_bound'])
    model.set_parameters(np.real(np.array(payload['params'])))

    np.random.seed(42)
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)
    _, val_pi, test_pi = train_val_test_split(pi_data, n_train)

    X_val  = ((val_pi  - mu) / std).T
    X_test = ((test_pi - mu) / std).T
    X_rho  = ((rho_data - mu) / std).T

    return model, val_pi, X_val, X_test, X_rho


def plot_density_check(model, X_val, val_pi, outdir):
    try:
        log_probs = np.real(model(X_val)).flatten()
    except np.linalg.LinAlgError:
        log_probs = np.zeros(X_val.shape[1])
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    w = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)

    labels = [r'$x_{vis}$',
              r'$\mathrm{Iso} = \Sigma E_{ph}/E_{track}$',
              r'$f_{had} = E_{HCAL}/E_{tot}$',
              r'$\eta$']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(r'RTBM: Density Check on $\tau\to\pi\nu$ (Validation)', fontweight='bold')

    for i, ax in enumerate(axes.flat):
        feat = val_pi[:, i]
        bins = np.linspace(feat.min(), feat.max(), 40)
        ax.hist(feat, bins=bins, histtype='step', linewidth=1.5,
                density=True, label='Validation data')
        ax.hist(feat, bins=bins, weights=w * len(feat),
                histtype='step', linewidth=1.5, linestyle='--',
                density=True, label=r'RTBM $P(x)$ weighted')
        ax.set_title(labels[i])
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(outdir, 'density_check.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] {path}")


def plot_anomaly_scores(scores_pi, scores_rho, outdir):
    threshold_95  = np.percentile(scores_pi, 95)
    bkg_rejection = background_rejection(scores_pi, scores_rho, target_eff=0.95)

    print("\n=======================================================")
    print(f"Target Signal Efficiency : 95.00%")
    print(f"Optimal NLL Threshold    : {threshold_95:.4f}")
    print(f"Background Rejection     : {bkg_rejection * 100:.2f}%")
    print("=======================================================\n")

    all_scores = np.concatenate([scores_pi, scores_rho])
    lo   = np.percentile(all_scores, 0.5)
    bins = np.linspace(lo, 20, 100)

    w_pi  = np.ones_like(scores_pi)  / len(scores_pi)
    w_rho = np.ones_like(scores_rho) / len(scores_rho)

    plt.figure(figsize=(8, 6))
    plt.hist(scores_pi,  bins=bins, weights=w_pi,  histtype='step',
             linewidth=2, label=r'Signal: $\tau\to\pi\nu$',      color='blue')
    plt.hist(scores_rho, bins=bins, weights=w_rho, histtype='step',
             linewidth=2, label=r'Background: $\tau\to\rho\nu$', color='red')
    plt.hist(scores_pi[scores_pi <= threshold_95], bins=bins,
             weights=w_pi[scores_pi <= threshold_95],
             histtype='stepfilled', alpha=0.2, color='blue')
    plt.hist(scores_rho[scores_rho >= threshold_95], bins=bins,
             weights=w_rho[scores_rho >= threshold_95],
             histtype='stepfilled', alpha=0.2, color='red')
    plt.axvline(threshold_95, color='black', linestyle='--', linewidth=1.5,
                label=r'Threshold: 95% eff. in $\pi$ data')
    plt.title(r'RTBM: Anomaly Scores ($-\log P(x)$)', fontweight='bold')
    plt.xlabel(r'$-\log P(x)$')
    plt.xlim([0.0, 20])
    plt.ylabel('Fraction of Events')
    plt.legend(frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    path = os.path.join(outdir, 'anomaly_scores.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] {path}")


def plot_roc(scores_pi, scores_rho, outdir):
    fpr, tpr, roc_auc = compute_roc(scores_pi, scores_rho)

    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr, tpr, color='darkorange', alpha=0.2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'RTBM AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [1, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (Signal Loss)')
    plt.ylabel('TPR (Background Rejection)')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc='center', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    path = os.path.join(outdir, 'roc_curve.pdf')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] {path}")
    print(f"\n[RESULT] RTBM AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder', help='Subfolder inside training/ (e.g. simple_training)')
    args = p.parse_args()

    outdir   = os.path.join('training', args.folder)
    pkl_path = os.path.join(outdir, 'model.pkl')
    if not os.path.isdir(outdir):
        p.error(f"Training folder not found: {outdir}")
    if not os.path.isfile(pkl_path):
        p.error(f"No model.pkl in '{outdir}' — run: python training_simple.py --save -o {args.folder}")

    print(f"[INFO] Loading model from '{pkl_path}'...")
    model, val_pi, X_val, X_test, X_rho = load_model_and_data(args.folder)

    print("[INFO] Computing anomaly scores...")
    sc_pi  = anomaly_scores(model, X_test)
    sc_rho = anomaly_scores(model, X_rho)

    plot_density_check(model, X_val, val_pi, outdir)
    plot_anomaly_scores(sc_pi, sc_rho, outdir)
    plot_roc(sc_pi, sc_rho, outdir)