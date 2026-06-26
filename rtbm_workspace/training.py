import os
import sys
import argparse
import pickle
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import multiprocessing as mp
from contextlib import closing
from cma import CMAEvolutionStrategy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))
from theta.rtbm import RTBM
from theta.minimizer import worker_initialize, worker_compute
from theta.costfunctions import sum as log_nll_cost

# ── constants ─────────────────────────────────────────────────────────────────
ETA_MAX   = 2.5
N_VISIBLE = 4
PHYS_CORES = mp.cpu_count()
PARALLEL_CORES = int(PHYS_CORES / 2)

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(prog='RTBMTraining',
                                 description='RTBM training and hyperparameter search')
parser.add_argument('-o',  '--outfile',     default='trained_rtbm')
parser.add_argument('-nh', '--n_hidden',    type=int,   default=2)
parser.add_argument('--maxiter',            type=int,   default=200)
parser.add_argument('--tolfun',             type=float, default=1e-5)
parser.add_argument('--ncores',             type=int,   default=PARALLEL_CORES)
parser.add_argument('--param_bound',        type=float, default=5.0)
parser.add_argument('--n_train',            type=int,   default=20000,
                    help='Number of pi events used for CMA-ES training (80/20 train/val split)')
parser.add_argument('--optimize',           action='store_true',
                    help='Run Bayesian hyperparameter search before final training')
parser.add_argument('--max_evals',          type=int,   default=15,
                    help='Number of hyperopt evaluations')
parser.add_argument('--save',               action='store_true',
                    help='Pickle the trained model parameters')
args = parser.parse_args()

OUTDIR = os.path.join('training', args.outfile)
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, 'args.txt'), 'w') as _f:
    for _k, _v in vars(args).items():
        _f.write(f"{_k}: {_v}\n")

np.random.seed(42)


# ── data ───────────────────────────────────────────────────────────────────────
def load_datasets():
    pi  = np.load('../datasets/pi.npy')
    rho = np.load('../datasets/rho.npy')
    pi[:,  1] = np.clip(pi[:,  1], 0.0, 1.0)
    rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
    pi[:,  3] = (pi[:,  3] - ETA_MAX) / 5.0
    rho[:, 3] = (rho[:, 3] - ETA_MAX) / 5.0
    eps = 1e-4
    for arr in (pi, rho):
        arr[:, 0] = np.log(np.clip(arr[:, 0], eps, 1-eps) / (1 - np.clip(arr[:, 0], eps, 1-eps)))
    return pi, rho


def standardize(train, *others):
    mu  = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mu) / std, [(x - mu) / std for x in others], (mu, std)


def make_rtbm(nv, nh, param_bound):
    """Create RTBM with full (non-diagonal) T and W=0 at initialisation.

    diagonal_T=True was a numerical workaround that permanently constrains T
    to be diagonal, preventing the Gaussian envelope from capturing feature
    correlations (e.g. f_had vs Iso, both probing the same missing pi0).
    Instead, we use diagonal_T=False and zero W after random_init: this gives
    theta ratio = 1 at init (same as diagonal_T=True) without the architectural
    restriction.

    random_bound scales as sqrt(param_bound) so E[T_ii]/sigma stays ~3.3,
    keeping the Schur complement Q - W^T T^{-1} W positive after the first
    CMA perturbation.
    """
    random_bound = max(2.0, param_bound ** 0.5)
    sigma = param_bound * 0.1
    for _ in range(200):
        m = RTBM(nv, nh, init_max_param_bound=param_bound, random_bound=random_bound,
                 diagonal_T=False, mode=RTBM.Mode.LogProbability)
        if np.all(np.diag(m.t) > sigma) and np.all(np.diag(m.q) > sigma):
            params = np.real(m.get_parameters()).copy()
            params[nv + nh : nv + nh + nv * nh] = 0.0
            if m.set_parameters(params):
                # Widen CMA bounds to contain the actual initial T entries.
                # With diagonal_T=False, T diagonal entries are sums of squares
                # (≈2*random_bound²) and can exceed param_bound. sigma stays at
                # param_bound*0.1 for conservative initial exploration.
                actual_max = float(np.max(np.abs(params)))
                m.set_bounds(max(param_bound, actual_max) * 1.2)
                return m
    return m


# ── training with history capture ─────────────────────────────────────────────
def train_rtbm(model, x_theta, ncores=PARALLEL_CORES, maxiter=200, tolfun=1e-5):
    """CMA-ES training that returns (solution, per-iteration best-cost list)."""
    initsol  = np.real(model.get_parameters())
    sigma    = np.max(model.get_bounds()[1]) * 0.1
    cma_opts = {
        'bounds':           model.get_bounds(),
        'tolfun':           0,        # disabled: with NAN_PENALTY all-equal populations trigger this
        'maxiter':          maxiter,
        'verb_log':         0,
        'tolflatfitness':   maxiter,  # convergence criterion: stop when best NLL stagnates
    }
    es      = CMAEvolutionStrategy(initsol, sigma, cma_opts)
    history = []

    NAN_PENALTY = 1e9   # finite stand-in for constraint-violating candidates

    if ncores > 1:
        with closing(mp.Pool(ncores, initializer=worker_initialize,
                             initargs=(log_nll_cost, model, x_theta, None))) as pool:
            while not es.stop():
                candidates = es.ask()
                fits = pool.map(worker_compute, candidates)
                f_values = [v if np.isfinite(v) else NAN_PENALTY for v in fits]
                es.tell(candidates, f_values)
                es.disp()
                history.append(es.best.f)
            pool.terminate()
    else:
        worker_initialize(log_nll_cost, model, x_theta, None)
        while not es.stop():
            candidates = es.ask()
            f_values = [worker_compute(s) for s in candidates]
            f_values = [v if np.isfinite(v) else NAN_PENALTY for v in f_values]
            es.tell(candidates, f_values)
            es.disp()
            history.append(es.best.f)

    model.set_parameters(es.result[0])
    return es.result[0], history


# ── evaluation helpers ─────────────────────────────────────────────────────────
def anomaly_scores(model, x_theta):
    try:
        log_probs = np.real(model(x_theta)).flatten()
        log_probs = np.where(np.isfinite(log_probs), log_probs, -1e6)
        return -log_probs
    except np.linalg.LinAlgError:
        return np.full(x_theta.shape[1], 1e6)


def mean_nll(model, x_theta):
    try:
        log_probs = np.real(model(x_theta)).flatten()
        finite = log_probs[np.isfinite(log_probs)]
        return float(-np.mean(finite)) if len(finite) > 0 else 1e9
    except np.linalg.LinAlgError:
        return 1e9


# ── hyperopt ──────────────────────────────────────────────────────────────────
SEARCH_MAXITER = 150   # fast per-trial iterations during search
SEARCH_TOLFUN  = 1e-4

search_space = {
    'n_hidden':    hp.choice('n_hidden',    [2,3,4]),
    'param_bound': hp.loguniform('param_bound', np.log(1.0), np.log(8.0)),
}


def make_objective(X_tr, X_val, ncores):
    def objective(params):
        nh = int(params['n_hidden'])
        pb = float(params['param_bound'])
        m = make_rtbm(N_VISIBLE, nh, pb)
        try:
            train_rtbm(m, X_tr, ncores=ncores, maxiter=SEARCH_MAXITER, tolfun=SEARCH_TOLFUN)
            loss = mean_nll(m, X_val)
        except Exception as exc:
            print(f"  [HYPEROPT] Trial failed: {exc}")
            loss = 1e9
        print(f"  n_hidden={nh}  param_bound={pb:.2f}  val_NLL={loss:.4f}")
        return {'loss': loss, 'status': STATUS_OK}
    return objective


# ── plots ──────────────────────────────────────────────────────────────────────
def plot_loss_history(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history, color='steelblue', linewidth=1.5)
    plt.title('RTBM: CMA-ES Convergence', fontweight='bold')
    plt.xlabel('CMA-ES Iteration')
    plt.ylabel(r'Best $-\!\sum\log P(x)$')
    plt.yscale('log')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(OUTDIR, 'history.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved training history to '{path}'")


def plot_density_check(model, X_val_theta, val_raw):
    """
    Analogue of the autoencoder reconstruction plot.
    Shows per-feature histograms of validation data overlaid with the P(x)-weighted
    histogram, so the learned density can be compared to the true distribution.
    """
    # model is in LogProbability mode; use log-sum-exp for numerically stable weights
    try:
        log_probs = np.real(model(X_val_theta)).flatten()
    except np.linalg.LinAlgError:
        log_probs = np.zeros(X_val_theta.shape[1])
    log_probs -= log_probs.max()
    probs = np.exp(log_probs)
    w     = probs / probs.sum() if probs.sum() > 0 else np.ones(len(probs)) / len(probs)

    labels = [r'$x_{vis}$', r'$Iso = \Sigma E_{ph}/E_{track}$',
              r'$f_{had} = E_{HCAL}/E_{tot}$', r'$\eta$ (Scaled)']

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(r'RTBM: Density Check on $\tau\to\pi\nu$ (Validation)', fontweight='bold')

    for i in range(4):
        row, col = divmod(i, 2)
        feat = val_raw[:, i]
        bins = np.linspace(feat.min(), feat.max(), 40)
        ax[row, col].hist(feat, bins=bins, histtype='step', linewidth=1.5,
                          density=True, label='Validation data')
        ax[row, col].hist(feat, bins=bins, weights=w * len(feat),
                          histtype='step', linewidth=1.5, linestyle='--',
                          density=True, label=r'RTBM $P(x)$ weighted')
        ax[row, col].set_title(labels[i])
        ax[row, col].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTDIR, 'density_check.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved density check to '{path}'")


def plot_anomaly_scores(scores_pi, scores_rho):
    threshold_95  = np.percentile(scores_pi, 95)
    bkg_rejection = np.sum(scores_rho > threshold_95) / len(scores_rho)

    print("\n=======================================================")
    print(f"Target Signal Efficiency : 95.00%")
    print(f"Optimal NLL Threshold    : {threshold_95:.4f}")
    print(f"Background Rejection     : {bkg_rejection * 100:.2f}%")
    print("=======================================================\n")

    lo   = np.percentile(np.concatenate([scores_pi, scores_rho]), 0.5)
    hi   = np.percentile(np.concatenate([scores_pi, scores_rho]), 99.5)
    bins = np.linspace(lo, hi, 80)

    w_pi  = np.ones_like(scores_pi)  / len(scores_pi)
    w_rho = np.ones_like(scores_rho) / len(scores_rho)

    plt.figure(figsize=(8, 6))
    plt.hist(scores_pi,  bins=bins, weights=w_pi,  histtype='step',
             linewidth=2, label=r'Signal: $\tau\to\pi\nu$',      color='blue')
    plt.hist(scores_rho, bins=bins, weights=w_rho, histtype='step',
             linewidth=2, label=r'Background: $\tau\to\rho\nu$', color='red')

    pi_h,  _ = np.histogram(scores_pi,  bins=bins, weights=w_pi)
    rho_h, _ = np.histogram(scores_rho, bins=bins, weights=w_rho)
    centers  = 0.5 * (bins[:-1] + bins[1:])
    plt.fill_between(centers, pi_h,  0, where=(centers <= threshold_95),
                     step='mid', color='blue', alpha=0.2)
    plt.fill_between(centers, rho_h, 0, where=(centers >= threshold_95),
                     step='mid', color='red',  alpha=0.2)
    plt.axvline(threshold_95, color='black', linestyle='--', linewidth=1.5,
                label=r'Threshold: 95% eff. in $\pi$ data')

    plt.title(r'RTBM: Anomaly Scores ($-\log P(x)$)', fontweight='bold')
    plt.xlabel(r'$-\log P(x)$')
    plt.ylabel('Fraction of Events')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    path = os.path.join(OUTDIR, 'anomaly_scores.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved anomaly scores to '{path}'")


def plot_roc(scores_pi, scores_rho):
    y_true   = np.concatenate([np.zeros(len(scores_pi)), np.ones(len(scores_rho))])
    y_scores = np.concatenate([scores_pi, scores_rho])
    mask     = np.isfinite(y_scores)
    y_true, y_scores = y_true[mask], y_scores[mask]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr, tpr, color='darkorange', alpha=0.2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'RTBM AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [1, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR (Signal Loss)')
    plt.ylabel('TPR (Background Rejection)')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc='lower right', frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    path = os.path.join(OUTDIR, 'roc_curve.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved ROC curve to '{path}'")
    print(f"\n[RESULT] Final RTBM AUC: {roc_auc:.4f}")


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    start = time.time()
    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    # split: n_train events for CMA-ES (80/20 train/val), rest for test
    N       = min(args.n_train, len(pi_data))
    N_tr    = int(0.8 * N)
    train_pi = pi_data[:N_tr]
    val_pi   = pi_data[N_tr:N]
    test_pi  = pi_data[N:]        # held-out for final ROC evaluation

    print(f"[INFO] Train: {len(train_pi)} | Val: {len(val_pi)} | Test pions: {len(test_pi)} | Rho: {len(rho_data)}")

    # standardize
    tr_std, [val_std, test_std, rho_std], (mu, scl) = standardize(
        train_pi, val_pi, test_pi, rho_data)

    # theta expects shape (N_features, N_events)
    X_tr   = tr_std.T
    X_val  = val_std.T
    X_test = test_std.T
    X_rho  = rho_std.T

    ncores = min(args.ncores, mp.cpu_count())

    # ── optional hyperparameter search ────────────────────────────────────────
    n_hidden    = args.n_hidden
    param_bound = args.param_bound

    if args.optimize:
        print(f"\n[HYPEROPT] Starting Bayesian search ({args.max_evals} evaluations)...")
        trials      = Trials()
        best_idx    = fmin(fn=make_objective(X_tr, X_val, ncores),
                           space=search_space, algo=tpe.suggest,
                           max_evals=args.max_evals, trials=trials)
        best_params = space_eval(search_space, best_idx)
        n_hidden    = int(best_params['n_hidden'])
        param_bound = float(best_params['param_bound'])

        print("\n=======================================================")
        print("BEST HYPERPARAMETERS:")
        print(f"  n_hidden    : {n_hidden}")
        print(f"  param_bound : {param_bound:.2f}")
        print(f"  Best val NLL: {trials.best_trial['result']['loss']:.4f}")
        print("=======================================================\n")

        with open(os.path.join(OUTDIR, 'hyperopt_trials.pkl'), 'wb') as fh:
            pickle.dump(trials, fh)

    # ── final training ─────────────────────────────────────────────────────────
    print(f"\n[TRAIN] RTBM({N_VISIBLE}, {n_hidden}), param_bound={param_bound:.1f}, "
          f"maxiter={args.maxiter}, tolfun={args.tolfun}")
    model = make_rtbm(N_VISIBLE, n_hidden, param_bound)

    _, history = train_rtbm(model, X_tr, ncores=ncores,
                             maxiter=args.maxiter, tolfun=args.tolfun)

    val_nll = mean_nll(model, X_val)
    print(f"\n[INFO] Validation NLL: {val_nll:.4f}")

    if args.save:
        payload = {
            'params':      model.get_parameters(),
            'n_visible':   N_VISIBLE,
            'n_hidden':    n_hidden,
            'param_bound': param_bound,
            'mu':          mu,
            'std':         scl,
        }
        model_path = os.path.join(OUTDIR, 'model.pkl')
        with open(model_path, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f"[INFO] Saved model to '{model_path}'")

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_loss_history(history)
    plot_density_check(model, X_val, val_pi)

    sc_pi  = anomaly_scores(model, X_test)
    sc_rho = anomaly_scores(model, X_rho)

    plot_anomaly_scores(sc_pi, sc_rho)
    plot_roc(sc_pi, sc_rho)

    print(f'Tot time: {time.time()-start:.2f}')
