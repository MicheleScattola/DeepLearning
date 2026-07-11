import os
import sys
import argparse
import pickle
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize, train_val_test_split,
    make_rtbm, train_rtbm, mean_nll,
)

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(prog='RTBMTraining',
                                 description='RTBM training and hyperparameter search')
parser.add_argument('-o',  '--outfile',     default='trained_rtbm')
parser.add_argument('-nh', '--n_hidden',    type=int,   default=2)
parser.add_argument('--maxiter',            type=int,   default=200)
parser.add_argument('--tolfun',             type=float, default=1e-5)
parser.add_argument('--ncores',             type=int,   default=PARALLEL_CORES)
parser.add_argument('--param_bound',        type=float, default=5.0)
parser.add_argument('--n_train',            type=int,   default=8000,
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


# ── hyperopt ──────────────────────────────────────────────────────────────────
SEARCH_MAXITER = 150   # fast per-trial iterations during search
SEARCH_TOLFUN  = 1e-4

search_space = {
    'n_hidden':    hp.choice('n_hidden',    [4]),
    'param_bound': hp.loguniform('param_bound', np.log(5.0), np.log(7.0)),
}


def make_objective(X_tr, X_val, ncores):
    def objective(params):
        nh = int(params['n_hidden'])
        pb = float(params['param_bound'])
        m  = make_rtbm(N_VISIBLE, nh, pb)
        try:
            train_rtbm(m, X_tr, ncores=ncores, maxiter=SEARCH_MAXITER,
                       tolfun=SEARCH_TOLFUN, init_sigma=pb * 0.1)
            loss = mean_nll(m, X_val)
        except Exception as exc:
            print(f"  [HYPEROPT] Trial failed: {exc}")
            loss = 1e9
        print(f"  n_hidden={nh}  param_bound={pb:.2f}  val_NLL={loss:.4f}")
        return {'loss': loss, 'status': STATUS_OK}
    return objective


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    # split: n_train events for CMA-ES (80/20 train/val), rest for test
    train_pi, val_pi, test_pi = train_val_test_split(pi_data, args.n_train)

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
        trials   = Trials()
        best_idx = fmin(fn=make_objective(X_tr, X_val, ncores), space=search_space,
                        algo=tpe.suggest,
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
    print(f"\n[OPT] RTBM({N_VISIBLE}, {n_hidden}), param_bound={param_bound:.1f}, "
          f"maxiter={args.maxiter}, tolfun={args.tolfun}")
    with open(os.path.join(OUTDIR, 'args.txt'), 'a') as _f:
        _f.write(f"\n[OPT] RTBM({N_VISIBLE}, {n_hidden}), param_bound={param_bound:.1f}, "
          f"maxiter={args.maxiter}, tolfun={args.tolfun}")
        
    model  = make_rtbm(N_VISIBLE, n_hidden, param_bound)
    result = train_rtbm(model, X_tr, ncores=ncores,
                        maxiter=args.maxiter, tolfun=args.tolfun,
                        init_sigma=param_bound * 0.1)
    history = result[1]

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
            'n_train':     args.n_train,
            'history':     history,
        }
        model_path = os.path.join(OUTDIR, 'model.pkl')
        with open(model_path, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f"[INFO] Saved model to '{model_path}'")
        print(f"[INFO] Run: python evaluation.py {args.outfile}")