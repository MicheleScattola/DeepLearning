"""RTBM training using the original theta CMA.train() function.

"""
import os
import sys
import argparse
import pickle
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize, train_val_test_split,
    make_rtbm, mean_nll, anomaly_scores, compute_auc,
)

parser = argparse.ArgumentParser()
parser.add_argument('-o',  '--outfile',    default='simple_training')
parser.add_argument('-nh', '--n_hidden',   type=int,   nargs='+', default=[2, 3])
parser.add_argument('--param_bound',       type=float, default=5)
parser.add_argument('--n_train',           type=int,   default=80000)
parser.add_argument('--maxiter',           type=int,   default=300)
parser.add_argument('--ncores',            type=int,   default=PARALLEL_CORES)
parser.add_argument('--optimize',          action='store_true')
parser.add_argument('--max_evals',         type=int,   default=70)
parser.add_argument('--pb_range',          type=float, nargs=2, default=[8.0, 12.0],
                    metavar=('MIN', 'MAX'), help='param_bound search range for --optimize')
parser.add_argument('--save',              action='store_true')
args = parser.parse_args()

OUTDIR = os.path.join('training', args.outfile)
os.makedirs(OUTDIR, exist_ok=True)
with open(os.path.join(OUTDIR, 'args.txt'), 'w') as f:
    for k, v in vars(args).items():
        f.write(f"{k}: {v}\n")

np.random.seed(42)


def train_once(model, X_tr, ncores, maxiter, tol):
    CMA(parallel=ncores > 1, ncores=ncores).train(
        log_nll_cost, model, X_tr, tolfun=tol, maxiter=maxiter)


if __name__ == '__main__':
    ncores = min(args.ncores, mp.cpu_count())

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)

    train_pi, val_pi, test_pi = train_val_test_split(pi_data, args.n_train)
    print(f"[INFO] Train={len(train_pi)}  Val={len(val_pi)}  Test={len(test_pi)}  Rho={len(rho_data)}")

    tr_std, [val_std, test_std, rho_std], (mu, scl) = standardize(
        train_pi, val_pi, test_pi, rho_data)
    X_tr   = tr_std.T
    X_val  = val_std.T
    X_test = test_std.T
    X_rho  = rho_std.T

    n_hidden    = args.n_hidden[0]
    param_bound = args.param_bound

    if args.optimize:
        print(f"\n[HYPEROPT] Bayesian search: n_hidden ∈ {args.n_hidden}  "
              f"param_bound ∈ {args.pb_range}  ({args.max_evals} evals)...")
        space = {
            'n_hidden':    hp.choice('n_hidden',    args.n_hidden),
            'param_bound': hp.uniform('param_bound', args.pb_range[0], args.pb_range[1]),
        }

        def objective(params):
            nh = int(params['n_hidden'])
            pb = float(params['param_bound'])
            m  = make_rtbm(N_VISIBLE, nh, pb)
            try:
                train_once(m, X_tr, ncores, maxiter=150,tol=1e-6)
                loss = mean_nll(m, X_val)
            except Exception as e:
                print(f"  [HYPEROPT] failed: {e}")
                loss = 1e9
            print(f"  n_hidden={nh}  param_bound={pb:.3f}  val_NLL={loss:.4f}")
            return {'loss': loss, 'status': STATUS_OK}

        trials   = Trials()
        best_idx = fmin(fn=objective, space=space, algo=tpe.suggest,
                        max_evals=args.max_evals, trials=trials)
        best      = space_eval(space, best_idx)
        n_hidden    = int(best['n_hidden'])
        param_bound = float(best['param_bound'])

        print(f"\n[HYPEROPT] Best n_hidden={n_hidden}  param_bound={param_bound:.3f}  "
              f"NLL={trials.best_trial['result']['loss']:.4f}")
        with open(os.path.join(OUTDIR, 'hyperopt_trials.pkl'), 'wb') as fh:
            pickle.dump(trials, fh)

    print(f"\n[TRAIN] RTBM(N_v={N_VISIBLE}, N_h={n_hidden})  "
          f"param_bound={param_bound:.2f}  maxiter={args.maxiter}  ncores={ncores}")

    model = make_rtbm(N_VISIBLE, n_hidden, param_bound)
    train_once(model, X_tr, ncores, args.maxiter,tol=1e-11)

    val_nll  = mean_nll(model, X_val)
    test_auc = compute_auc(anomaly_scores(model, X_test), anomaly_scores(model, X_rho))
    print(f"[RESULT] val NLL={val_nll:.4f}  test AUC={test_auc:.4f}")

    with open(os.path.join(OUTDIR, 'args.txt'), 'a') as f:
        f.write(f"\n[RESULT]\n")
        f.write(f"n_hidden    : {n_hidden}\n")
        f.write(f"param_bound : {param_bound:.4f}\n")
        f.write(f"val_nll     : {val_nll:.4f}\n")
        f.write(f"test_auc    : {test_auc:.4f}\n")

    if args.save:
        payload = {
            'params':      model.get_parameters(),
            'n_visible':   N_VISIBLE,
            'n_hidden':    n_hidden,
            'param_bound': param_bound,
            'mu':          mu,
            'std':         scl,
            'n_train':     args.n_train,
        }
        path = os.path.join(OUTDIR, 'model.pkl')
        with open(path, 'wb') as fh:
            pickle.dump(payload, fh)
        print(f"[INFO] Saved model to '{path}'")
        print(f"[INFO] Run: python evaluation.py {args.outfile}")