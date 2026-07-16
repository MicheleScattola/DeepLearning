"""Test whether the original theta CMA training hangs.

Uses make_rtbm for initialisation, then calls theta.minimizer.CMA.train()
unchanged. A SIGALRM fires after HANG_TIMEOUT seconds so the script
doesn't freeze forever.

Run from rtbm_workspace/ with the Python 3.8 venv:
    .venv/bin/python3 debug.py [--nh N] [--pb F] [--n_train N] [--maxiter N]
"""
import os
import sys
import time
import signal
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.minimizer import CMA
from theta.costfunctions import sum as log_nll_cost
from rtbmlib import N_VISIBLE, PARALLEL_CORES, load_datasets, standardize, make_rtbm
import multiprocessing as mp

HANG_TIMEOUT = 120

p = argparse.ArgumentParser()
p.add_argument('--nh',      type=int,   default=3)
p.add_argument('--pb',      type=float, default=4.3)
p.add_argument('--n_train', type=int,   default=5000)
p.add_argument('--maxiter', type=int,   default=50)
p.add_argument('--ncores',  type=int,   default=min(PARALLEL_CORES, mp.cpu_count()))
args = p.parse_args()

print(f"[CONFIG] nh={args.nh}  pb={args.pb}  n_train={args.n_train}  "
      f"maxiter={args.maxiter}  ncores={args.ncores}  timeout={HANG_TIMEOUT}s")

pi, _ = load_datasets()
np.random.seed(42)
np.random.shuffle(pi)
n_tr = int(0.8 * args.n_train)
tr_std, [_], _ = standardize(pi[:n_tr], pi[n_tr:args.n_train])
X_tr = tr_std.T

print("[INFO] Initialising with make_rtbm ...")
model = make_rtbm(N_VISIBLE, args.nh, args.pb)
print("[INFO] Init OK — calling theta CMA.train() ...")

trainer = CMA(parallel=args.ncores > 1, ncores=args.ncores)

def _on_timeout(signum, frame):
    raise RuntimeError(f"Hang after {HANG_TIMEOUT}s — likely inner resampling while")

signal.signal(signal.SIGALRM, _on_timeout)
signal.alarm(HANG_TIMEOUT)

t0 = time.perf_counter()
try:
    trainer.train(log_nll_cost, model, X_tr, tolfun=0, maxiter=args.maxiter)
    signal.alarm(0)
    print(f"[OK] Finished in {time.perf_counter() - t0:.1f}s")
except RuntimeError as e:
    signal.alarm(0)
    print(f"[HANG] {time.perf_counter() - t0:.1f}s elapsed — {e}")