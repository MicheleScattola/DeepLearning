"""RTBM inference time benchmark.

Run from rtbm_workspace/ with the Python 3.8 venv:
    .venv/bin/python3 inference_time.py [--model PATH]
"""
import os
import sys
import time
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from theta.rtbm import RTBM
from rtbmlib import load_datasets, anomaly_scores

N_WARMUP = 200
N_RUNS   = 3

DEFAULT_MODEL = 'training/f_single_70k_nh3_pb4_3_maxiter300/model.pkl'


def load_model(path):
    payload = pickle.load(open(path, 'rb'))
    m = RTBM(payload['n_visible'], payload['n_hidden'],
             init_max_param_bound=payload['param_bound'],
             diagonal_T=False, mode=RTBM.Mode.LogProbability)
    m.set_parameters(payload['params'])
    return m, payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    args = p.parse_args()

    print(f"[INFO] Loading model from '{args.model}'...")
    model, payload = load_model(args.model)
    mu, std = payload['mu'], payload['std']
    print(f"[INFO] RTBM: N_visible={payload['n_visible']}, N_hidden={payload['n_hidden']}")

    print("[INFO] Loading data...")
    pi, _ = load_datasets()
    X = ((pi - mu) / std).T  # (4, N) — theta convention
    N = X.shape[1]
    print(f"[INFO] {N} events")

    print(f"[INFO] Warm-up ({N_WARMUP} events)...")
    _ = anomaly_scores(model, X[:, :N_WARMUP])

    print(f"[INFO] Benchmarking ({N_RUNS} runs)...")
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _ = anomaly_scores(model, X)
        times.append(time.perf_counter() - t0)
        print(f"  run {i+1}: {times[-1]*1000:.1f} ms")

    mean_t = np.mean(times)
    lines = [
        "[RTBM RESULT]",
        f"  N_hidden        : {payload['n_hidden']}",
        f"  Events          : {N}",
        f"  Runs            : {N_RUNS}",
        f"  Mean total time : {mean_t*1000:.1f} ms",
        f"  Time per event  : {mean_t/N*1e6:.2f} us/event",
        f"  Throughput      : {N/mean_t:.0f} events/s",
    ]
    for line in lines:
        print(line)
    with open('inference_time.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("[INFO] Results written to 'inference_time.txt'")