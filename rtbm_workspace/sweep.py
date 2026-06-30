"""Grid + seed sweep over (n_hidden, param_bound) for RTBM performance studies.

Implements performance.md Sections 2, 3, 5: fixes a function-evaluation
budget (not a fixed iteration count) so models with different N_hidden are
compared fairly, runs every trial to completion (NAN_PENALTY stays constant,
no early truncation of stuck configs), and separates the data split (fixed
data_seed) from the model/CMA-ES randomness (varying run seed) so that
seed-to-seed variance reflects optimisation luck, not which events ended up
in train/val/test.

Does not call training.py or parse its args.txt/stdout -- imports rtbmlib
directly and writes its own structured CSV, one row per (nh, pb, seed) run.

Usage:
    python sweep.py --nh 2,3,4 --pb 1,2,3,5,8 --seeds 1,2,3,4,5 \\
        --feval_budget 3000 --n_train 8000 --ncores 4 \\
        --out sweep_results.csv --runs_dir sweep_runs
"""
import os
import sys
import csv
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'theta')))

from rtbmlib import (
    N_VISIBLE, PARALLEL_CORES,
    load_datasets, standardize, train_val_test_split,
    make_rtbm, train_rtbm, mean_nll, anomaly_scores,
    compute_auc, background_rejection, default_popsize, run_tag,
)

CSV_FIELDS = [
    'nh', 'param_bound', 'seed', 'n_params', 'popsize', 'maxiter', 'fevals',
    'time_sec', 'val_nll', 'auc', 'bkg_rejection_95',
    'final_valid_fraction', 'mean_valid_fraction', 'status',
]


def parse_float_list(s):
    return [float(x) for x in s.split(',')]


def parse_int_list(s):
    return [int(x) for x in s.split(',')]


def main():
    p = argparse.ArgumentParser(description='RTBM hyperparameter performance sweep')
    p.add_argument('--nh',           type=parse_int_list,   default=[2, 3, 4],
                    help='Comma-separated N_hidden values, e.g. 2,3,4')
    p.add_argument('--pb',           type=parse_float_list, default=[1.0, 2.0, 3.0, 5.0, 8.0],
                    help='Comma-separated param_bound values, e.g. 1,2,3,5,8')
    p.add_argument('--seeds',        type=parse_int_list,   default=[1, 2, 3],
                    help='Comma-separated model/CMA-ES seeds, e.g. 1,2,3')
    p.add_argument('--feval_budget', type=int,   default=3000,
                    help='Target CMA-ES function evaluations per run (converted to maxiter via popsize)')
    p.add_argument('--n_train',      type=int,   default=8000)
    p.add_argument('--data_seed',    type=int,   default=42,
                    help='Fixed seed for data shuffling/splitting, independent of --seeds')
    p.add_argument('--ncores',       type=int,   default=PARALLEL_CORES)
    p.add_argument('--gen_timeout',  type=int,   default=60,
                    help='Per-CMA-generation timeout in seconds; aborts a run instead of '
                         'hanging if a candidate hits a near-singular T/Q (see rtbmlib.TrainingTimeout)')
    p.add_argument('--out',          default='sweep_results.csv')
    p.add_argument('--runs_dir',     default='sweep_runs',
                    help='Where per-run convergence/valid-fraction .npy diagnostics are saved')
    args = p.parse_args()

    os.makedirs(args.runs_dir, exist_ok=True)

    # ── data: split/standardize ONCE with a fixed seed, reused for every run ──
    # This isolates "optimisation luck" (varied via --seeds) from "which events
    # landed in train/val/test" (fixed), per performance.md Section 5.
    np.random.seed(args.data_seed)
    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()
    np.random.shuffle(pi_data)
    train_pi, val_pi, test_pi = train_val_test_split(pi_data, args.n_train)
    print(f"[INFO] Train: {len(train_pi)} | Val: {len(val_pi)} | "
          f"Test pions: {len(test_pi)} | Rho: {len(rho_data)}")

    tr_std, [val_std, test_std, rho_std], _ = standardize(train_pi, val_pi, test_pi, rho_data)
    X_tr, X_val, X_test, X_rho = tr_std.T, val_std.T, test_std.T, rho_std.T

    combos = [(nh, pb, seed) for nh in args.nh for pb in args.pb for seed in args.seeds]
    print(f"[INFO] {len(combos)} runs: {len(args.nh)} nh x {len(args.pb)} pb x {len(args.seeds)} seeds")

    write_header = not os.path.exists(args.out)
    csv_file = open(args.out, 'a', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    for i, (nh, pb, seed) in enumerate(combos):
        tag = run_tag(nh, pb, seed)
        print(f"\n[{i+1}/{len(combos)}] nh={nh} param_bound={pb:.2f} seed={seed}")

        row = {'nh': nh, 'param_bound': pb, 'seed': seed, 'status': 'ok'}

        # model-and-CMA randomness uses --seeds; data above used --data_seed
        np.random.seed(seed)

        try:
            model = make_rtbm(N_VISIBLE, nh, pb)
        except RuntimeError as exc:
            print(f"  [FAIL] init: {exc}")
            row.update(n_params=None, popsize=None, maxiter=None, fevals=None,
                       time_sec=None, val_nll=1e9, auc=None, bkg_rejection_95=None,
                       final_valid_fraction=None, mean_valid_fraction=None,
                       status='init_failed')
            writer.writerow(row); csv_file.flush()
            continue

        n_params = model.size()
        popsize  = default_popsize(n_params)
        maxiter  = max(5, args.feval_budget // popsize)
        row.update(n_params=n_params, popsize=popsize, maxiter=maxiter)

        t0 = time.time()
        try:
            _, history, diag = train_rtbm(model, X_tr, ncores=args.ncores, maxiter=maxiter,
                                           seed=seed, return_diagnostics=True,
                                           gen_timeout=args.gen_timeout)
        except Exception as exc:
            print(f"  [FAIL] train: {exc}")
            row.update(fevals=None, time_sec=time.time() - t0, val_nll=1e9, auc=None,
                       bkg_rejection_95=None, final_valid_fraction=None,
                       mean_valid_fraction=None, status='train_failed')
            writer.writerow(row); csv_file.flush()
            continue
        elapsed = time.time() - t0

        val_nll = mean_nll(model, X_val)
        sc_pi   = anomaly_scores(model, X_test)
        sc_rho  = anomaly_scores(model, X_rho)
        auc_val = compute_auc(sc_pi, sc_rho)
        bkg_rej = background_rejection(sc_pi, sc_rho, target_eff=0.95)

        vf = diag['valid_fraction']
        row.update(
            fevals=diag['fevals'], time_sec=elapsed, val_nll=val_nll, auc=auc_val,
            bkg_rejection_95=bkg_rej, final_valid_fraction=vf[-1] if vf else None,
            mean_valid_fraction=float(np.mean(vf)) if vf else None,
        )
        writer.writerow(row); csv_file.flush()

        np.save(os.path.join(args.runs_dir, f'{tag}_history.npy'), np.array(history))
        np.save(os.path.join(args.runs_dir, f'{tag}_validfrac.npy'), np.array(vf))

        print(f"  val_NLL={val_nll:.4f}  AUC={auc_val:.4f}  bkg_rej={bkg_rej*100:.1f}%  "
              f"fevals={diag['fevals']}  time={elapsed:.1f}s")

    csv_file.close()
    print(f"\n[DONE] Results written to '{args.out}', diagnostics in '{args.runs_dir}/'")


if __name__ == '__main__':
    main()