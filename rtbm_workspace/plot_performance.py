"""Generate the performance.md Section 6 plot catalogue from sweep.py output.

Reads the CSV written by sweep.py and the per-run .npy diagnostics in
--runs_dir, producing:
    scaling_nh.png        AUC / val_NLL vs N_hidden, mean +/- std over seeds,
                           each N_hidden evaluated at its own best param_bound
    scaling_nh_pb<v>.png   same, but with param_bound held FIXED at <v> across
                           all N_hidden -- one file per param_bound present in
                           the data, isolating N_h from the param_bound/seed-
                           variance trade-off (math.md Section 11.5)
    heatmap_pb_nh.png      mean val_NLL over the (N_hidden, param_bound) grid
    convergence_overlay.png   CMA best-NLL vs fevals, one curve per N_hidden
    valid_fraction.png    fraction of non-penalty CMA candidates per
                           iteration, one curve per N_hidden (diagnoses how
                           hard the Schur constraint is for each config)
    compute_scaling.png    wall-clock time / ncores vs N_hidden, pooled
                           across param_bound and seeds (timing is driven
                           by the theta-function genus, not param_bound)

No pandas dependency -- aggregation is done with plain csv + numpy, matching
what is actually installed in the rtbm_workspace venv.

Usage:
    python plot_performance.py --csv sweep_results.csv --runs_dir sweep_runs \\
        --outdir sweep_plots
"""
import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rtbmlib import run_tag

# val_NLL above this is the post-training LinAlgError fallback sentinel
# (math.md Section 7: a marginal-Schur es.result[0] that passed CMA's
# eigenvalue check but fails Cholesky in the main process). This is a
# categorical "this seed crashed" event, not a heavy-tailed continuation
# of normal variation -- with only a few seeds per config, folding it into
# a mean/std (or even median/IQR) silently makes the entire plot unreadable,
# since one crashed seed at 1e9 swamps real values of order 5-10 by eight
# orders of magnitude. We exclude crashed rows from the quality-metric
# aggregates and report the crash rate separately instead.
CRASH_THRESHOLD = 1e8


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    def to_num(v):
        if v is None or v == '':
            return np.nan
        try:
            return float(v)
        except ValueError:
            return v  # leave strings (status) as-is

    return [{k: to_num(v) for k, v in row.items()} for row in rows]


def ok_rows(rows):
    return [r for r in rows if r['status'] == 'ok']


def uncrashed(rows):
    return [r for r in rows if r['val_nll'] < CRASH_THRESHOLD]


def best_pb_per_nh(rows):
    """For each nh, the param_bound with lowest mean val_NLL among non-crashed seeds.

    A pb whose every seed crashed is still returned (so plot_scaling has
    something to show and report as 100% crash rate) but is never preferred
    over a pb with at least one successful seed.
    """
    by_nh = {}
    for r in rows:
        by_nh.setdefault(r['nh'], {}).setdefault(r['param_bound'], []).append(r['val_nll'])

    def score(vals):
        good = [v for v in vals if v < CRASH_THRESHOLD]
        return np.mean(good) if good else np.inf

    return {nh: min(pbs, key=lambda pb: score(pbs[pb])) for nh, pbs in by_nh.items()}


def _quality_by_nh(rows, nh_to_pb, all_rows=None, label_fn=None):
    """Core aggregation shared by plot_scaling and plot_scaling_per_param_bound.

    nh_to_pb: {nh: param_bound} -- which param_bound's rows to use for each nh
    (the two callers differ only in how this mapping is built: best-per-nh,
    vs the same fixed pb for every nh).

    all_rows, if given, includes train_failed (timeout/init) rows -- used only
    to report the true attempted-seed count in the warning text, since `rows`
    is the already status=='ok'-filtered set.
    """
    nh_values = sorted(nh_to_pb)
    nll_mean, nll_std, auc_mean, auc_std, n_good_list = [], [], [], [], []
    for nh in nh_values:
        pb = nh_to_pb[nh]
        sel = [r for r in rows if r['nh'] == nh and r['param_bound'] == pb]
        n_attempted = (sum(1 for r in all_rows if r['nh'] == nh and r['param_bound'] == pb)
                       if all_rows is not None else len(sel))
        n_crash = sum(1 for r in sel if r['val_nll'] >= CRASH_THRESHOLD)
        n_timeout_or_init_fail = n_attempted - len(sel)
        good = uncrashed(sel)
        n_good_list.append(len(good))
        tag = label_fn(nh, pb) if label_fn else f"N_h={int(nh)} (pb={pb:.2g})"
        if n_crash or n_timeout_or_init_fail:
            print(f"  [WARN] {tag}: of {n_attempted} attempted seed(s), "
                  f"{n_timeout_or_init_fail} timed out/failed init, {n_crash} hit the post-training "
                  f"LinAlgError fallback (math.md Sec.7) -- {len(good)} usable")
        if len(good) < 2:
            print(f"  [WARN] {tag}: only {len(good)} usable seed(s) "
                  f"out of {n_attempted} attempted -- error bar is not a real uncertainty estimate")
        nll = [r['val_nll'] for r in good]
        auc = [r['auc'] for r in good]
        nll_mean.append(np.mean(nll) if nll else np.nan); nll_std.append(np.std(nll) if nll else 0.0)
        auc_mean.append(np.mean(auc) if auc else np.nan); auc_std.append(np.std(auc) if auc else 0.0)
    return nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list


def _plot_quality_panels(nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list, suptitle, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.errorbar(nh_values, nll_mean, yerr=nll_std, marker='o', capsize=4, color='steelblue')
    for nh, y, n in zip(nh_values, nll_mean, n_good_list):
        ax1.annotate(f'n={n}', (nh, y), textcoords='offset points', xytext=(8, 8), fontsize=8)
    ax1.set_xlabel('$N_h$'); ax1.set_ylabel('Validation NLL / event')
    ax1.set_title('Density quality vs hidden units')
    ax1.set_xticks(nh_values)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.errorbar(nh_values, auc_mean, yerr=auc_std, marker='o', capsize=4, color='darkorange')
    for nh, y, n in zip(nh_values, auc_mean, n_good_list):
        ax2.annotate(f'n={n}', (nh, y), textcoords='offset points', xytext=(8, 8), fontsize=8)
    ax2.set_xlabel('$N_h$'); ax2.set_ylabel('AUC')
    ax2.set_title('Physics separation vs hidden units')
    ax2.set_xticks(nh_values)
    ax2.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(suptitle, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_scaling(rows, outdir, all_rows=None):
    """N_h scaling, each N_h evaluated at its own mean-best param_bound.

    Answers "what's the best achievable at each N_h". Note this can conflate
    architecture (N_h) with how stable the winning param_bound happens to be
    -- see plot_scaling_per_param_bound for the deconfounded view, and
    math.md Section 11.5.
    """
    best_pb = best_pb_per_nh(rows)
    nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list = \
        _quality_by_nh(rows, best_pb, all_rows=all_rows)
    _plot_quality_panels(
        nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list,
        'RTBM Scaling with $N_h$ (each at its own best param\\_bound, '
        'mean $\\pm$ std over non-crashed seeds)',
        os.path.join(outdir, 'scaling_nh.png'),
    )
    return best_pb


def plot_scaling_per_param_bound(rows, outdir, all_rows=None):
    """One scaling_nh_pb<value>.png per param_bound present in the data,
    holding param_bound FIXED across N_h (instead of each N_h picking its
    own best). This isolates the N_h comparison from the param_bound/
    seed-variance trade-off in math.md Section 11.5 -- variance in val_NLL
    grows with param_bound for every N_h, so comparing each N_h at its own
    (possibly differently unstable) best param_bound can make some N_h
    values look noisier than others for reasons that have nothing to do
    with N_h itself.
    """
    pb_values = sorted(set(r['param_bound'] for r in rows))
    for pb in pb_values:
        nh_values_present = sorted(set(r['nh'] for r in rows if r['param_bound'] == pb))
        if not nh_values_present:
            continue
        nh_to_pb = {nh: pb for nh in nh_values_present}
        nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list = _quality_by_nh(
            rows, nh_to_pb, all_rows=all_rows,
            label_fn=lambda nh, pb: f"N_h={int(nh)} @ pb={pb:.2g}")
        tag = f'{pb:.3g}'.replace('.', 'p')
        _plot_quality_panels(
            nh_values, nll_mean, nll_std, auc_mean, auc_std, n_good_list,
            f'RTBM Scaling with $N_h$ at fixed param\\_bound={pb:.2g} '
            f'(mean $\\pm$ std over non-crashed seeds)',
            os.path.join(outdir, f'scaling_nh_pb{tag}.png'),
        )


def plot_heatmap(rows, outdir):
    nh_values = sorted(set(r['nh'] for r in rows))
    pb_values = sorted(set(r['param_bound'] for r in rows))
    sums   = np.zeros((len(nh_values), len(pb_values)))
    counts = np.zeros((len(nh_values), len(pb_values)))
    crash_n = np.zeros((len(nh_values), len(pb_values)))
    crash_total = np.zeros((len(nh_values), len(pb_values)))
    for r in rows:
        i = nh_values.index(r['nh']); j = pb_values.index(r['param_bound'])
        crash_total[i, j] += 1
        if r['val_nll'] >= CRASH_THRESHOLD:
            crash_n[i, j] += 1
        else:
            sums[i, j] += r['val_nll']; counts[i, j] += 1
    grid = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(grid, aspect='auto', cmap='viridis_r', origin='lower')
    plt.colorbar(im, ax=ax, label='Mean validation NLL (crashed seeds excluded)')
    # annotate cells with a partial or total crash rate so the exclusion is visible
    for i in range(len(nh_values)):
        for j in range(len(pb_values)):
            if crash_n[i, j] > 0:
                ax.text(j, i, f'{int(crash_n[i, j])}/{int(crash_total[i, j])}\ncrashed',
                        ha='center', va='center', fontsize=7, color='white')
    ax.set_xticks(range(len(pb_values))); ax.set_xticklabels([f'{pb:.2g}' for pb in pb_values])
    ax.set_yticks(range(len(nh_values))); ax.set_yticklabels([int(nh) for nh in nh_values])
    ax.set_xlabel('param_bound'); ax.set_ylabel('$N_h$')
    ax.set_title('Validation NLL across (param\\_bound, $N_h$)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, 'heatmap_pb_nh.png')
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_compute_scaling(rows, outdir):
    """Wall-clock time / ncores vs N_h, pooled across param_bound and seeds.

    Pooled rather than restricted to each N_h's best param_bound (unlike
    plot_scaling): compute cost is primarily driven by the theta-function
    genus (= N_h), not by param_bound, so pooling gives a much larger and
    more robust sample -- exactly the kind of n=1 fragility fixed in
    plot_scaling is avoided here by construction.

    Pooling is not just for sample size: it is what surfaces N_h=4's cost
    being genuinely bimodal in param_bound (math.md Section 10, "Sub-timeout
    cost"). At small param_bound, CMA-ES gets trapped near a marginally-
    singular starting region for the whole run, inflating the theta lattice
    sum's R^N_h cost on nearly every evaluation -- a >10x time difference
    between param_bound=1 and param_bound=8 at N_h=4 alone, at fixed ncores
    and generation count. This is real signal, not noise that a larger n
    would average away -- restricting to one param_bound would hide it.

    Only status=='ok' rows are used (rows already filtered that way by
    main()). Timed-out runs (status=='train_failed') are excluded: their
    time_sec reflects however many generations completed before hitting
    gen_timeout, not a real measurement of completed-run cost. A row with
    val_NLL >= CRASH_THRESHOLD (post-training LinAlgError fallback) is
    still included here -- that crash happens in mean_nll/anomaly_scores,
    *after* the timed train_rtbm() call returns, so its time_sec is a
    perfectly valid training-time measurement even though its quality
    metrics are not (see plot_scaling / CRASH_THRESHOLD).
    """
    nh_values = sorted(set(r['nh'] for r in rows))
    time_mean, time_std, n_list = [], [], []
    for nh in nh_values:
        sel = [r['time_sec'] / r['ncores'] for r in rows
               if r['nh'] == nh and r['ncores'] > 0]
        n_list.append(len(sel))
        time_mean.append(np.mean(sel) if sel else np.nan)
        time_std.append(np.std(sel) if sel else 0.0)
        if len(sel) < 2:
            print(f"  [WARN] N_h={int(nh)}: only {len(sel)} timed run(s) -- "
                  f"compute_scaling.png error bar is not a real uncertainty estimate")

    plt.figure(figsize=(7, 5))
    plt.errorbar(nh_values, time_mean, yerr=time_std, marker='o', capsize=4, color='seagreen')
    for nh, y, n in zip(nh_values, time_mean, n_list):
        plt.annotate(f'n={n}', (nh, y), textcoords='offset points', xytext=(8, 8), fontsize=8)
    plt.xlabel('$N_h$'); plt.ylabel('Wall-clock time / ncores (s)')
    plt.xticks(nh_values)
    plt.title('Compute Cost vs Hidden Units\n(pooled across param\\_bound and seeds)', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, 'compute_scaling.png')
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_convergence_and_valid_fraction(rows, best_pb, runs_dir, outdir):
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for nh in sorted(best_pb):
        pb = best_pb[nh]
        candidates = [r for r in rows if r['nh'] == nh and r['param_bound'] == pb]
        if not candidates:
            continue
        seed = int(candidates[0]['seed'])
        popsize = int(candidates[0]['popsize'])
        tag = run_tag(int(nh), pb, seed)

        hist_path = os.path.join(runs_dir, f'{tag}_history.npy')
        vf_path   = os.path.join(runs_dir, f'{tag}_validfrac.npy')
        if not (os.path.exists(hist_path) and os.path.exists(vf_path)):
            continue

        history = np.load(hist_path)
        vf      = np.load(vf_path)
        fevals  = popsize * np.arange(1, len(history) + 1)

        ax1.plot(fevals, history, label=f'$N_h$={int(nh)} (pb={pb:.2g})')
        ax2.plot(np.arange(1, len(vf) + 1), vf, label=f'$N_h$={int(nh)} (pb={pb:.2g})')

    ax1.set_xlabel('CMA-ES function evaluations'); ax1.set_ylabel(r'Best $-\sum\log P(x)$')
    ax1.set_yscale('log')
    ax1.set_title('Convergence vs fevals (representative seed per $N_h$)', fontweight='bold')
    ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)
    fig1.tight_layout()
    path1 = os.path.join(outdir, 'convergence_overlay.png')
    fig1.savefig(path1, dpi=300); plt.close(fig1)
    print(f"[PLOT] {path1}")

    ax2.set_xlabel('CMA-ES iteration'); ax2.set_ylabel('Fraction of valid candidates')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title('Schur-constraint difficulty per $N_h$', fontweight='bold')
    ax2.legend(); ax2.grid(True, linestyle=':', alpha=0.6)
    fig2.tight_layout()
    path2 = os.path.join(outdir, 'valid_fraction.png')
    fig2.savefig(path2, dpi=300); plt.close(fig2)
    print(f"[PLOT] {path2}")


def main():
    p = argparse.ArgumentParser(description='Plot RTBM performance sweep results')
    p.add_argument('--csv',      default='sweep_results.csv')
    p.add_argument('--runs_dir', default='sweep_runs')
    p.add_argument('--outdir',   default='sweep_plots')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    all_rows = load_rows(args.csv)
    rows = ok_rows(all_rows)
    n_failed = sum(1 for r in all_rows if r['status'] != 'ok')
    print(f"[INFO] Loaded {len(rows)} successful runs ({n_failed} failed) from '{args.csv}'")

    if not rows:
        print("[ERROR] No successful runs to plot.")
        return

    best_pb = plot_scaling(rows, args.outdir, all_rows=all_rows)
    plot_scaling_per_param_bound(rows, args.outdir, all_rows=all_rows)
    plot_heatmap(rows, args.outdir)
    plot_compute_scaling(rows, args.outdir)
    plot_convergence_and_valid_fraction(rows, best_pb, args.runs_dir, args.outdir)


if __name__ == '__main__':
    main()
