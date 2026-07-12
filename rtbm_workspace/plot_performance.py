"""Generate the performance.md Section 6 plot catalogue from sweep.py output.

Reads the CSV written by sweep.py and the per-run .npy diagnostics in
--runs_dir, producing:
    scaling_nh.png             AUC / val_NLL vs N_hidden, mean ± std over seeds,
                               each N_hidden evaluated at its own best param_bound
    scaling_nh_pb<v>.png       same, param_bound held FIXED across all N_hidden
    heatmap_pb_nh.png          mean val_NLL over the (N_hidden, param_bound) grid
    convergence_overlay.png    CMA best-NLL vs fevals, one curve per N_hidden
    valid_fraction.png         fraction of non-penalty CMA candidates per iteration
    compute_scaling.png        wall-clock time / ncores vs N_hidden

Usage:
    python plot_performance.py --csv sweep_results.csv --runs_dir sweep_runs \\
        --outdir sweep_plots
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rtbmlib import run_tag

# val_NLL above this is the post-training LinAlgError fallback sentinel
# (math.md Section 7). Excluded from quality aggregates; crash rate reported separately.
CRASH_THRESHOLD = 1e8


def load_csv(csv_path):
    return pd.read_csv(csv_path)


def best_pb_per_nh(df):
    """For each nh, the param_bound with lowest mean val_NLL among non-crashed ok seeds."""
    good = df[(df['status'] == 'ok') & (df['val_nll'] < CRASH_THRESHOLD)]
    if good.empty:
        return {}
    mean_nll = good.groupby(['nh', 'param_bound'])['val_nll'].mean().reset_index()
    idx = mean_nll.groupby('nh')['val_nll'].idxmin()
    return mean_nll.loc[idx].set_index('nh')['param_bound'].to_dict()


def _quality_by_nh(df, nh_to_pb):
    """Aggregate NLL and AUC mean ± std for each (nh, pb) pair in nh_to_pb."""
    records = []
    for nh, pb in sorted(nh_to_pb.items()):
        sel  = df[(df['nh'] == nh) & (df['param_bound'] == pb)]
        ok   = sel[sel['status'] == 'ok']
        good = ok[ok['val_nll'] < CRASH_THRESHOLD]

        n_attempted = len(sel)
        n_timeout   = len(sel) - len(ok)
        n_crash     = len(ok)  - len(good)
        label       = f"N_h={int(nh)} (pb={pb:.2g})"

        if n_crash or n_timeout:
            print(f"  [WARN] {label}: {n_attempted} attempted, "
                  f"{n_timeout} timeout/init-fail, {n_crash} post-training crash "
                  f"(math.md Sec.7) -- {len(good)} usable")
        if len(good) < 2:
            print(f"  [WARN] {label}: only {len(good)} usable seed(s) "
                  f"-- error bar is not a real uncertainty estimate")

        records.append(dict(
            nh       = nh,
            pb       = pb,
            nll_mean = good['val_nll'].mean() if len(good) else np.nan,
            nll_std  = good['val_nll'].std()  if len(good) >= 2 else 0.0,
            auc_mean = good['auc'].mean()     if len(good) else np.nan,
            auc_std  = good['auc'].std()      if len(good) >= 2 else 0.0,
            n        = len(good),
        ))
    return pd.DataFrame(records)


def _plot_quality_panels(agg, suptitle, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    nh = agg['nh'].tolist()

    ax1.errorbar(nh, agg['nll_mean'], yerr=agg['nll_std'], marker='o', capsize=4, color='steelblue')
    for _, row in agg.iterrows():
        ax1.annotate(f"n={int(row['n'])}", (row['nh'], row['nll_mean']),
                     textcoords='offset points', xytext=(8, 8), fontsize=8)
    ax1.set(xlabel='$N_h$', ylabel='Validation NLL / event', title='NLL vs hidden units')
    ax1.set_xticks(nh); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.errorbar(nh, agg['auc_mean'], yerr=agg['auc_std'], marker='o', capsize=4, color='darkorange')
    for _, row in agg.iterrows():
        ax2.annotate(f"n={int(row['n'])}", (row['nh'], row['auc_mean']),
                     textcoords='offset points', xytext=(8, 8), fontsize=8)
    ax2.set(xlabel='$N_h$', ylabel='AUC', title='AUC vs hidden units')
    ax2.set_xticks(nh); ax2.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(suptitle, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_scaling(df, outdir):
    best_pb = best_pb_per_nh(df)
    agg = _quality_by_nh(df, best_pb)
    _plot_quality_panels(
        agg,
        r'RTBM Scaling with $N_h$ (best param_bound, $\mu \pm \sigma$)',
        os.path.join(outdir, 'scaling_nh.png'),
    )
    return best_pb


def plot_scaling_per_param_bound(df, outdir):
    ok = df[df['status'] == 'ok']
    for pb in sorted(ok['param_bound'].unique()):
        nhs = sorted(ok.loc[ok['param_bound'] == pb, 'nh'].unique())
        if not nhs:
            continue
        agg = _quality_by_nh(df, {nh: pb for nh in nhs})
        tag = f'{pb:.3g}'.replace('.', 'p')
        _plot_quality_panels(
            agg,
            f'RTBM Scaling with $N_h$ at fixed param_bound={pb:.2g} ($\\mu \\pm \\sigma$)',
            os.path.join(outdir, f'scaling_nh_pb{tag}.png'),
        )


def plot_heatmap(df, outdir):
    ok    = df[(df['status'] == 'ok') & (df['param_bound'] < 15)]
    good  = ok[ok['val_nll'] < CRASH_THRESHOLD]
    if good.empty:
        print("[WARN] No non-crashed successful runs with param_bound > 15; skipping heatmap.")
        return
    grid  = good.pivot_table(values='val_nll', index='nh', columns='param_bound', aggfunc='mean')

    n_ok    = ok.groupby(['nh', 'param_bound']).size().unstack(fill_value=0)
    n_crash = (ok[ok['val_nll'] >= CRASH_THRESHOLD]
                 .groupby(['nh', 'param_bound']).size()
                 .unstack(fill_value=0))

    nh_vals = grid.index.tolist()
    pb_vals = grid.columns.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(grid.values, aspect='auto', cmap='viridis_r', origin='lower')
    plt.colorbar(im, ax=ax, label='Mean validation NLL (crashed seeds excluded)')

    for i, nh in enumerate(nh_vals):
        for j, pb in enumerate(pb_vals):
            nc = n_crash.at[nh, pb] if (nh in n_crash.index and pb in n_crash.columns) else 0
            nt = n_ok.at[nh, pb]    if (nh in n_ok.index    and pb in n_ok.columns)    else 0
            if nc > 0:
                ax.text(j, i, f'{int(nc)}/{int(nt)}\ncrashed',
                        ha='center', va='center', fontsize=7, color='white')

    ax.set_xticks(range(len(pb_vals))); ax.set_xticklabels([f'{pb:.2g}' for pb in pb_vals])
    ax.set_yticks(range(len(nh_vals))); ax.set_yticklabels([str(int(nh)) for nh in nh_vals])
    ax.set_xlabel('param_bound'); ax.set_ylabel('$N_h$')
    ax.set_title(r'Validation NLL across (param_bound, $N_h$)', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, 'heatmap_pb_nh.png')
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_compute_scaling(df, outdir):
    ok = df[(df['status'] == 'ok') & (df['ncores'] > 0)].copy()
    ok['time_per_core'] = ok['time_sec'] / ok['ncores']
    agg = (ok.groupby('nh')['time_per_core']
             .agg(['mean', 'std', 'count'])
             .reset_index()
             .rename(columns={'count': 'n'}))

    for _, row in agg.iterrows():
        if row['n'] < 2:
            print(f"  [WARN] N_h={int(row['nh'])}: only {int(row['n'])} timed run(s) "
                  f"-- error bar is not a real uncertainty estimate")

    plt.figure(figsize=(7, 5))
    plt.errorbar(agg['nh'], agg['mean'], yerr=agg['std'], marker='o', capsize=4, color='seagreen')
    for _, row in agg.iterrows():
        plt.annotate(f"n={int(row['n'])}", (row['nh'], row['mean']),
                     textcoords='offset points', xytext=(8, 8), fontsize=8)
    plt.xlabel('$N_h$'); plt.ylabel('Wall-clock time / ncores (s)')
    plt.xticks(agg['nh'].tolist())
    plt.title('Compute Cost vs Hidden Units', fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, 'compute_scaling.png')
    plt.savefig(path, dpi=300); plt.close()
    print(f"[PLOT] {path}")


def plot_convergence_and_valid_fraction(df, best_pb, runs_dir, outdir):
    ok  = df[df['status'] == 'ok']
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for nh in sorted(best_pb):
        pb  = best_pb[nh]
        sel = ok[(ok['nh'] == nh) & (ok['param_bound'] == pb)]
        if sel.empty:
            continue
        seed    = int(sel.iloc[0]['seed'])
        popsize = int(sel.iloc[0]['popsize'])
        tag     = run_tag(int(nh), pb, seed)

        hist_path = os.path.join(runs_dir, f'{tag}_history.npy')
        vf_path   = os.path.join(runs_dir, f'{tag}_validfrac.npy')
        if not (os.path.exists(hist_path) and os.path.exists(vf_path)):
            continue

        history = np.load(hist_path)
        vf      = np.load(vf_path)
        fevals  = popsize * np.arange(1, len(history) + 1)

        ax1.plot(fevals, history, label=f'$N_h$={int(nh)} (pb={pb:.2g})')
        ax2.plot(np.arange(1, len(vf) + 1), vf, label=f'$N_h$={int(nh)} (pb={pb:.2g})')

    ax1.set(xlabel='CMA-ES function evaluations', ylabel='Best NLL',
            title='Convergence vs fevals')
    ax1.set_yscale('log'); ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)
    fig1.tight_layout()
    path1 = os.path.join(outdir, 'convergence_overlay.png')
    fig1.savefig(path1, dpi=300); plt.close(fig1)
    print(f"[PLOT] {path1}")

    ax2.set(xlabel='CMA-ES iteration', ylabel='Fraction of valid candidates',
            title=r'Schur-constraint difficulty per $N_h$')
    ax2.set_ylim(-0.05, 1.05); ax2.legend(); ax2.grid(True, linestyle=':', alpha=0.6)
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
    df = load_csv(args.csv)
    n_failed = (df['status'] != 'ok').sum()
    print(f"[INFO] Loaded {len(df)} runs ({n_failed} failed) from '{args.csv}'")

    if df[df['status'] == 'ok'].empty:
        print("[ERROR] No successful runs to plot.")
        return

    best_pb = plot_scaling(df, args.outdir)
    plot_scaling_per_param_bound(df, args.outdir)
    plot_heatmap(df, args.outdir)
    plot_compute_scaling(df, args.outdir)
    plot_convergence_and_valid_fraction(df, best_pb, args.runs_dir, args.outdir)


if __name__ == '__main__':
    main()