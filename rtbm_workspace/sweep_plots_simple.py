"""Plots for sweep_simple.csv.

Run from rtbm_workspace/:
    python sweep_plots_simple.py [--csv sweep_simple.csv] [--outdir sweep_simple_plots]
"""
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load(path):
    df = pd.read_csv(path)
    ok = df[df['status'] == 'ok'].copy()
    ok = ok[ok['val_nll'] < 1e8]
    print(f"[INFO] {len(ok)}/{len(df)} successful runs  "
          f"| N_h: {sorted(ok['nh'].unique())}  "
          f"| pb: {sorted(ok['param_bound'].unique())}")
    return ok


def plot_heatmap(df, metric, title, cmap, cbar_label, outdir):
    pivot = df.pivot_table(values=metric, index='nh', columns='param_bound', aggfunc='mean').sort_index(ascending=False)
    _, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.7), 4))
    sns.heatmap(pivot, annot=False, fmt='.3f', cmap=cmap,
                cbar_kws={'label': cbar_label}, ax=ax)
    ax.set_xlabel('param_bound')
    ax.set_ylabel('$N_h$')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(outdir, f'heatmap_{metric}.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"[PLOT] {path}")


def plot_time(df, outdir):
    d = df.copy()
    d = d[d['time_sec']<2000]
    d['nh_label'] = d['nh'].apply(lambda v: f'$N_h={int(v)}$')
    order = [f'$N_h={int(nh)}$' for nh in sorted(d['nh'].unique())]
    _, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=d, x='nh_label', y='time_sec', order=order,
                   inner='point', ax=ax, palette='tab10', hue='nh_label',legend=True)
    ax.set_xlabel('$N_h$')
    ax.set_ylabel('Training time [s] (8 CPU cores)')
    ax.set_title('Training time vs $N_h$ (250 CMA iterations)', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    plt.tight_layout()
    path = os.path.join(outdir, 'time_vs_nh.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"[PLOT] {path}")


def plot_auc_vs_pb(df, outdir):
    agg = df.groupby(['nh', 'param_bound'])['auc'].agg(['mean', 'std']).reset_index()
    
    _, ax = plt.subplots(figsize=(8, 5))
    for nh, grp in agg.groupby('nh'):
        ax.errorbar(grp['param_bound'], grp['mean'], yerr=grp['std'],
                    marker='o', capsize=4, label=f'$N_h={nh}$')
    ax.set_xlabel('param_bound')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs param_bound', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, 'auc_vs_pb.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"[PLOT] {path}")


def plot_violin(df, metric, ylabel, title, outdir):
    df = df.copy()
    df['nh_label'] = df['nh'].apply(lambda v: f'$N_h={int(v)}$')
    order = [f'$N_h={int(nh)}$' for nh in sorted(df['nh'].unique())]
    _, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df, x='nh_label', y=metric, order=order, inner='point', ax=ax,
                    palette='tab10', hue='nh_label', legend=True)
    ax.set_xlabel('$N_h$')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, f'violin_{metric}.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"[PLOT] {path}")


def plot_time_vs_pb(df, outdir):
    df = df[df['time_sec'] < 2000]
    palette = sns.color_palette('tab10', n_colors=len(df['nh'].unique()))
    color_map = {nh: palette[i] for i, nh in enumerate(sorted(df['nh'].unique()))}
    _, ax = plt.subplots(figsize=(8, 5))
    for nh, grp in df.groupby('nh'):
        ax.scatter(grp['param_bound'], grp['time_sec'],
                   color=color_map[nh], label=f'$N_h={nh}$',
                   alpha=0.8, s=40, edgecolors='grey', linewidths=0.4)
    ax.set_xlabel('param_bound')
    ax.set_ylabel('Training time [s] (8 CPU cores)')
    ax.set_title('Training time vs param_bound (250 CMA iterations)', fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(outdir, 'time_vs_pb_scatter.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"[PLOT] {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',    default='sweep_simple.csv')
    p.add_argument('--outdir', default='sweep_simple_plots')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.csv)

    plot_heatmap(df, 'val_nll', 'RTBM: Mean VAL NLL - 5 runs , 250 maxiter', 'viridis_r', 'NLL / event', args.outdir)
    plot_heatmap(df, 'auc',     'RTBM: Mean AUC - 5 runs , 250 maxiter', 'viridis',   'AUC',         args.outdir)
    plot_time(df, args.outdir)
    plot_time_vs_pb(df, args.outdir)
    plot_auc_vs_pb(df, args.outdir)
    plot_violin(df, 'auc',     'AUC',         'RTBM: AUC distribution per $N_h$',  args.outdir)
    plot_violin(df, 'val_nll', 'NLL / event', 'RTBM: NLL distribution per $N_h$',  args.outdir)


if __name__ == '__main__':
    main()