"""Compare sweep_simple (rb=sqrt(pb)) vs sweep_rb1 (rb=1, matched maxiter/n_train).

Shows mean NLL ± 1σ vs param_bound for nh=2 and nh=3.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTFILE_NLL   = 'sweep_comparison_nll.pdf'
OUTFILE_DELTA = 'sweep_comparison_delta.pdf'
OUTFILE_TIME  = 'sweep_comparison_time.pdf'
NLL_CAP = 1e6   # drop NaN-penalty / diverged runs

TAB10 = plt.get_cmap('tab10')
COLOR_SIMPLE = TAB10(0)   # blue
COLOR_RB1    = TAB10(1)   # orange


def load(path, nh_vals=(2, 3)):
    df = pd.read_csv(path)
    df = df[(df['status'] == 'ok') & (df['val_nll'] < NLL_CAP) & (df['time_sec']<2000)]
    df = df[df['nh'].isin(nh_vals)]
    return df


def summarize(df):
    return df.groupby(['nh', 'param_bound']).agg(
        mean_nll=('val_nll', 'mean'),
        std_nll=('val_nll', 'std'),
        n=('val_nll', 'count'),
    ).reset_index()


def plot_delta_nll(ax, summary, color, label, ls='-'):
    pb       = summary['param_bound'].values
    mean     = summary['mean_nll'].values
    std      = summary['std_nll'].values
    baseline = mean[0]
    baseline = 5.0
    delta    = mean - baseline
    ax.plot(pb, delta, color=color, lw=2, ls=ls, marker='o', ms=4, label=label)
    ax.fill_between(pb, delta - std, delta + std, color=color, alpha=0.2)
    ax.axhline(0, color='black', lw=0.8, linestyle=':')


def summarize_time(df):
    return df.groupby(['nh', 'param_bound']).agg(
        mean_t=('time_sec', 'mean'),
        std_t=('time_sec', 'std'),
    ).reset_index()


def plot_time(ax, sm, color_simple, rb, color_rb1):
    for data, color, label, ls in [
        (sm, color_simple, r'$r_b = \sqrt{p_b}$', '-'),
        (rb, color_rb1,    r'$r_b = 1$',           '--'),
    ]:
        pb   = data['param_bound'].values
        mean = data['mean_t'].values
        std  = data['std_t'].values
        ax.plot(pb, mean, color=color, lw=2, marker='o', ms=4, ls=ls, label=label)
        ax.fill_between(pb, np.maximum(mean - std, 0), mean + std, color=color, alpha=0.2)
    ax.legend()


def plot_nll(ax, summary, color, label, ls='-'):
    pb   = summary['param_bound'].values
    mean = summary['mean_nll'].values
    std  = summary['std_nll'].values
    ax.plot(pb, mean, color=color, lw=2, ls=ls, label=label, marker='o', ms=4)
    ax.fill_between(pb, mean - std, mean + std, color=color, alpha=0.2)


def main():
    df_simple = load('sweep_simple.csv')
    df_rb1    = load('sweep_rb1.csv')

    sm_s = summarize(df_simple)
    rb_s = summarize(df_rb1)

    sm_t = summarize_time(df_simple)
    rb_t = summarize_time(df_rb1)

    plt.rcParams.update({
        'axes.titlesize':  15,
        'axes.labelsize':  13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
    })

    nh_vals = [2, 3]

    # --- Figure 1: Mean NLL ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for col, nh in enumerate(nh_vals):
        ax = axes[col]
        plot_nll(ax, sm_s[sm_s['nh'] == nh], COLOR_SIMPLE,
                 r'$r_b = \sqrt{p_b}$')
        plot_nll(ax, rb_s[rb_s['nh'] == nh], COLOR_RB1,
                 r'$r_b = 1$', ls='--')
        ax.set_title(fr'$N_h = {nh}$')
        ax.set_xlabel(r'param_bound $p_b$')
        ax.set_ylabel(r'Mean NLL $\pm\,1\sigma$')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(left=0.5)
    plt.tight_layout()
    plt.savefig(OUTFILE_NLL, dpi=300)
    plt.close()
    print(f'[PLOT] {OUTFILE_NLL}')

    # --- Figure 2: ΔNLL ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    pass
    for col, nh in enumerate(nh_vals):
        ax = axes[col]
        plot_delta_nll(ax, sm_s[sm_s['nh'] == nh], COLOR_SIMPLE,
                       r'$r_b = \sqrt{p_b}$')
        plot_delta_nll(ax, rb_s[rb_s['nh'] == nh], COLOR_RB1,
                       r'$r_b = 1$', ls='--')
        ax.set_title(fr'$N_h = {nh}$')
        ax.set_xlabel(r'param_bound $p_b$')
        ax.set_ylabel(r'$\Delta$NLL (rel. to NLL$_0=5.0$)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(left=0.5)
    plt.tight_layout()
    plt.savefig(OUTFILE_DELTA, dpi=300)
    plt.close()
    print(f'[PLOT] {OUTFILE_DELTA}')

    # --- Figure 3: Training time ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    pass
    for col, nh in enumerate(nh_vals):
        ax = axes[col]
        plot_time(ax, sm_t[sm_t['nh'] == nh], COLOR_SIMPLE,
                      rb_t[rb_t['nh'] == nh], COLOR_RB1)
        ax.set_title(fr'$N_h = {nh}$')
        ax.set_xlabel(r'param_bound $p_b$')
        ax.set_ylabel('Mean training time (s)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(left=0.5)
    plt.tight_layout()
    plt.savefig(OUTFILE_TIME, dpi=300)
    plt.close()
    print(f'[PLOT] {OUTFILE_TIME}')

    # also print best per (scheme, nh)
    print('\nBest mean NLL per scheme and nh:')
    for label, s in [('sweep_simple', sm_s), ('sweep_rb1', rb_s)]:
        for nh in nh_vals:
            sub = s[s['nh'] == nh]
            best = sub.loc[sub['mean_nll'].idxmin()]
            print(f'  {label}  nh={nh}  pb={best.param_bound:.0f}'
                  f'  mean_nll={best.mean_nll:.3f}  std={best.std_nll:.3f}')


if __name__ == '__main__':
    main()