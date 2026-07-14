"""Violin plots for the RTBM Bayesian hyperparameter search.

Expects two separate optimization runs with N_h fixed (one per pkl),
each searching only over param_bound. Produces one subplot per N_h,
showing val_NLL distribution binned by param_bound.

Run from rtbm_workspace/:
    python plot_hyperopt.py \
        --nh2 training/opt_nh2_.../hyperopt_trials.pkl \
        --nh3 training/opt_nh3_.../hyperopt_trials.pkl
"""
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CRASH_THRESHOLD = 1e8
N_BINS          = 4   # param_bound bins per subplot


def load_trials(path):
    trials = pickle.load(open(path, 'rb'))
    ok = [t for t in trials.trials
          if t['result']['status'] == 'ok' and t['result']['loss'] < CRASH_THRESHOLD]
    pb  = [t['misc']['vals']['param_bound'][0] for t in ok]
    nll = [t['result']['loss'] for t in ok]
    return np.array(pb), np.array(nll)


def violin_panel(ax, pb, nll, nh, color):
    edges  = np.linspace(pb.min(), pb.max(), N_BINS + 1)
    labels = [f'[{edges[i]:.1f}, {edges[i+1]:.1f}]' for i in range(N_BINS)]
    groups = [nll[(pb >= edges[i]) & (pb < edges[i + 1])] for i in range(N_BINS)]
    # last bin is closed on the right
    groups[-1] = nll[(pb >= edges[-2]) & (pb <= edges[-1])]

    data        = [g for g in groups if len(g) >= 2]
    tick_labels = [labels[i] for i, g in enumerate(groups) if len(g) >= 2]

    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(tick_labels, rotation=15, ha='right')

    n_removed = len(pb) - sum(len(g) for g in data)
    ax.set_title(f'$N_h$ = {nh}  (n={len(pb)}, {n_removed} sparse bins dropped)',
                 fontsize=10)
    ax.set_xlabel('param_bound range')
    ax.grid(True, linestyle=':', alpha=0.6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nh2', default='training/v_10k_40evals_nh2_pb47/hyperopt_trials.pkl')
    p.add_argument('--nh3', default='training/v_10k_40evals_nh3_pb47/hyperopt_trials.pkl')
    p.add_argument('--out', default='hyperopt_violin.png')
    args = p.parse_args()

    pb2, nll2 = load_trials(args.nh2)
    pb3, nll3 = load_trials(args.nh3)
    print(f"[INFO] N_h=2: {len(pb2)} trials | N_h=3: {len(pb3)} trials")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    fig.suptitle('RTBM: val NLL vs param_bound (Bayesian search)', fontweight='bold')

    violin_panel(ax1, pb2, nll2, nh=2, color='steelblue')
    violin_panel(ax2, pb3, nll3, nh=3, color='darkorange')
    ax1.set_ylabel('Val NLL / event')

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved '{args.out}'")


if __name__ == '__main__':
    main()