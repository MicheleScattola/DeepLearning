"""Violin + scatter plots for training_simple.py --optimize hyperopt results.

Run from rtbm_workspace/:
    python plot_hyperopt_simple.py
    python plot_hyperopt_simple.py --pkl training/simple_training/hyperopt_trials.pkl
                                   --n_hidden 2 3 --outdir training/simple_training
"""
import os
import argparse
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_trials(pkl_path, n_hidden_choices):
    with open(pkl_path, 'rb') as f:
        trials = pickle.load(f)
    rows = []
    for t in trials.trials:
        if t['result']['status'] != 'ok':
            continue
        loss = t['result']['loss']
        if loss >= 1e8:
            continue
        vals = t['misc']['vals']
        nh = n_hidden_choices[vals['n_hidden'][0]]
        pb = vals['param_bound'][0]
        rows.append({'n_hidden': nh, 'param_bound': pb, 'val_nll': loss})
    df = pd.DataFrame(rows)
    print(f"[INFO] {len(df)} valid trials  | N_h: {sorted(df['n_hidden'].unique())}")
    return df


def plot(df, outdir):
    nhs = sorted(df['n_hidden'].unique())
    tab10 = sns.color_palette('tab10', n_colors=len(nhs))
    palette = {nh: tab10[i] for i, nh in enumerate(nhs)}

    df = df.copy()
    df['nh_label'] = df['n_hidden'].apply(lambda v: f'$N_h={int(v)}$')
    order = [f'$N_h={int(nh)}$' for nh in nhs]
    pal_label = {f'$N_h={int(nh)}$': palette[nh] for nh in nhs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RTBM hyperopt search', fontweight='bold')

    # Left: violin per N_h (marginal over param_bound)
    sns.violinplot(data=df, x='nh_label', y='val_nll', order=order,
                   hue='nh_label', hue_order=order, palette=pal_label,
                   inner='point', legend=False, ax=ax1)
    ax1.set_xlabel('$N_h$')
    ax1.set_ylabel('Val NLL / event')
    ax1.set_title('NLL distribution per $N_h$', fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Right: scatter param_bound vs val_NLL, colored by N_h
    for nh in nhs:
        sub = df[df['n_hidden'] == nh]
        ax2.scatter(sub['param_bound'], sub['val_nll'],
                    color=palette[nh], label=f'$N_h={int(nh)}$',
                    alpha=0.8, s=40, edgecolors='grey', linewidths=0.4)
    ax2.set_xlabel('param_bound')
    ax2.set_ylabel('Val NLL / event')
    ax2.set_title('NLL vs param_bound', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    path = os.path.join(outdir, 'hyperopt_simple.pdf')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pkl',      default='training/simple_training/hyperopt_trials.pkl')
    p.add_argument('--n_hidden', type=int, nargs='+', default=[2, 3],
                   help='Same n_hidden choices passed to training_simple.py --n_hidden')
    p.add_argument('--outdir',   default=None,
                   help='Output directory (default: same directory as pkl)')
    args = p.parse_args()

    outdir = args.outdir or os.path.dirname(os.path.abspath(args.pkl))
    os.makedirs(outdir, exist_ok=True)

    df = load_trials(args.pkl, args.n_hidden)
    plot(df, outdir)


if __name__ == '__main__':
    main()