"""Violin plots for the RTBM Bayesian hyperparameter search.

One violin per N_h file, showing the NLL distribution across all
param_bound values tested.

Run from rtbm_workspace/:
    python plot_hyperopt.py \
        --nh2 training/opt_nh2_.../hyperopt_trials.pkl \
        --nh3 training/opt_nh3_.../hyperopt_trials.pkl
"""
import argparse
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CRASH_THRESHOLD = 1e8
PALETTE         = {2: 'steelblue', 3: 'darkorange'}


def load_trials(path, nh):
    trials = pickle.load(open(path, 'rb'))
    ok  = [t for t in trials.trials
           if t['result']['status'] == 'ok' and t['result']['loss'] < CRASH_THRESHOLD]
    nll = [t['result']['loss'] for t in ok]
    print(f"[INFO] N_h={nh}: {len(nll)} trials")
    return pd.DataFrame({'NLL': nll, 'nh': f'$N_h={nh}$'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--nh2', default='training/v_10k_40evals_nh2_pb47/hyperopt_trials.pkl')
    p.add_argument('--nh3', default='training/v_10k_40evals_nh3_pb47/hyperopt_trials.pkl')
    p.add_argument('--out', default='hyperopt_violin.png')
    args = p.parse_args()

    df = pd.concat([load_trials(args.nh2, 2), load_trials(args.nh3, 3)], ignore_index=True)

    palette = {f'$N_h={nh}$': color for nh, color in PALETTE.items()}
    order   = [f'$N_h={nh}$' for nh in sorted(PALETTE)]

    _, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(data=df, x='nh', y='NLL', order=order,
                   hue='nh', hue_order=order, palette=palette,
                   inner='point', legend=True, ax=ax)
    ax.set_xlabel('Hidden units $N_h$')
    ax.set_ylabel('NLL / event')
    ax.set_title(r'RTBM hyperopt: NLL vs $N_h$', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    #if (leg := ax.get_legend()):
        #leg.set(loc='upper center', ncols=2)
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved '{args.out}'")


if __name__ == '__main__':
    main()