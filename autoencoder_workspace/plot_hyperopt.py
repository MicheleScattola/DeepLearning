"""Violin plots for the autoencoder Bayesian hyperparameter search.

Requires 'hyperopt_trials.pkl' produced by opt_training.py.
Run from autoencoder_workspace/:
    python plot_hyperopt.py
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# misc['vals'] stores each sampled value as a 1-element list.
# For hp.choice the value is the index into the choices array.
# For hp.loguniform the value is already the sampled float (not the log).
CHOICES = {
    'bottleneck': [2, 3],
    'dense_1':    [16, 32],
    'dense_2':    [8, 16],
    'activation': ['tanh', 'relu'],
}

def decode(vals):
    p = {'learning_rate': vals['learning_rate'][0]}
    for key, choices in CHOICES.items():
        p[key] = choices[vals[key][0]]
    return p

trials = pickle.load(open('hyperopt_trials.pkl', 'rb'))
ok     = [t for t in trials.trials if t['result']['status'] == 'ok']
params = [decode(t['misc']['vals']) for t in ok]
losses = [t['result']['loss'] for t in ok]
print(f"[INFO] Loaded {len(ok)} successful trials")

cat_params = {
    'bottleneck': ([2, 3],            'Bottleneck dim'),
    'dense_1':    ([16, 32],          'Dense layer 1'),
    'dense_2':    ([8, 16],           'Dense layer 2'),
    'activation': (['tanh', 'relu'],  'Activation'),
}

fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
fig.suptitle('Hyperopt search', fontweight='bold')

for i, (ax, (key, (choices, label))) in enumerate(zip(axes, cat_params.items())):
    groups     = [[losses[j] for j, p in enumerate(params) if p[key] == c] for c in choices]
    data       = [g for g in groups if g]
    tick_labels = [str(c) for c, g in zip(choices, groups) if g]
    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(tick_labels)
    if i == 0:
        ax.set_ylabel('Val Loss (MSE)')
    ax.set_xlabel(label)
    ax.set_yscale('log')
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
outfile = 'hyperopt_violin.png'
plt.savefig(outfile, dpi=300)
plt.close()
print(f"[PLOT] Saved '{outfile}'")