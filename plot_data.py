import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

INPUTDIR = './datasets'
OUTDIR = './plots/datasets'
os.makedirs(OUTDIR,exist_ok=True)

labels = ['x','E_cone','f_had','eta']
colors = ['red','orange','blue','cyan']

def plot(PROC,OUTDIR,title):

  FILE = os.path.join(INPUTDIR,f'{PROC}.npy')
  data = np.load(FILE)

  fig, ax = plt.subplots(2, 2, figsize=(8,8))
  fig.suptitle(title,fontweight='bold')
  ax[0,0].hist(data[:,0],color=colors[0],label=labels[0],bins=30)
  ax[0,0].set(title=r'Visible fraction $x_{vis} = E_{track} / E_\tau$',xlabel=r'$x_{vis}$',ylabel='Events')
  ax[0,1].hist(data[:,1],range=[0.0,1.0],color=colors[1],label=labels[1],bins=30)
  ax[0,1].set(title=r'Track isolation in $\Delta R<0.4$',xlabel=r'$\Sigma E_{ph} / E_{track}$ (EFlowPhotons)',ylabel='Events')
  ax[1,0].hist(data[:,2],color=colors[2],label=labels[2],bins=30)
  ax[1,0].set(title=r'Fraction of HCAL energy in $\Delta R<0.2$',xlabel=r'$f_{had} = E_{HCAL} / E_{TOT}$',ylabel='Events')
  ax[1,1].hist(data[:,3],color=colors[3],label=labels[3],bins=30)
  ax[1,1].set(title=r'Pseudorapidity $\eta$',xlabel=r'$\eta_{track}$',ylabel='Events')

  fig.tight_layout()

  outfile = os.path.join(OUTDIR,f'{PROC}.png')

  plt.savefig(outfile)
  print(f'[PLOT] saved file as {outfile}')


# MAIN
if __name__ == "__main__":
  #plot('pi_LH',OUTDIR,r'$\tau \to \pi\nu$ LH decays')
  #plot('pi_RH',OUTDIR,r'$\tau \to \pi\nu$ RH decays')
  #plot('rho_LH',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ LH decays')
  #plot('rho_RH',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ RH decays')
  #plot('pi',OUTDIR,r'$\tau \to \pi\nu$ Standard Model mixture')
  #plot('rho',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ Standard Model mixture')

  # plot differences
  pi_file  = os.path.join(INPUTDIR,'pi.npy')
  rho_file = os.path.join(INPUTDIR,'rho.npy')
  pi  = np.load(pi_file)
  rho = np.load(rho_file)
  pi[:, 1] = np.clip(pi[:,  1], 0.0, 1.0)
  rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
  fig, ax = plt.subplots(2, 2, figsize=(8,8))
  fig.suptitle(r'$\pi$ VS $\rho$ Standar-Model like datasets',fontweight='bold')
  ax[0,0].hist(pi[:,0] ,histtype='step', linewidth=1.5,label='PI', bins=30)
  ax[0,0].hist(rho[:,0],histtype='step', linewidth=1.5,label='RHO',bins=30)
  ax[0,0].set(title=r'Visible fraction $x_{vis} = E_{track} / E_\tau$',xlabel=r'$x_{vis}$',ylabel='Events')
  ax[0,1].hist(pi[:,1] ,range=[0.0,1.0],histtype='step', linewidth=1.5,label='PI', bins=30)
  ax[0,1].hist(rho[:,1],range=[0.0,1.0],histtype='step', linewidth=1.5,label='RHO',bins=30)
  ax[0,1].set(title=r'Track isolation in $\Delta R<0.4$',xlabel=r'$\Sigma E_{ph} / E_{track}$ (EFlowPhotons)',ylabel='Events')
  ax[1,0].hist(pi[:,2] ,histtype='step', linewidth=1.5,label='PI', bins=30)
  ax[1,0].hist(rho[:,2],histtype='step', linewidth=1.5,label='RHO',bins=30)
  ax[1,0].set(title=r'Fraction of HCAL energy in $\Delta R<0.2$',xlabel=r'$f_{had} = E_{HCAL} / E_{TOT}$',ylabel='Events')
  ax[1,1].hist(pi[:,3] ,histtype='step', linewidth=1.5,label='PI', bins=30)
  ax[1,1].hist(rho[:,3],histtype='step', linewidth=1.5,label='RHO',bins=30)
  ax[1,1].set(title=r'Pseudorapidity $\eta$',xlabel=r'$\eta_{track}$',ylabel='Events')

  plt.legend()
  fig.tight_layout()

  outfile = os.path.join(OUTDIR,'diff.png')
  plt.savefig(outfile)
  print(f'[PLOT] saved file as {outfile}')

  # clip Iso to [0,1] — same as training preprocessing; raw values reach ~60
  # and compress all π vs ρ separation into an invisible sliver on the left
  pi_vis  = pi.copy();  pi_vis[:,  1] = np.clip(pi_vis[:,  1], 0.0, 1.0)
  rho_vis = rho.copy(); rho_vis[:, 1] = np.clip(rho_vis[:, 1], 0.0, 1.0)

  # correlation heatmaps — justify non-diagonal T
  feat_labels = [r'$x_\mathrm{vis}$', r'$\mathrm{Iso}$', r'$f_\mathrm{had}$', r'$\eta$']
  fig, axes = plt.subplots(1, 2, figsize=(8, 4))
  for ax, data, title in zip(axes,
                              [pi_vis, rho_vis],
                              [r'$\pi$ signal', r'$\rho$ background']):
      C = pd.DataFrame(data, columns=feat_labels).corr()
      sns.heatmap(C, ax=ax, annot=True, fmt='.2f', vmin=-1, vmax=1,
                  cmap='RdBu_r', square=True, cbar=ax is axes[1])
      ax.set_title(title, fontsize=12)
  fig.suptitle('Feature correlation matrices (SM mixture)', fontweight='bold')
  fig.tight_layout()
  outfile = os.path.join(OUTDIR, 'correlations.png')
  plt.savefig(outfile, dpi=150)
  print(f'[PLOT] saved file as {outfile}')

  # pairplot — 2D joint distributions for pi vs rho
  rng = np.random.default_rng(42)
  n_plot = 5000   # subsample for readability
  pi_idx  = rng.choice(len(pi),  n_plot, replace=False)
  rho_idx = rng.choice(len(rho), n_plot, replace=False)
  pi_df  = pd.DataFrame(pi_vis[pi_idx],   columns=feat_labels)
  rho_df = pd.DataFrame(rho_vis[rho_idx], columns=feat_labels)
  pi_df['process']  = r'$\pi$ signal'
  rho_df['process'] = r'$\rho$ background'
  # rho first so pi (blue) is drawn on top
  combined = pd.concat([rho_df, pi_df], ignore_index=True)
  g = sns.pairplot(combined, hue='process', vars=feat_labels,
                   hue_order=[r'$\rho$ background', r'$\pi$ signal'],
                   plot_kws={'alpha': 0.3, 's': 4},
                   diag_kind='kde')
  for ax in g.axes.flatten():
      ax.xaxis.label.set_size(14)
      ax.yaxis.label.set_size(14)
      ax.tick_params(labelsize=11)
  if g.legend is not None:
      from matplotlib.collections import PathCollection
      from matplotlib.lines import Line2D
      labels_leg = [t.get_text() for t in g.legend.get_texts()]
      title_leg  = g.legend.get_title().get_text()
      colors = []
      for h in g.legend.legend_handles:
          if isinstance(h, PathCollection) and len(h.get_facecolor()) > 0:
              colors.append(h.get_facecolor()[0])
          elif isinstance(h, Line2D):
              colors.append(h.get_color())
          else:
              colors.append('gray')
      g.legend.remove()
      new_handles = [plt.scatter([], [], s=80, color=c) for c in colors]
      leg = g.figure.legend(new_handles, labels_leg, title=title_leg,
                            title_fontsize=13, fontsize=12, loc='upper right')
  g.figure.suptitle(r'Pairwise feature distributions: $\pi$ vs $\rho$ (SM mixture)',
                    y=1.01, fontweight='bold', fontsize=14)
  outfile = os.path.join(OUTDIR, 'pairplot.png')
  g.figure.savefig(outfile, dpi=150, bbox_inches='tight')
  print(f'[PLOT] saved file as {outfile}')
