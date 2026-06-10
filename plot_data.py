import matplotlib.pyplot as plt
import numpy as np
import os

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
  ax[0,0].hist(data[:,0],color=colors[0],label=labels[0])
  ax[0,0].set(title=r'$x_{vis} = E_{track} / E_\tau$',xlabel='x',ylabel='Events')
  ax[0,1].hist(data[:,1],color=colors[1],label=labels[1])
  ax[0,1].set(title=r'$E_{cone}$',xlabel=r'$E_{cone}$ [GeV]',ylabel='Events')
  ax[1,0].hist(data[:,2],color=colors[2],label=labels[2])
  ax[1,0].set(title=r'$f_{had}$',xlabel=r'$f_{had} = E_{HCAL} / E_{TOT}$',ylabel='Events')
  ax[1,1].hist(data[:,3],color=colors[3],label=labels[3])
  ax[1,1].set(title=r'$\eta$',xlabel=r'$\eta_{track}$',ylabel='Events')

  fig.tight_layout()

  outfile = os.path.join(OUTDIR,f'{PROC}.png')

  plt.savefig(outfile)


# MAIN
if __name__ == "__main__":
  plot('pi_LH',OUTDIR,r'$\pi$ LH decays')
  plot('pi_RH',OUTDIR,r'$\pi$ RH decays')
  plot('rho_LH',OUTDIR,r'$\rho$ LH decays')
  plot('rho_RH',OUTDIR,r'$\rho$ RH decays')