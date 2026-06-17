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
  plot('pi_LH',OUTDIR,r'$\tau \to \pi\nu$ LH decays')
  plot('pi_RH',OUTDIR,r'$\tau \to \pi\nu$ RH decays')
  plot('rho_LH',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ LH decays')
  plot('rho_RH',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ RH decays')
  plot('pi',OUTDIR,r'$\tau \to \pi\nu$ Standard Model mixture')
  plot('rho',OUTDIR,r'$\tau \to \rho\nu \to \pi \pi^0 \nu$ Standard Model mixture')

  # plot differences
  pi_file  = os.path.join(INPUTDIR,'pi.npy')
  rho_file = os.path.join(INPUTDIR,'rho.npy')
  pi  = np.load(pi_file)
  rho = np.load(rho_file)
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
