from myFunctions import NegativeLogLikelihood,progress_monitor
import os 
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from theta.rtbm import RTBM
import theta.minimizer  
from theta.costfunctions import logarithmic, mse
#from theta.costfunctions import logarithmic
#from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
#from sklearn.metrics import roc_curve, auc

PHYS_CORES = multiprocessing.cpu_count()
PARALLEL_CORES = int(PHYS_CORES / 2)
ETA_MAX   = 2.5
N_VISIBLE = 4
np.random.seed(42) 

def load_datasets():
    pi  = np.load('../datasets/pi.npy')
    rho = np.load('../datasets/rho.npy')
    pi[:,  1] = np.clip(pi[:,  1], 0.0, 1.0)
    rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
    pi[:,  3] = (pi[:,  3] - ETA_MAX) / 5.0
    rho[:, 3] = (rho[:, 3] - ETA_MAX) / 5.0
    return pi, rho

#def train_rtbm(model,x_train,x_val,y_train,y_val):
  

if __name__ == '__main__':
  start = time.time()
  print("[INFO] Loading datasets...")
  pi_data, rho_data = load_datasets()
  np.random.shuffle(pi_data)
  split_index = int(0.05 * len(pi_data))
  x_train = pi_data[:split_index].copy().T
  x_val = pi_data[split_index:].copy().T

  model = RTBM(visible_units=N_VISIBLE,
                hidden_units=3,
                mode=RTBM.Mode.LogProbability,
                diagonal_T=True,
                init_max_param_bound=20.0,
                random_bound=0.5
                )

  minimizer = theta.minimizer.CMA(parallel=True,
                                  ncores=PARALLEL_CORES)

  # list for loss history
  manager = multiprocessing.Manager()
  shared_history = manager.list()
  MAX_ITER = 100
  POP_SIZE = 30
  TOTAL_EVALS = MAX_ITER * POP_SIZE
  opt_params = minimizer.train(
                  cost=NegativeLogLikelihood(shared_history),
                  model=model,
                  x_data=x_train,
                  y_data=None,
                  maxiter=150)

  print("\n[SUCCESS] Optimal parameters saved to 'best_pion_rtbm_params.npy'")
    
  # --- 6. Process and Plot the Loss History ---
  print("[INFO] Plotting training history...")
  raw_history = list(shared_history)
  popsize = 30

  # Group parallel evaluations by generation
  num_complete_gens = len(raw_history) // popsize
  clean_history = np.array(raw_history[:num_complete_gens * popsize])

  # Reshape to (Generations, Population Size)
  history_matrix = clean_history.reshape(num_complete_gens, popsize)

  # CMA-ES progresses based on the BEST clone in each generation
  best_loss_per_gen = np.min(history_matrix, axis=1)
  mean_loss_per_gen = np.mean(history_matrix, axis=1)

  plt.figure(figsize=(8, 5))
  plt.plot(best_loss_per_gen, label='Best Clone Loss (NLL)', color='blue', linewidth=2)
  plt.plot(mean_loss_per_gen, label='Population Mean Loss', color='orange', alpha=0.6)
  plt.title('RTBM Evolutionary Training History', fontweight='bold')
  plt.xlabel('Generation')
  plt.ylabel('Cost (Negative Log-Likelihood)')
  plt.legend(frameon=False)
  plt.grid(True, linestyle=':', alpha=0.6)

  outfile = 'rtbm_training_history.png'
  plt.tight_layout()
  plt.savefig('mytraining.png')
  end = time.time()
  print(f'Time passed: {end-start:.2f}')