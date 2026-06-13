import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure the freshly compiled theta library can be found locally
sys.path.append(os.path.abspath("./theta"))
from theta.rtbm import RTBM
from theta.minimizer import CMA
from theta.costfunctions import logarithmic

MAXITER = 280

# --- 1. Load Datasets ---
# Feature Order: [0='x', 1='E_cone', 2='f_had', 3='eta']
print("Loading phase-space matrices...")
X_signal_raw = np.load("../datasets/pi_LH.npy")
X_bkg_raw = np.load("../datasets/rho_LH.npy")

print(f"Loaded {X_signal_raw.shape[0]} Signal Events.")
print(f"Loaded {X_bkg_raw.shape[0]} Background Events.")

# --- 2. Train / Test Split ---
num_train = int(0.7 * len(X_signal_raw))
X_train = X_signal_raw[:num_train]
X_test_pion = X_signal_raw[num_train:]
X_test_rho = X_bkg_raw

# --- 3. Feature Standardization ---
print("Standardizing phase space...")
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0

X_train = (X_train - mu) / std
X_test_pion = (X_test_pion - mu) / std
X_test_rho = (X_test_rho - mu) / std

# CRITICAL GEOMETRIC TWEAK: 
# The theta framework requires the shape inverse: (N_features, N_events)
X_train_theta = X_train.T
X_test_pion_theta = X_test_pion.T
X_test_rho_theta = X_test_rho.T

# --- 4. Initialize the RTBM Model ---
n_visible = 4  # Your 4 physical phase space variables
n_hidden = 2   # Start small (2)! Parameter space size scales with n_hidden

print(f"Initializing RTBM ({n_visible} visible units, {n_hidden} hidden units)...")
model = RTBM(n_visible, n_hidden, init_max_param_bound=20, random_bound=1)

# --- 5. Run the Evolutionary Minimizer ---
print("Spawning CMA-ES evolutionary workers across all available CPU threads...")
minimizer = CMA(parallel=True,ncores=6)

# Minimize the negative log-likelihood over the clean pion dataset
solution = minimizer.train(logarithmic, model, X_train_theta, tolfun=1e-3,maxiter=MAXITER)
print("Optimization converged successfully! Model parameters updated.")

# --- 6. Extract Anomaly Scores ---
print("Evaluating event log-likelihoods across testing splits...")
# Extract probability density curves natively
# Low probability density = higher anomaly score
prob_pions = np.real(model(X_test_pion_theta))
prob_rho = np.real(model(X_test_rho_theta))

# Prevent log(0) runtime issues
prob_pions = np.clip(prob_pions, 1e-12, None)
prob_rho = np.clip(prob_rho, 1e-12, None)

anomaly_score_pions = -np.log(prob_pions).flatten()
anomaly_score_rho = -np.log(prob_rho).flatten()

# --- 7. Plot the Discrimination Results ---
plt.figure(figsize=(10, 6))
plt.hist(anomaly_score_pions, bins=100, range=(0, 15), alpha=0.5, density=True, label=r"True Pions ($\pi_{LH}$)", color="blue")
plt.hist(anomaly_score_rho, bins=100, range=(0, 15), alpha=0.5, density=True, label=r"Background ($\rho_{LH}$)", color="red")
plt.xlabel("RTBM Anomaly Score: $-\log(P(x))$")
plt.ylabel("Probability Density")
plt.title("Unsupervised Background Rejection via Riemann-Theta Energy Gradients")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

plt.savefig(f"rtbm_anomaly_separation_{MAXITER}.png", dpi=300)
print(f"Analysis complete! Plot saved as: rtbm_anomaly_separation_{MAXITER}.png")
