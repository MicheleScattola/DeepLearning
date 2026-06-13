import numpy as np

# Set random seed for perfect scientific reproducibility
np.random.seed(42)

# Target Standard Model polarization parameters
P_tau = -0.147
f_RH = (1 + P_tau) / 2.0
f_LH = 1.0 - f_RH

print(f"Target Fractions for P_tau = {P_tau}:")
print(f"  -> Left-Handed (LH):  {f_LH:.4f} ({f_LH*100:.2f}%)")
print(f"  -> Right-Handed (RH): {f_RH:.4f} ({f_RH*100:.2f}%)")

# --- 1. Load the Pure Datasets ---
print("\nLoading native numpy arrays...")
pi_LH = np.load("../datasets/pi_LH.npy")
pi_RH = np.load("../datasets/pi_RH.npy")
rho_LH = np.load("../datasets/rho_LH.npy")
rho_RH = np.load("../datasets/rho_RH.npy")

# --- 2. Determine Maximum Safe Dataset Size ---
# Find the limiting factor among your available datasets to prevent out-of-bounds sampling
max_pi_available = min(len(pi_LH) / f_LH, len(pi_RH) / f_RH)
max_rho_available = min(len(rho_LH) / f_LH, len(rho_RH) / f_RH)

total_events = 100000 

n_LH = int(total_events * f_LH)
n_RH = total_events - n_LH

print(f"\nSampling Breakdown for {total_events} Total Events:")
print(f"  -> Need {n_LH} Left-Handed events")
print(f"  -> Need {n_RH} Right-Handed events")

# --- 3. Extract and Mix Pions (pi) ---
print("\nMixing Pion populations...")
idx_pi_LH = np.random.choice(len(pi_LH), size=n_LH, replace=False)
idx_pi_RH = np.random.choice(len(pi_RH), size=n_RH, replace=False)

mixed_pi = np.concatenate([pi_LH[idx_pi_LH], pi_RH[idx_pi_RH]], axis=0)

# Shuffle the combined matrix so the model doesn't learn the block order
np.random.shuffle(mixed_pi)

# --- 4. Extract and Mix Rhos (rho) ---
print("Mixing Rho background populations...")
idx_rho_LH = np.random.choice(len(rho_LH), size=n_LH, replace=False)
idx_rho_RH = np.random.choice(len(rho_RH), size=n_RH, replace=False)

mixed_rho = np.concatenate([rho_LH[idx_rho_LH], rho_RH[idx_rho_RH]], axis=0)
np.random.shuffle(mixed_rho)

# --- 5. Save Pristine SM Mixed Datasets ---
np.save("../datasets/pi.npy", mixed_pi)
np.save("../datasets/rho.npy", mixed_rho)

print("\nProcessing Complete!")
print(f"Saved: ../datasets/pi.npy  shape={mixed_pi.shape}")
print(f"Saved: ../datasets/rho.npy shape={mixed_rho.shape}")