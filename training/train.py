import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Path Configuration & Data Loading ---
# Feature Order Reference: [0='x', 1='E_cone', 2='f_had', 3='eta']
print("Loading datasets from path: ../datasets/")

# Adjusting filenames to match your exact repository strings
# Note: We are starting with Left-Handed (LH) sets for this preliminary analysis.
# To analyze Right-Handed sets later, simply swap '_LH' with '_RH'
X_signal_raw = np.load("../datasets/pi_LH.npy")
X_bkg_raw = np.load("../datasets/rho_LH.npy")

print(f"Loaded {X_signal_raw.shape[0]} true pion events.")
print(f"Loaded {X_bkg_raw.shape[0]} background rho events.")

# --- 2. Train / Test Separation (Unsupervised Setup) ---
# We split our pure pion channel: 80% to train the model, 20% reserved for testing evaluation.
num_train = int(0.8 * len(X_signal_raw))
X_train = X_signal_raw[:num_train]
X_test_pion = X_signal_raw[num_train:]
X_test_bkg = X_bkg_raw  # Background is completely unseen during training

# --- 3. Feature Standardization (Fit ONLY on X_train) ---
print("Standardizing 4D feature spaces...")
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0  # Safe guard boundary to eliminate division by zero

# Apply transformation matrices symmetrically across all testing splits
X_train = (X_train - mu) / std
X_test_pion = (X_test_pion - mu) / std
X_test_bkg = (X_test_bkg - mu) / std

# Move our numpy arrays over to the GTX 1080 Ti CUDA environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_train = torch.tensor(X_train, dtype=torch.float32).to(device)
t_test_pion = torch.tensor(X_test_pion, dtype=torch.float32).to(device)
t_test_bkg = torch.tensor(X_test_bkg, dtype=torch.float32).to(device)

# --- 4. Autoencoder Structural Definition ---
class TauPhaseSpaceAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Compress the 4D space down to a 2D physical latent space
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        # Decompress 2D back to the original 4D representations
        self.decoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime

model = TauPhaseSpaceAE().to(device)
criterion = nn.MSELoss()  # Maps directly to the ||x - x'||^2 reconstruction metric
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 5. Acceleration Training Loop ---
print("Beginning training optimization on GTX 1080 Ti...")
epochs = 50
batch_size = 512

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(t_train.size(0))
    epoch_loss = 0.0
    
    for i in range(0, t_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = t_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * batch_x.size(0)
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{epochs} | Training MSE Loss: {epoch_loss / t_train.size(0):.6f}")

# --- 6. Reconstruction Loss Extraction (Anomaly Detection Evaluator) ---
model.eval()
with torch.no_grad():
    # Calculate unique event scores for our pure pion testing matrix
    pred_pion = model(t_test_pion)
    loss_pions = torch.mean((t_test_pion - pred_pion) ** 2, dim=1).cpu().numpy()
    
    # Calculate unique event scores for our contaminated background matrix
    pred_bkg = model(t_test_bkg)
    loss_bkg = torch.mean((t_test_bkg - pred_bkg) ** 2, dim=1).cpu().numpy()

# --- 7. Plot Performance Metric Results ---
print("Generating evaluation metrics...")
plt.figure(figsize=(10, 6))
plt.hist(loss_pions, bins=120, range=(0, 3), alpha=0.5, label=r"True Pions ($\pi_{LH}$)", color="blue", density=True)
plt.hist(loss_bkg, bins=120, range=(0, 3), alpha=0.5, label=r"Background ($\rho_{LH}$)", color="red", density=True)
plt.xlabel(r"Reconstruction Error Metric: $\|x - x'\|^2$ (Anomaly Score)")
plt.ylabel("Probability Density Distribution")
plt.title("Unsupervised Anomaly Separation Performance via Phase Space Autoencoder")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

output_plot = "preliminary_ae_separation.png"
plt.savefig(output_plot, dpi=300)
print(f"Analysis complete! Evaluation metrics visual plot saved to: {output_plot}")