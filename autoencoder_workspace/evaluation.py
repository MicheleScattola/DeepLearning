import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import multiprocessing

tf.keras.utils.set_random_seed(42)
ETA_MAX = 2.5

def setupParallel():
  '''
  Limit Tensorflow processes to the number of physical CPU cores.
  '''
  physical_cores = multiprocessing.cpu_count()
  target = int(physical_cores/2)
  tf.config.threading.set_intra_op_parallelism_threads(target)
  tf.config.threading.set_inter_op_parallelism_threads(target)
  print(f"\n[SETUP] {physical_cores} total cores available. Tensorflow loaded on {target} physical cores.")


def loadDatasetMix():
  '''
  Load the mixed polarization dataset.
  Corresponds to the Standard Model polarization P ~ -0.147
  '''
  pi  = np.load('../datasets/pi.npy')
  rho = np.load('../datasets/rho.npy')
  pi[:, 1] = np.clip(pi[:, 1], 0.0, 1.0)
  rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
  pi[:, 3] = (pi[:, 3] - ETA_MAX) / 5.0
  rho[:, 3] = (rho[:, 3] - ETA_MAX) / 5.0
  return pi, rho

def calculate_anomaly_scores(model, true_data):
  """
  Passes data through the AE and calculates the MSE for each individual event.
  """
  reconstructions = model.predict(true_data, batch_size=512, verbose=0)
  mse_scores = np.mean(np.square(true_data - reconstructions), axis=1)
  return mse_scores

if __name__ == "__main__":
  setupParallel()

  print("[INFO] Loading datasets...")
  pi_data, rho_data = loadDatasetMix()
  np.random.shuffle(pi_data)

  #split_index = int(0.8 * len(pi_data))
  split_index = len(pi_data) - 100000
  training_pions = pi_data[:split_index].copy()
  validation_pions = pi_data[split_index:].copy()

  rho_data = rho_data[:100000]

  print("[INFO] Loading trained Autoencoder...")
  model = tf.keras.models.load_model("best_pion_autoencoder.keras")

  print(f"[INFO] Running on {len(validation_pions)} pions and {len(rho_data)} rho.")

  scores_pi = calculate_anomaly_scores(model, validation_pions)
  scores_rho = calculate_anomaly_scores(model, rho_data)

  # calculating threshold and working point
  threshold_95 = np.percentile(scores_pi, 95)
  
  bkg_rejected = np.sum(scores_rho > threshold_95)
  total_bkg = len(scores_rho)
  bkg_rejection_rate = bkg_rejected / total_bkg
  
  print("\n=======================================================")
  print(f"Target Signal Efficiency : 95.00%")
  print(f"Optimal MSE Threshold    : {threshold_95:.7e}")
  print(f"Background Rejection     : {bkg_rejection_rate * 100:.2f}%")
  print("=======================================================\n")

  plt.figure(figsize=(8, 6))
  log_bins = np.logspace(-9, 0, 100)
  
  weights_pi = np.ones_like(scores_pi) / len(scores_pi)
  weights_rho = np.ones_like(scores_rho) / len(scores_rho)
  
  plt.hist(scores_pi, bins=log_bins, weights=weights_pi, histtype='step', 
            linewidth=2, label=r'Signal: $\tau \to \pi\nu$', color='blue')
  plt.hist(scores_rho, bins=log_bins, weights=weights_rho, histtype='step', 
            linewidth=2, label=r'Background: $\tau \to \rho\nu$', color='red')

  pi_hist, _ = np.histogram(scores_pi, bins=log_bins, weights=weights_pi)
  rho_hist, _ = np.histogram(scores_rho, bins=log_bins, weights=weights_rho)
  bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])

  plt.fill_between(bin_centers, pi_hist, 0, where=(bin_centers <= threshold_95),
                   step='mid', color='blue', alpha=0.2)
  plt.fill_between(bin_centers, rho_hist, 0, where=(bin_centers >= threshold_95),
                   step='mid', color='red', alpha=0.2)
  plt.axvline(threshold_95, color='black', linestyle='--', linewidth=1.5,
              label=r'Threshold: 95% eff. in $\pi$ data')
  
  plt.title('Autoencoder: Anomaly Scores (MSE)', fontweight='bold')
  plt.xlabel('Mean Squared Error [log]')
  plt.ylabel('Fraction of Events')
  plt.xscale('log') 
  
  plt.legend(frameon=False)
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.tight_layout()
  
  hist_outfile = 'ae_anomaly_scores.png'
  plt.savefig(hist_outfile, dpi=300)
  plt.close()
  print(f"[PLOT] Saved histogram to '{hist_outfile}'")

  # labels for roc (0 for pi, 1 for rho)
  y_true = np.concatenate([np.zeros(len(scores_pi)), np.ones(len(scores_rho))])
  y_scores = np.concatenate([scores_pi, scores_rho])
  
  fpr, tpr, thresholds = roc_curve(y_true, y_scores)
  roc_auc = auc(fpr, tpr)
  
  plt.figure(figsize=(7, 7))
  plt.fill_between(fpr, tpr, color='darkorange', alpha=0.2)
  plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Autoencoder AUC = {roc_auc:.3f}')
  plt.plot([0, 1], [1, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('FPR (Signal Loss)')
  plt.ylabel('TPR (Background Rejection)')
  plt.title('ROC curve', fontweight='bold')
  plt.legend(loc="lower right", frameon=False)
  plt.grid(True, linestyle=':', alpha=0.6)
  plt.tight_layout()
  
  roc_outfile = 'ae_roc_curve.png'
  plt.savefig(roc_outfile, dpi=300)
  plt.close()
  print(f"[PLOT] Saved ROC curve to '{roc_outfile}'")
  
  print(f"\n[RESULT] Final Autoencoder AUC: {roc_auc:.4f}")