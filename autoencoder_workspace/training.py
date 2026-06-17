import os
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import argparse
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

parser = argparse.ArgumentParser(
                    prog='ModelTraining',
                    description='Training for the encoder')

parser.add_argument('-o','--outfile',default='trained_pion_autoencoder') 
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-b', '--bottleneck', type=int, default=3)
parser.add_argument('-e', '--epochs', type=int, default=300)
parser.add_argument('-p','--patience', type=int, default=30)
parser.add_argument('--save', action='store_true')

# global setup
args = parser.parse_args()
OUTDIR = f'./training/{args.outfile}'
os.makedirs(OUTDIR,exist_ok=True)
with open(os.path.join(OUTDIR,'args.txt'), 'w') as f:
  for arg, value in vars(args).items():
    f.write(f"{arg}: {value}\n")

tf.keras.utils.set_random_seed(42) 
#tf.config.experimental.enable_op_determinism() 
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

def encoder(inputs):
  '''Compression from 4D data to 2D'''
  x = tf.keras.layers.Dense(32, activation='elu', name='enc_dense_1')(inputs)
  x = tf.keras.layers.Dense(16, activation='elu', name='enc_dense_2')(x)
  latent_space = tf.keras.layers.Dense(args.bottleneck, activation='linear', name='bottleneck')(x)
  return latent_space

def decoder(inputs):
  '''Reconstruction from 2D to 4D'''
  x = tf.keras.layers.Dense(16, activation='elu', name='dec_dense_1')(inputs)
  x = tf.keras.layers.Dense(32, activation='elu', name='dec_dense_2')(x)
  reconstruction = tf.keras.layers.Dense(4, activation='linear', name='reconstruction')(x)
  return reconstruction

def create_autoencoder():
  '''Assembles the keras model'''
  inputs = tf.keras.layers.Input(shape=(4,), name='phase_space_input')
  encoded_output = encoder(inputs)
  decoded_output = decoder(encoded_output)
  model = tf.keras.Model(inputs=inputs, outputs=decoded_output, name='pion_autoencoder')
  return model

def plot_history(history):
  '''Plots training VS validation datasets'''
  plt.figure(figsize=(8, 5))
  plt.plot(history.history['loss'], label='Training Loss (MSE)')
  plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
  plt.title('Autoencoder: training LOSS (MSE)',fontweight='bold')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Squared Error')
  plt.yscale('log')
  plt.legend(frameon=False)
  plt.grid(True, linestyle=':', alpha=0.6)
  
  outfile = os.path.join(OUTDIR,'history.png')
  plt.savefig(outfile, dpi=300)
  plt.close()
  print(f"[PLOT] Saved training history to '{outfile}'")
    
def check_reconstructions(model, val_data):
  '''Plots training VS validation to check overfitting in reconstruction'''
  print("\n[INFO] Generating reconstructions for physical validation...")
  predictions = model.predict(val_data, batch_size=256)
  
  labels = [r'$x_{vis}$', r'$Iso = \Sigma E_{ph} / E_{track}$', r'$f_{had} = E_{HCAL}/E_{tot}$', r'$\eta$ (Scaled)']
  fig, ax = plt.subplots(2, 2, figsize=(10, 8))
  fig.suptitle(r'Autoencoder: reconstruction of $\tau\to\pi\nu$', fontweight='bold')
  
  for i in range(4):
    row = i // 2
    col = i % 2
    ax[row, col].hist(val_data[:, i], bins=40, histtype='step', linewidth=1.5, label='Validation data')
    ax[row, col].hist(predictions[:, i], bins=40, histtype='step', linewidth=1.5, label='AE prediction')
    ax[row, col].set_title(labels[i])
    ax[row, col].legend()
    
  plt.tight_layout()
  
  outfile = os.path.join(OUTDIR,'reconstruction.png')
  plt.savefig(outfile, dpi=300)
  plt.close()
  print(f"[PLOT] Saved reconstructions to '{outfile}'")

if __name__ == "__main__":
  setupParallel()

  print("[INFO] Loading datasets...")
  pi_data, rho_data = loadDatasetMix()
  np.random.shuffle(pi_data)

  split_index = int(0.8 * len(pi_data))
  training_pions = pi_data[:split_index].copy()
  validation_pions = pi_data[split_index:].copy()

  print(f"[INFO] Training on {len(training_pions)} pions. Validating on {len(validation_pions)} pions.")

  model = create_autoencoder()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss='mse',      
                metrics=['mae']) 
  model.summary()

  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', 
      patience=args.patience, 
      restore_best_weights=True,
      verbose=1
  )

  history = model.fit(training_pions, training_pions,
                      validation_data=(validation_pions, validation_pions),
                      epochs=args.epochs,
                      batch_size=256,
                      callbacks=[early_stop],
                      verbose=0)

  if args.save:
    model.save(f"{args.outfile}.keras")

  plot_history(history)
  check_reconstructions(model, validation_pions)