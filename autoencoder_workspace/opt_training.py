import os
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval


ETA_MAX = 2.5
MAX_EVALS = 20  

def setupParallel():
    physical_cores = multiprocessing.cpu_count()
    target = int(physical_cores / 2)
    tf.config.threading.set_intra_op_parallelism_threads(target)
    tf.config.threading.set_inter_op_parallelism_threads(target)
    print(f"[SETUP] Tensorflow loaded on {target} physical cores.")

def loadDatasetMix():
    pi = np.load('../datasets/pi.npy')
    rho = np.load('../datasets/rho.npy')
    pi[:, 1] = np.clip(pi[:, 1], 0.0, 1.0)
    rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
    pi[:, 3] = (pi[:, 3] - ETA_MAX) / 5.0
    rho[:, 3] = (rho[:, 3] - ETA_MAX) / 5.0
    return pi, rho

# search space for optimization
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
    'bottleneck': hp.choice('bottleneck', [2, 3]), 
    'dense_1': hp.choice('dense_1', [16, 32]),
    'dense_2': hp.choice('dense_2', [8, 16])
}

def objective(params):
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(params['dense_1'], activation='elu')(inputs)
    x = tf.keras.layers.Dense(params['dense_2'], activation='elu')(x)
    latent = tf.keras.layers.Dense(params['bottleneck'], activation='linear')(x)
    
    x = tf.keras.layers.Dense(params['dense_2'], activation='elu')(latent)
    x = tf.keras.layers.Dense(params['dense_1'], activation='elu')(x)
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss='mse'
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )
    
    history = model.fit(
        training_pions, training_pions,
        validation_data=(validation_pions, validation_pions),
        epochs=150,
        batch_size=256,
        callbacks=[early_stop],
        verbose=0
    )
    
    best_val_loss = min(history.history['val_loss'])
    return {'loss': best_val_loss, 'status': STATUS_OK, 'params': params}

def plot_history(history):
    '''Plots training VS validation datasets'''
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Best autoencoder: training LOSS (MSE)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.yscale('log')
    plt.legend(frameon=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    outfile = 'best_history.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[PLOT] Saved training history to '{outfile}'")
    
def check_reconstructions(model, val_data):
    '''Plots training VS validation to check overfitting in reconstruction'''
    print("\n[INFO] Generating reconstructions for physical validation...")
    predictions = model.predict(val_data, batch_size=256, verbose=0)
    
    labels = [r'$x_{vis}$', r'$Iso = \Sigma E_{ph} / E_{track}$', r'$f_{had} = E_{HCAL}/E_{tot}$', r'$\eta$ (Scaled)']
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(r'Best autoencoder: reconstruction of $\tau\to\pi\nu$', fontweight='bold')
    
    for i in range(4):
        row = i // 2
        col = i % 2
        ax[row, col].hist(val_data[:, i], bins=40, histtype='step', linewidth=1.5, label='Validation data')
        ax[row, col].hist(predictions[:, i], bins=40, histtype='step', linewidth=1.5, label='AE prediction')
        ax[row, col].set_title(labels[i])
        ax[row, col].legend()
        
    plt.tight_layout()
    
    outfile = 'best_reconstruction.png'
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[PLOT] Saved reconstructions to '{outfile}'")


if __name__ == "__main__":
    setupParallel()

    print("\n[INFO] Loading datasets...")
    pi_data, _ = loadDatasetMix()
    
    np.random.seed(42) 
    np.random.shuffle(pi_data)

    split_index = int(0.8 * len(pi_data))
    training_pions = pi_data[:split_index].copy()
    validation_pions = pi_data[split_index:].copy()
    print(f"[INFO] Data ready. Training Pions: {len(training_pions)} | Validation Pions: {len(validation_pions)}")

    print(f"\n[HYPEROPT] Starting Bayesian Optimization ({MAX_EVALS} evaluations)...")
    trials = Trials()
    
    best_indices = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials
    )

    best_params = space_eval(space, best_indices)
    
    print("\n=======================================================")
    print("OPTIMIZED RESULTS:")
    print(f"Learning Rate : {best_params['learning_rate']:.5f}")
    print(f"Bottleneck    : {best_params['bottleneck']}D")
    print(f"Dense Layer 1 : {best_params['dense_1']} nodes")
    print(f"Dense Layer 2 : {best_params['dense_2']} nodes")
    print(f"Best Val Loss (MSE)   : {trials.best_trial['result']['loss']:.6e}")
    print("=======================================================\n")
    
    print("[INFO] Re-train the best model to save to disk...")
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(best_params['dense_1'], activation='elu')(inputs)
    x = tf.keras.layers.Dense(best_params['dense_2'], activation='elu')(x)
    latent = tf.keras.layers.Dense(best_params['bottleneck'], activation='linear')(x)
    x = tf.keras.layers.Dense(best_params['dense_2'], activation='elu')(latent)
    x = tf.keras.layers.Dense(best_params['dense_1'], activation='elu')(x)
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)
    
    champion_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    champion_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='mse'
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    history = champion_model.fit(
        training_pions, training_pions,
        validation_data=(validation_pions, validation_pions),
        epochs=300, batch_size=256, callbacks=[early_stop], verbose=1
    )
    
    champion_model.save("best_pion_autoencoder.keras")
    print("\n[SUCCESS] Saved optimal model as 'best_pion_autoencoder.keras'")
    
    # plot best results
    plot_history(history)
    check_reconstructions(champion_model, validation_pions)