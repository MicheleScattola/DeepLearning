import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve

ETA_MAX = 2.5
K       = 10

# Best hyperparameters from opt_training.py (best.txt)
BEST_PARAMS = {
    'learning_rate': 0.00046,
    'activation':    'relu',
    'bottleneck':    3,
    'dense_1':       32,
    'dense_2':       16,
}


def setup_parallel():
    target = 8
    tf.config.threading.set_intra_op_parallelism_threads(target)
    tf.config.threading.set_inter_op_parallelism_threads(target)
    print(f"[SETUP] Tensorflow on {target} cores.")


def load_datasets():
    pi  = np.load('../datasets/pi.npy')
    rho = np.load('../datasets/rho.npy')
    pi[:,  1] = np.clip(pi[:,  1], 0.0, 1.0)
    rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
    pi[:,  3] = (pi[:,  3] + ETA_MAX) / 5.0
    rho[:, 3] = (rho[:, 3] + ETA_MAX) / 5.0
    return pi, rho


def build_model(params):
    act = params.get('activation', 'tanh')
    inputs = tf.keras.layers.Input(shape=(4,))
    x = tf.keras.layers.Dense(params['dense_1'], activation=act)(inputs)
    x = tf.keras.layers.Dense(params['dense_2'], activation=act)(x)
    latent = tf.keras.layers.Dense(params['bottleneck'], activation='linear')(x)
    x = tf.keras.layers.Dense(params['dense_2'], activation=act)(latent)
    x = tf.keras.layers.Dense(params['dense_1'], activation=act)(x)
    outputs = tf.keras.layers.Dense(4, activation='linear')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss='mse')
    return model


def anomaly_scores(model, data):
    recon = model.predict(data, batch_size=512, verbose=0)
    return np.mean(np.square(data - recon), axis=1)


if __name__ == '__main__':
    setup_parallel()

    print("[INFO] Loading datasets...")
    pi_data, rho_data = load_datasets()

    np.random.seed(42)
    np.random.shuffle(pi_data)

    kf     = KFold(n_splits=K, shuffle=False)
    aucs   = []
    curves = []   # (fpr, tpr) per fold

    # labels: 0 = pion (signal), 1 = rho (background)
    y_rho = np.ones(len(rho_data))

    for fold, (train_idx, test_idx) in enumerate(kf.split(pi_data), start=1):
        print(f"\n[FOLD {fold}/{K}] train={len(train_idx)}  test={len(test_idx)}")

        train_pi = pi_data[train_idx]
        test_pi  = pi_data[test_idx]

        tf.keras.utils.set_random_seed(42)
        model = build_model(BEST_PARAMS)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30, restore_best_weights=True, verbose=0
        )
        model.fit(
            train_pi, train_pi,
            validation_split=0.1,
            epochs=400, batch_size=256,
            callbacks=[early_stop], verbose=0
        )

        scores_pi  = anomaly_scores(model, test_pi)
        scores_rho = anomaly_scores(model, rho_data)

        y_true  = np.concatenate([np.zeros(len(scores_pi)), y_rho])
        y_score = np.concatenate([scores_pi, scores_rho])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fold_auc = roc_auc_score(y_true, y_score)
        aucs.append(fold_auc)
        curves.append((fpr, tpr))
        print(f"[FOLD {fold}/{K}] AUC = {fold_auc:.4f}")

    aucs = np.array(aucs)
    lines = [
        "\n=======================================================",
        f"K-Fold Cross-Validation (k={K})",
        f"AUC per fold : {' '.join(f'{a:.4f}' for a in aucs)}",
        f"Mean AUC     : {aucs.mean():.4f}",
        f"Std AUC      : {aucs.std():.4f}",
        "=======================================================",
    ]
    for line in lines:
        print(line)
    with open("auc.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[INFO] Saved results to 'auc.txt'")

    # ROC overlay plot
    fpr_grid    = np.linspace(0, 1, 300)
    tprs_interp = np.array([np.interp(fpr_grid, fpr, tpr) for fpr, tpr in curves])
    mean_tpr    = tprs_interp.mean(axis=0)

    _, ax = plt.subplots(figsize=(6, 6))
    for i, (fpr, tpr) in enumerate(curves):
        ax.plot(fpr, tpr, color='steelblue', alpha=0.3, lw=1,
                label='Folds' if i == 0 else None)
    ax.plot(fpr_grid, mean_tpr, color='steelblue', lw=2,
            label=f'Mean AUC = {aucs.mean():.3f} ± {aucs.std():.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'AE ROC — {K}-fold', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('auc_roc.png', dpi=300)
    plt.close()
    print("[PLOT] Saved 'auc_roc.png'")
