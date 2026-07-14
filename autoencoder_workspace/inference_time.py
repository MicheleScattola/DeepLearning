"""Autoencoder inference time benchmark.

Run from autoencoder_workspace/:
    python inference_time.py
"""
import time
import multiprocessing
import numpy as np
import tensorflow as tf

ETA_MAX  = 2.5
N_WARMUP = 500
N_RUNS   = 5


def setup():
    cores = multiprocessing.cpu_count() // 2
    tf.config.threading.set_intra_op_parallelism_threads(cores)
    tf.config.threading.set_inter_op_parallelism_threads(cores)


def load_data():
    pi = np.load('../datasets/pi.npy')
    pi[:, 1] = np.clip(pi[:, 1], 0.0, 1.0)
    pi[:, 3] = (pi[:, 3] + ETA_MAX) / 5.0
    return pi.astype(np.float32)


if __name__ == '__main__':
    setup()

    print("[INFO] Loading model...")
    model = tf.keras.models.load_model('best_pion_autoencoder.keras')

    print("[INFO] Loading data...")
    pi = load_data()
    N  = len(pi)
    print(f"[INFO] {N} events")

    print(f"[INFO] Warm-up ({N_WARMUP} events)...")
    _ = model.predict(pi[:N_WARMUP], batch_size=256, verbose=0)

    print(f"[INFO] Benchmarking ({N_RUNS} runs)...")
    times = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        _ = model.predict(pi, batch_size=512, verbose=0)
        times.append(time.perf_counter() - t0)
        print(f"  run {i+1}: {times[-1]*1000:.1f} ms")

    mean_t = np.mean(times)
    lines = [
        "[AE RESULT]",
        f"  Events          : {N}",
        f"  Runs            : {N_RUNS}",
        f"  Mean total time : {mean_t*1000:.1f} ms",
        f"  Time per event  : {mean_t/N*1e6:.2f} us/event",
        f"  Throughput      : {N/mean_t:.0f} events/s",
    ]
    for line in lines:
        print(line)
    with open('inference_time.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("[INFO] Results written to 'inference_time.txt'")