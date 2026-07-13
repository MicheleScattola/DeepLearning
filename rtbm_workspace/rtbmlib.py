"""Shared RTBM construction, training, and evaluation utilities."""
import signal
import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from cma import CMAEvolutionStrategy

from theta.rtbm import RTBM
from theta.minimizer import worker_initialize, worker_compute
from theta.costfunctions import sum as log_nll_cost

from sklearn.metrics import roc_curve, auc

ETA_MAX        = 2.5
N_VISIBLE      = 4
NAN_PENALTY    = 1e9
PHYS_CORES     = mp.cpu_count()
PARALLEL_CORES = int(PHYS_CORES / 2)
GEN_TIMEOUT    = 60 


@contextmanager
def _time_limit(seconds):
    """SIGALRM-based timeout for the single-core (ncores=1) training path."""
    def _handler(signum, frame):
        raise RuntimeError(f"Generation exceeded {seconds}s time limit")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# data manipulation
def load_datasets(eta_max=ETA_MAX):
    pi  = np.load('../datasets/pi.npy')
    rho = np.load('../datasets/rho.npy')
    pi[:,  1] = np.clip(pi[:,  1], 0.0, 1.0)
    rho[:, 1] = np.clip(rho[:, 1], 0.0, 1.0)
    pi[:,  3] = (pi[:,  3] + eta_max) / 5.0
    rho[:, 3] = (rho[:, 3] + eta_max) / 5.0
    eps = 1e-4
    for arr in (pi, rho):
        arr[:, 0] = np.log(np.clip(arr[:, 0], eps, 1-eps) / (1 - np.clip(arr[:, 0], eps, 1-eps)))
    return pi, rho


def standardize(train, *others):
    mu  = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mu) / std, [(x - mu) / std for x in others], (mu, std)


def train_val_test_split(pi_data, n_train):
    """80/20 train/val split of the first n_train events; remainder is test set."""
    N    = min(n_train, len(pi_data))
    N_tr = int(0.8 * N)
    return pi_data[:N_tr], pi_data[N_tr:N], pi_data[N:]


# construct model
def make_rtbm(nv, nh, param_bound, max_tries=200):
    """Create RTBM with full T and W=0 at initialisation. Raises RuntimeError if no valid init found."""
    random_bound = param_bound ** 0.5
    sigma = param_bound * 0.1
    for _ in range(max_tries):
        m = RTBM(nv, nh, init_max_param_bound=param_bound, random_bound=random_bound,
                 diagonal_T=False, mode=RTBM.Mode.LogProbability)
        if np.all(np.diag(m.t) > sigma) and np.all(np.diag(m.q) > sigma):
            params = np.real(m.get_parameters()).copy()
            params[nv + nh : nv + nh + nv * nh] = 0.0
            if m.set_parameters(params):
                actual_max = float(np.max(np.abs(params)))
                m.set_bounds(max(param_bound, actual_max) * 1.2)
                return m
    raise RuntimeError(f"make_rtbm: no valid init in {max_tries} tries (nv={nv}, nh={nh}, param_bound={param_bound})")


def run_tag(nh, pb, seed):
    """Filename for sweep run."""
    return f"nh{nh}_pb{pb:.3f}_seed{seed}".replace('.', 'p')


def default_popsize(n_params):
    """CMA-ES population size: 4 + floor(3 ln N)."""
    return 4 + int(3 * np.log(n_params))


# training
def train_rtbm(model, x_theta, ncores=PARALLEL_CORES, maxiter=200, tolfun=0.0, seed=None,
                return_diagnostics=False, gen_timeout=GEN_TIMEOUT, init_sigma=None):
    """CMA-ES training loop.

    - Returns (best_params, history).
    - With return_diagnostics=True adds a third
    element {'valid_fraction': [...], 'fevals': int}.
    - Raises RuntimeError on
    gen_timeout (near-singular candidate hangs the lattice sum).
    """
    initsol  = np.real(model.get_parameters())
    sigma    = init_sigma if init_sigma is not None else np.max(model.get_bounds()[1]) * 0.1
    cma_opts = {
        'bounds':         model.get_bounds(),
        'tolfun':         tolfun,
        'maxiter':        maxiter,
        'verb_log':       0,
        'tolflatfitness': maxiter,
    }
    if seed is not None:
        cma_opts['seed'] = seed
    es      = CMAEvolutionStrategy(initsol, sigma, cma_opts)
    history = []
    valid_fraction_history = []

    def _record(fits):
        """Substitute NaN -> NAN_PENALTY and append valid fraction for history.
        """
        valid_fraction_history.append(sum(np.isfinite(v) for v in fits) / len(fits))
        return [v if np.isfinite(v) else NAN_PENALTY for v in fits]

    if ncores > 1:
        pool = mp.Pool(ncores, initializer=worker_initialize,
                       initargs=(log_nll_cost, model, x_theta, None))
        try:
            while not es.stop():
                candidates = es.ask()
                try:
                    fits = pool.map_async(worker_compute, candidates).get(timeout=gen_timeout)
                except mp.TimeoutError:
                    raise RuntimeError(f"CMA generation exceeded {gen_timeout}s (near-singular candidate)")
                f_values = _record(fits)
                es.tell(candidates, f_values)
                es.disp()
                history.append(es.best.f)
        finally:
            pool.terminate()
            pool.join()
    else:
        worker_initialize(log_nll_cost, model, x_theta, None)
        while not es.stop():
            candidates = es.ask()
            with _time_limit(gen_timeout):
                fits = [worker_compute(s) for s in candidates]
            f_values = _record(fits)
            es.tell(candidates, f_values)
            es.disp()
            history.append(es.best.f)

    model.set_parameters(es.result[0])

    if return_diagnostics:
        diagnostics = {'valid_fraction': valid_fraction_history, 'fevals': es.countevals}
        return es.result[0], history, diagnostics
    return es.result[0], history


# evalution
def anomaly_scores(model, x_theta):
    try:
        log_probs = np.real(model(x_theta)).flatten()
        log_probs = np.where(np.isfinite(log_probs), log_probs, -1e6)
        return -log_probs
    except np.linalg.LinAlgError:
        return np.full(x_theta.shape[1], 1e6)


def mean_nll(model, x_theta):
    try:
        log_probs = np.real(model(x_theta)).flatten()
        finite = log_probs[np.isfinite(log_probs)]
        return float(-np.mean(finite)) if len(finite) > 0 else 1e9
    except np.linalg.LinAlgError:
        return 1e9


def compute_auc(scores_pi, scores_rho):
    """ROC AUC for pi (label 0) vs rho (label 1), using -log P anomaly scores."""
    
    y_true   = np.concatenate([np.zeros(len(scores_pi)), np.ones(len(scores_rho))])
    y_scores = np.concatenate([scores_pi, scores_rho])
    mask     = np.isfinite(y_scores)
    fpr, tpr, _ = roc_curve(y_true[mask], y_scores[mask])
    return float(auc(fpr, tpr))


def background_rejection(scores_pi, scores_rho, target_eff=0.95):
    """Fraction of rho rejected at a threshold giving target_eff pion signal efficiency."""
    threshold = np.percentile(scores_pi, target_eff * 100)
    return float(np.sum(scores_rho > threshold) / len(scores_rho))
