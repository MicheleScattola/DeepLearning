"""Shared RTBM construction / training / evaluation utilities.

Extracted from training.py to avoid duplicating model-construction, the
CMA-ES training loop, and evaluation logic across the CLI driver and the
performance sweep scripts (sweep.py, plot_performance.py). See math.md
for the derivations behind make_rtbm and train_rtbm's defaults.

mytraining.py is intentionally NOT refactored to use this module -- it is
a personal rewrite kept separate for learning purposes.
"""
import signal
import numpy as np
import multiprocessing as mp
from contextlib import contextmanager
from cma import CMAEvolutionStrategy

from theta.rtbm import RTBM
from theta.minimizer import worker_initialize, worker_compute
from theta.costfunctions import sum as log_nll_cost

ETA_MAX        = 2.5
N_VISIBLE      = 4
NAN_PENALTY    = 1e9
PHYS_CORES     = mp.cpu_count()
PARALLEL_CORES = int(PHYS_CORES / 2)
GEN_TIMEOUT    = 60  # seconds; see TrainingTimeout below


class TrainingTimeout(RuntimeError):
    """Raised when a single CMA-ES generation exceeds the per-generation time limit.

    This is a distinct failure mode from the Schur-complement collapse in
    math.md (which produces a clean NaN/inf almost instantly): here the
    parameters pass the positivity check but land near a singular T/Q,
    making the Riemann theta lattice-summation radius explode (the number
    of lattice points to enumerate grows roughly as R^N_h, and R itself
    diverges as the smallest eigenvalue of Im(Omega) shrinks toward zero).
    For N_h >= 4 this can hang a worker process for an unbounded time on a
    single candidate. Without a timeout, multiprocessing.Pool.map() blocks
    forever waiting for that one worker's result, since Pool has no
    built-in cancellation for in-flight tasks.
    """


@contextmanager
def _time_limit(seconds):
    """SIGALRM-based timeout for the single-core (ncores=1) training path."""
    def _handler(signum, frame):
        raise TrainingTimeout(f"Generation exceeded {seconds}s time limit")
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── data ──────────────────────────────────────────────────────────────────────
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
    """80/20 train/val split of the first n_train events; remainder held out for testing."""
    N    = min(n_train, len(pi_data))
    N_tr = int(0.8 * N)
    return pi_data[:N_tr], pi_data[N_tr:N], pi_data[N:]


# ── model construction ───────────────────────────────────────────────────────
def make_rtbm(nv, nh, param_bound, max_tries=200):
    """Create RTBM with full (non-diagonal) T and W=0 at initialisation.

    diagonal_T=True permanently restricts T to be diagonal, preventing the
    Gaussian envelope from capturing feature correlations. Instead we use
    diagonal_T=False and zero W after random_init: this gives theta ratio = 1
    at init (same numerical benefit as diagonal_T=True) without the
    architectural restriction. See math.md Section 3.

    random_bound scales as sqrt(param_bound), with NO floor: a floor (e.g.
    max(2, ...)) makes random_bound -- and hence the entire initial draw --
    IDENTICAL for any param_bound below the floor's square, silently
    collapsing the param_bound sweep into a handful of duplicate points
    (math.md Section 11). Without the floor, E[T_ii]/sigma stays constant
    across param_bound by construction (both scale with param_bound), so
    the Schur-complement safety margin from math.md Section 9 is preserved
    at every param_bound, not just large ones.

    CMA bounds are still widened to contain the actual initial parameters
    (full-T diagonal entries are sums of squares and can exceed param_bound
    -- math.md Section 9.3), but train_rtbm is given the *requested*
    param_bound separately via init_sigma, so the widened box constraint
    never silently overrides the user's intended exploration step size.
    """
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
    raise RuntimeError(
        f"make_rtbm: no valid initialisation found in {max_tries} tries "
        f"(nv={nv}, nh={nh}, param_bound={param_bound}). Returning an unzeroed/"
        f"unbounded model would crash later in train_rtbm with an out-of-bounds "
        f"ValueError; failing here instead so a sweep script can catch and log it."
    )


def run_tag(nh, pb, seed):
    """Filename-safe identifier for a single (n_hidden, param_bound, seed) run.

    Shared between sweep.py (writer) and plot_performance.py (reader) so the
    two scripts agree on where per-run diagnostics (CMA history, valid
    fraction) are stored without either parsing the other's output.
    """
    return f"nh{nh}_pb{pb:.3f}_seed{seed}".replace('.', 'p')


def default_popsize(n_params):
    """CMA-ES default population size: 4 + floor(3 ln N).

    Used to convert a target function-evaluation budget into a maxiter
    value that is comparable across different N_hidden (popsize grows
    with the parameter count, so fixed maxiter would bias larger models
    toward more total fevals). See performance.md Section 2.
    """
    return 4 + int(3 * np.log(n_params))


# ── training ──────────────────────────────────────────────────────────────────
def train_rtbm(model, x_theta, ncores=1, maxiter=200, tolfun=0.0, seed=None,
                return_diagnostics=False, gen_timeout=GEN_TIMEOUT, init_sigma=None):
    """CMA-ES training loop.

    Returns (best_params, history) by default, matching training.py's
    original signature. If return_diagnostics=True, returns a third element:
    {'valid_fraction': [...], 'fevals': int} -- the fraction of non-penalty
    candidates per generation and the total CMA function evaluations,
    tracked at negligible cost (performance.md Section 4).

    seed, if given, sets cma_opts['seed'] for CMA-ES reproducibility. Combined
    with np.random.seed(seed) called before make_rtbm, this fully determines
    the run (performance.md Section 5) -- worker processes draw no random
    numbers, so multiprocessing introduces no additional seed-dependence.

    gen_timeout bounds the wall-clock time of a single generation's batch
    of fitness evaluations. Raises TrainingTimeout (and, for ncores>1,
    forcibly terminates the worker pool) if exceeded -- protects against a
    near-singular T/Q candidate making the Riemann theta lattice sum hang
    indefinitely (see TrainingTimeout docstring above). Callers should
    treat this the same as any other training failure (catch and skip).

    init_sigma, if given, sets CMA's initial step size directly (callers
    should pass param_bound * 0.1). If omitted, falls back to
    max(model.get_bounds()[1]) * 0.1 -- but that fallback silently loses
    param_bound's identity whenever make_rtbm had to widen the bounds past
    the requested param_bound (math.md Section 11): different param_bound
    values can then produce byte-for-byte identical runs. Always pass
    init_sigma explicitly when param_bound is meant to vary the search.
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
                    raise TrainingTimeout(
                        f"CMA generation exceeded {gen_timeout}s -- a candidate likely hit "
                        f"a near-singular T/Q causing the Riemann theta lattice sum to "
                        f"explode (more likely at higher N_hidden). Aborting this run."
                    )
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


# ── evaluation ────────────────────────────────────────────────────────────────
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
    from sklearn.metrics import roc_curve, auc
    y_true   = np.concatenate([np.zeros(len(scores_pi)), np.ones(len(scores_rho))])
    y_scores = np.concatenate([scores_pi, scores_rho])
    mask     = np.isfinite(y_scores)
    fpr, tpr, _ = roc_curve(y_true[mask], y_scores[mask])
    return float(auc(fpr, tpr))


def background_rejection(scores_pi, scores_rho, target_eff=0.95):
    """Fraction of rho rejected at a threshold giving target_eff pion signal efficiency."""
    threshold = np.percentile(scores_pi, target_eff * 100)
    return float(np.sum(scores_rho > threshold) / len(scores_rho))
