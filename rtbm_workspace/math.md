# RTBM Training: Numerical Issues and Fixes

## 1. The RTBM Probability Formula

The RTBM assigns probability density to a visible vector $v \in \mathbb{R}^{N_v}$ via:

$$P(v) = \sqrt{\frac{\det T}{(2\pi)^{N_v}}} \cdot \underbrace{e^{-\frac{1}{2}v^T T v - B_v^T v - \frac{1}{2}B_v^T T^{-1} B_v}}_{\text{Gaussian envelope}} \cdot \underbrace{\frac{\theta\!\left(\frac{v^T W + B_h^T}{2\pi i} \;\Big|\; \frac{-Q}{2\pi i}\right)}{\theta\!\left(\frac{B_h^T - B_v^T T^{-1} W}{2\pi i} \;\Big|\; \frac{-Q + W^T T^{-1} W}{2\pi i}\right)}}_{\text{theta ratio}}$$

The Gaussian envelope captures the bulk shape. The **theta ratio** captures non-Gaussian structure via the hidden units: the numerator is event-dependent (through $v^T W$), while the denominator is a normalisation constant.

The Riemann theta function itself is:

$$\theta(z \mid \Omega) = \sum_{n \in \mathbb{Z}^{N_h}} \exp\!\left(2\pi i \left(\tfrac{1}{2} n^T \Omega n + n^T z\right)\right)$$

In Phase I (the setting used here), $z$ is purely imaginary and $\Omega$ is purely imaginary with positive-definite imaginary part, so the sum becomes a real decaying exponential and is always convergent and positive. The library factors this as $\theta = e^u \cdot v$ where $u$ is the dominant exponential growth and $v$ is the bounded oscillatory part.

---

## 2. The Root Cause of Training Failure: $W \neq 0$ at Initialisation

The `random_init` method builds initial parameters via the Schur complement trick. It draws a random matrix $X \in \mathbb{R}^{(N_v + N_h) \times (N_v + N_h)}$ and computes $A = X^T X$ (positive semi-definite by construction), then carves out the blocks:

$$Q = A_{[:N_h,\, :N_h]}, \quad T = A_{[N_h:,\, N_h:]}, \quad W = A_{[N_h:,\, :N_h]}$$

When $X$ is a **full matrix**, the off-diagonal block $W = A_{[N_h:,\, :N_h]}$ is generally **non-zero**. This has catastrophic consequences for the theta ratio at initialisation.

Substituting $W \neq 0$ and small biases ($B_v \approx 0$, $B_h \approx 0$) into the theta ratio:

- **Numerator**: $\theta\!\left(\frac{v^T W}{2\pi i} \;\Big|\; \frac{-Q}{2\pi i}\right)$ — depends on $v$ through $v^T W$
- **Denominator**: $\theta\!\left(0 \;\Big|\; \frac{-(Q - W^T T^{-1} W)}{2\pi i}\right)$ — the normalisation evaluated at zero

For a typical training event $v$ drawn from a standardised 4D distribution, the argument $v^T W$ has magnitude $O(1)$, which is finite. However, the denominator involves the Schur complement $Q - W^T T^{-1} W$, and for a randomly initialised $W$ and $T$ the ratio $\theta_\text{num}/\theta_\text{denom}$ can be astronomically small — empirically this gives $\log P \approx -600$ per event at initialisation.

Concretely, from the computation of the oscillatory part in `log_eval`:

$$\log \theta = u + \log v$$

where $v$ is the oscillatory part (a sum of decaying Gaussians over the integer lattice). When the theta ratio collapses, $v \to 0$ numerically, so $\log v \to -\infty$, and the cost becomes $+\infty$. Since `worker_compute` filters `NaN` values but **not `+inf`**, a population of all-`+inf` candidates is accepted by CMA-ES. With all fitness values identical, `tolflatfitness=1` (the library default) triggers after one generation and training stops.

---

## 3. Fix: Force $W = 0$ at Initialisation (without permanently constraining $T$)

### Why $W = 0$ at init is necessary

The theta ratio must equal 1 at initialisation so that $\log P$ starts at a finite, manageable value. This requires the numerator and denominator theta functions to be equal, which happens when $W = 0$:

$$\frac{\theta\!\left(\frac{B_h^T}{2\pi i} \;\Big|\; \frac{-Q}{2\pi i}\right)}{\theta\!\left(\frac{B_h^T}{2\pi i} \;\Big|\; \frac{-Q}{2\pi i}\right)} = 1$$

The probability then reduces to a pure Gaussian:

$$P(v) = \sqrt{\frac{\det T}{(2\pi)^{N_v}}} \cdot e^{-\frac{1}{2}v^T T v - B_v^T v - \dots}$$

For standardised data with $T \approx I$: $\log P \approx -5.6$ per event, NLL $\approx 36\,000$ — a finite, well-scaled landscape CMA-ES can navigate.

### Why `diagonal_T=True` is the wrong architectural fix

One way to force $W = 0$ is `diagonal_T=True`: with a diagonal $X$, $A = X^T X$ is diagonal, so the off-diagonal block $W = 0$. But `diagonal_T=True` is **permanent** — the theta library's `set_parameters` only updates the diagonal of $T$ throughout training, leaving off-diagonal elements at zero forever.

This permanently constrains the Gaussian envelope to factorise as independent 1D Gaussians:

$$e^{-\frac{1}{2}v^T T v} \xrightarrow{\text{diagonal }T} \prod_i e^{-\frac{1}{2}T_{ii}\,v_i^2}$$

For this physics problem, the four features are physically correlated:
- **$f_\text{had}$ and $\text{Iso}$** both probe the same missing $\pi^0 \to \gamma\gamma$. High Iso typically accompanies low $f_\text{had}$ (EM calorimeter picking up the photons) — a genuine covariance.
- **$x_\text{vis}$ and $\text{Iso}$** are correlated through energy conservation: a missed $\pi^0$ that carries away energy raises Iso and also biases $x_\text{vis}$.

A diagonal $T$ cannot represent a tilted ellipse in the $(f_\text{had}, \text{Iso})$ plane. All cross-feature correlations must be carried entirely by the hidden units through $W$ — a severe capacity constraint for $N_h = 2$.

### The correct fix: zero $W$ after `random_init`, keep full $T$

Using `diagonal_T=False` with `random_init`, the Schur complement block gives a full PSD matrix $T$ and $W \neq 0$ (causing the theta collapse described in Section 2). The fix is to use `diagonal_T=False` but then **manually zero the $W$ entries** in the parameter vector after `random_init`. Since $W = 0$, the Schur complement $Q - W^T T^{-1} W = Q > 0$ is automatically satisfied, and `set_parameters` succeeds:

```python
params = np.real(m.get_parameters()).copy()
params[N_v + N_h : N_v + N_h + N_v \cdot N_h] = 0.0   # zero W entries
m.set_parameters(params)   # returns True since Schur = Q > 0
```

This gives:
- **Same numerical starting point** as `diagonal_T=True`: theta ratio = 1, log $P$ ≈ −5.6/event
- **Full $T$ matrix** free to develop off-diagonal entries during CMA training
- **Full parameter count** (27 for $N_v=4$, $N_h=2$ vs 21 for diagonal $T$)

---

## 4. Fix: `random_bound=2` for Stable Initial Scale

With `diagonal_T=True` and `random_bound=b`, the initial T diagonal entries are:

$$T_{ii}^{(0)} = x_i^2, \quad x_i \sim \text{Uniform}(-b, b) \implies \mathbb{E}[T_{ii}^{(0)}] = \frac{b^2}{3}$$

The CMA-ES initial step size is $\sigma = \text{param\_bound} \times 0.1$.

For a typical entry to survive a one-sigma perturbation without going negative:

$$T_{ii}^{(0)} - \sigma > 0 \implies \frac{b^2}{3} > \text{param\_bound} \times 0.1$$

With `param_bound=20` ($\sigma = 2.0$) and `random_bound=1` ($\mathbb{E}[T_{ii}] = 0.33$): the mean entry is far below $\sigma$, so almost every candidate violates positivity and returns NaN. With **`random_bound=2`** ($\mathbb{E}[T_{ii}] = 1.33$): a one-sigma perturbation of 2.0 applied to the mean gives $1.33 - 2.0 = -0.67$ (failing), but the full distribution $T_{ii} \sim b^2 \cdot \text{Beta}$-like covers $[0, 4]$ with most mass above $\sigma$, making a large fraction of candidates valid.

The same argument applies to the diagonal entries of $Q$.

---

## 5. Fix: `LogProbability` Mode Avoids Underflow

In `Probability` mode, `rtbm_probability` computes:

$$P(v) = \sqrt{\frac{\det T}{(2\pi)^{N_v}}} \cdot e^{\text{ExpF}} \cdot \frac{v_1}{v_2} \cdot e^{u_1 - u_2}$$

where $(u_i, v_i)$ are the exponential and oscillatory parts of each theta evaluation. As training progresses and the model explores regions with small theta ratios, $e^{u_1 - u_2}$ can underflow to zero in float64 ($e^{-710}$ is the limit), making $P = 0$ and then $\log P = -\infty$.

In **`LogProbability` mode**, `rtbm_log_probability` computes:

$$\log P(v) = \frac{1}{2}\log\frac{\det T}{(2\pi)^{N_v}} + \text{ExpF} + (u_1 + \log v_1) - (u_2 + \log v_2)$$

The intermediate `exp` that could underflow is never taken. The computation lives entirely in log-space, so even if $e^{u_1-u_2}$ would be $10^{-300}$, the log-space result $u_1 - u_2 = -690$ is a perfectly representable float64.

The paired cost function `costfunctions.sum.cost(x) = -\sum x` then takes the log-probabilities directly, giving NLL $= -\sum \log P(v_i)$.

---

## 6. Fix: `tolflatfitness=maxiter` and `tolfun=0` Prevent Premature Stopping

The `cma` library has two stopping criteria that fire spuriously in this problem.

### `tolflatfitness`

The default `tolflatfitness=1` halts optimisation if **all fitness values in the current population are equal** for one consecutive generation. At early iterations (near the Gaussian initialisation, where the landscape is flat), this can trigger after just 1–2 generations.

Setting `tolflatfitness=maxiter` allows the optimiser to pass through flat regions for up to `maxiter` generations before stopping.

### `tolfun` — the more dangerous criterion

The `tolfun` criterion halts optimisation when:

$$\max(\text{fitness}) - \min(\text{fitness}) < \texttt{tolfun}$$

i.e. when the **range** of fitness values in the current population is below the tolerance. This is intended to detect convergence, but it misfires catastrophically when the NAN penalty strategy is in use.

When all candidates in a generation fail the Schur positivity check, they all receive `NAN_PENALTY = 1e9`. The range is exactly 0:

$$\max(1\mathrm{e}9, \ldots, 1\mathrm{e}9) - \min(1\mathrm{e}9, \ldots, 1\mathrm{e}9) = 0 < 10^{-5} = \texttt{tolfun}$$

CMA-ES interprets this as convergence and stops after a single iteration.

This affects any configuration where the Schur constraint is harder to satisfy — specifically $N_h \geq 3$, where the 3×3 (or larger) Schur complement has a lower probability of being positive definite for a given $\sigma$. With $N_h = 2$ the 2×2 Schur passes often enough that most generations have at least one valid candidate, keeping the range large. With $N_h = 3$ a larger fraction of generations are all-invalid, and `tolfun` fires on the first all-invalid generation.

The same issue appears with `--optimize`: hyperopt explores $N_h \in \{1,2,3,4\}$, so trials with $N_h \geq 3$ hit this.

**Fix**: set `tolfun=0`. Since the check becomes $0 < 0$, which is always False, the criterion is disabled. Convergence is then handled solely by `tolflatfitness=maxiter` (stops when the best NLL stagnates for `maxiter` consecutive generations) and the hard `maxiter` cap.

---

## 7. Fix: `param_bound=5` Prevents Schur Collapse as W Grows

Even with `diagonal_T=True` (ensuring $W=0$ at init), the first CMA-ES generation immediately perturbs $W$ away from zero. With `param_bound=20` ($\sigma=2.0$), the first-generation candidates have $W_{ij} \sim \mathcal{N}(0, \sigma^2) = \mathcal{N}(0, 4)$. The Schur complement then evaluates to:

$$Q - W^T T^{-1} W \approx Q - N_v \cdot \frac{\sigma^2}{\mathbb{E}[T_{ii}]} \cdot I_{N_h} \approx 1.33 - \frac{4 \cdot 4}{1.33} \approx -10.7$$

which is strongly negative — virtually every first-generation candidate fails the positivity check. The handful that barely pass (Schur eigenvalue $\varepsilon \approx 0^+$) produce an astronomically large $\log\theta_2$:

$$\log\theta_2 = u_2 + \log v_2, \quad u_2 = \pi \cdot \|B_h / (2\pi)\|^2 \cdot (2\pi \cdot \text{Schur}^{-1}) \propto \text{Schur}^{-1} \to \infty$$

This drives $\log P \approx -10^5$ per event and cost $= -\sum \log P \approx N_\text{train} \times 10^5 = 10^9$. CMA-ES accepts these as valid (non-NaN) solutions, receives a flat population of cost $\approx 10^9$ in generation 1, and stops.

With `param_bound=5` ($\sigma=0.5$):

$$Q - W^T T^{-1} W \approx 1.33 - \frac{4 \cdot 0.25}{1.33} \approx 0.58 \gg 0$$

Most first-generation candidates have a well-separated Schur complement and produce log probabilities of the expected order ($\sim -6$ per event). The NLL starts at $\sim 46\,000$, drops to $\sim 35\,000$ over 200 iterations — consistent with the convergence curve.

The same marginal-Schur issue also causes a `LinAlgError` crash in the main process after training: `es.result[0]` (the CMA best-seen candidate) can sit right on the Schur boundary due to floating-point rounding between worker and main processes, making the `cholesky` call inside `oscillatory_part` fail. The fix is `try/except np.linalg.LinAlgError` in `mean_nll`, `anomaly_scores`, and `plot_density_check`.

---

## 8. Residual Reconstruction Failure on $x_\text{vis}$ and $\eta$: Architectural Limits

The density check after a successful 200-iteration training shows reasonable fits for $\text{Iso}$ and $\eta$, but $x_\text{vis}$ is reconstructed as a bell-shaped peak rather than the expected near-flat distribution. This is **not** an initialisation or boundary-clipping issue — it is a fundamental property of the Gaussian kernel architecture.

### Why the Gaussian envelope cannot represent flat or linear distributions

The RTBM marginalises over hidden units to give:

$$P(v) = \underbrace{\sqrt{\frac{\det T}{(2\pi)^{N_v}}} \cdot e^{-\frac{1}{2}v^T T v - B_v^T v - \dots}}_{\text{Gaussian in }v} \cdot (\text{theta ratio})$$

For the $x_\text{vis}$ direction in standardised space, the data has a nearly uniform density on a bounded interval $[\tilde{a}, \tilde{b}] \approx [-1.72, +1.72]$ (from the SM mix $P_\tau = -0.147$, which gives $f(x_\text{vis}) \propto 2[f_\text{LH}\, x + f_\text{RH}(1-x)]$, nearly flat). A Gaussian marginal in $v_\text{vis}$ with precision $T_{11}$ assigns:

$$\log P_\text{vis}(x) \approx -\tfrac{1}{2} T_{11}\, x^2 + \text{const}$$

For this to mimic a flat distribution, the optimizer drives $T_{11} \to 0$ (infinitely wide Gaussian). But the normalisation $\sqrt{T_{11}/(2\pi)}$ then also goes to zero, meaning the model simultaneously needs $T_{11} \to 0$ to flatten the shape and $T_{11} \to \infty$ to maintain proper normalisation. The theta ratio can partially compensate, but with $N_h = 2$ hidden units the correction has limited expressiveness.

The practical consequence is that the optimizer settles on a compromise: a moderately wide Gaussian centered in $[\tilde{a}, \tilde{b}]$ with the theta ratio broadening it slightly. The hard physical boundaries at $x_\text{vis} = 0$ and $x_\text{vis} = 1$ are completely invisible to the model — no parameter configuration can force $P = 0$ outside those limits.

The same argument applies to $\eta$: after the initial rescaling $(η - 2.5)/5$, the variable is bounded within the detector acceptance $[-1, 0]$ and roughly uniform within that range. The Gaussian envelope cannot represent this without leaking probability mass outside the physical region.

### Fixing this: logit preprocessing for bounded variables

For any variable $u \in [0, 1]$ (here $x_\text{vis}$ and $f_\text{had}$), the logit transform maps the hard boundaries to $\pm\infty$:

$$u_\text{logit} = \log\frac{u}{1 - u} : [0, 1] \to (-\infty, +\infty)$$

After this transformation, the distribution in logit space is approximately bell-shaped (the nearly-uniform $u$ maps to a logistic distribution in $u_\text{logit}$, which has Gaussian-like tails). The RTBM can then fit the logit-space distribution accurately without fighting the hard boundaries.

$\eta$ looks like it should also benefit — it is bounded by the IDEA detector acceptance $[-2.5, +2.5]$ — but the logit transform fails here for a different reason. From the density check, the pion distribution in $\eta$ is heavily concentrated near the **lower** acceptance boundary $\eta \approx -2.5$. Normalising to $[0,1]$ via $u = (\eta + 2.5)/5$ maps those events to $u \approx 0$, where they get clipped to $\varepsilon = 10^{-4}$ and pile up at $\text{logit}(\varepsilon) \approx -9.2$. The resulting distribution in logit space has a hard spike at one extreme — not Gaussian-like at all. This degenerates the theta function computation and drives the CMA cost back to $10^9$. The old linear rescaling $(\eta - 2.5)/5$ is kept for $\eta$: it already bounds the distribution in $[-1, 0]$ and the RTBM fits it reasonably.

$f_\text{had}$ is also excluded: its distribution is bimodal with spikes at exactly 0 and 1, so logit would scatter those spikes to $\pm 9.2$, making the model's task harder.

In code, inside `load_datasets()`, **only** $x_\text{vis}$ (column 0) receives the logit transform:

```python
pi[:,  3] = (pi[:,  3] - ETA_MAX) / 5.0   # eta: keep original linear rescaling
rho[:, 3] = (rho[:, 3] - ETA_MAX) / 5.0
eps = 1e-4
for arr in (pi, rho):
    arr[:, 0] = np.log(np.clip(arr[:, 0], eps, 1-eps) / (1 - np.clip(arr[:, 0], eps, 1-eps)))
```

After this transform, $x_\text{vis}$ is on $(-\infty, +\infty)$ with an approximately logistic marginal. The density check plot will show $x_\text{vis}$ on a logit axis (roughly $[-4, +4]$) rather than $[0, 1]$. Anomaly scores are unaffected — the relative ordering of $-\log P$ is preserved, so the ROC curve and AUC are identical in physical or logit space.

### Alternatively: more hidden units

Increasing $N_h$ from 2 to 3 or 4 gives the theta ratio more degrees of freedom (more summation directions over $\mathbb{Z}^{N_h}$) to deviate from the Gaussian shape. The parameter count grows as:

$$\text{size}(N_v{=}4,\, N_h) = 2N_v + N_h + \frac{N_h^2+N_h}{2} + N_v N_h$$

giving 21 (Nh=2), 26 (Nh=3), 32 (Nh=4). CMA-ES runtime scales roughly as $N_\text{params}^{1.5}$, so Nh=4 is about $2\times$ slower per iteration than Nh=2. For a distribution that is genuinely non-Gaussian (piecewise linear with hard bounds), Nh=3 with logit preprocessing would be a better-conditioned problem than Nh=4 without it.

---

## 9. Fix: Arcsine Initialisation Trap and `make_rtbm` Retry Loop

### The arcsine distribution of `T_ii` at initialisation

`random_init` with `diagonal_T=True` draws a diagonal matrix with entries $x_i \sim \text{Uniform}(-b, b)$ and then sets $T_{ii} = x_i^2$. The distribution of $x_i^2$ is arcsine-like on $[0, b^2]$, with PDF:

$$f(y) = \frac{1}{2b\sqrt{y}}, \quad y \in [0, b^2]$$

This is concentrated near **both** 0 and $b^2$. In particular:

$$P(T_{ii} < \varepsilon) = \frac{\sqrt{\varepsilon}}{b}$$

With $b = 2$ (the fixed `random_bound`) and $\sigma = \texttt{param\_bound} \times 0.1$:

$$P(T_{ii} < \sigma) = \frac{\sqrt{\sigma}}{2}$$

For `param_bound=5` ($\sigma = 0.5$): $P(T_{ii} < 0.5) \approx 0.35$ per entry. Across all $N_v = 4$ diagonal entries, the probability that **all** survive is only $(1-0.35)^4 \approx 0.18$. So in roughly 82% of initialisations, at least one $T_{ii}$ is dangerously close to zero.

When $T_{ii} \ll 1$, the first CMA-ES generation perturbs $W_{ij}$ from 0 by $\sim\sigma$, causing:

$$Q - W^T T^{-1} W \approx Q - N_v \frac{\sigma^2}{T_{ii}} \to -\infty$$

Every candidate in the population fails the Schur positivity check, all receive `NAN_PENALTY = 1e9`, and CMA-ES is stuck from iteration 1. This failure mode is **random-seed-dependent**: the previous run with `param_bound=5` happened to draw $T_{ii}$ values safely above $\sigma$; subsequent runs with different random states hit the trap systematically.

The failure becomes **worse for larger `param_bound`**: $\sigma$ grows while the $T_{ii}$ distribution is unchanged (still in $[0, 4]$ for `random_bound=2`), so the fraction of candidates with $T_{ii} < \sigma$ increases. This is why hyperopt trials at `param_bound > 5` fail universally.

### Fix: `make_rtbm` — dynamic `random_bound` and retry

The function `make_rtbm(nv, nh, param_bound)` replaces direct `RTBM(...)` calls and combines two mechanisms:

**1. Scale `random_bound` with `sqrt(param_bound)`**

Setting $b = \max(2,\, \sqrt{\texttt{param\_bound}})$ gives:

$$\frac{\mathbb{E}[T_{ii}]}{\sigma} = \frac{b^2/3}{\texttt{param\_bound} \times 0.1} = \frac{\texttt{param\_bound}/3}{\texttt{param\_bound} \times 0.1} = \frac{10}{3} \approx 3.3$$

The ratio $\mathbb{E}[T_{ii}]/\sigma$ is now **constant regardless of `param_bound`**, removing the systematic degradation at larger bounds. The first-generation Schur complement estimate:

$$Q - W^T T^{-1} W \approx \mathbb{E}[Q_{ii}] - N_v \frac{\sigma^2}{\mathbb{E}[T_{ii}]} \approx 3.3\sigma - 4 \times \frac{\sigma^2}{3.3\sigma} = 3.3\sigma - 1.2\sigma = 2.1\sigma > 0$$

remains well-positive in expectation.

**2. Retry loop until $T$ and $Q$ diagonals exceed $\sigma$**

Even with the scaling above, the arcsine variance means individual draws can still be near zero. The retry loop explicitly checks:

$$\forall i:\; T_{ii} > \sigma \quad\text{and}\quad Q_{ii} > \sigma$$

before accepting the initialisation. The probability of passing per attempt:

$$P(\text{pass}) = \left(1 - \frac{\sqrt{\sigma}}{b}\right)^{N_v} \cdot \left(1 - \frac{\sqrt{\sigma}}{b}\right)^{N_h} = \left(1 - \sqrt{\frac{\texttt{param\_bound} \times 0.1}{\texttt{param\_bound}}}\right)^{N_v + N_h} = \left(1 - \sqrt{0.1}\right)^{N_v + N_h}$$

For $N_v = 4, N_h \in \{2,3,4\}$: $P(\text{pass}) \approx (0.684)^{6\text{–}8} \approx 0.10\text{–}0.08$. The expected number of retries is $\sim 10$–$13$ — negligible overhead compared to a CMA-ES iteration.

```python
def make_rtbm(nv, nh, param_bound):
    random_bound = max(2.0, param_bound ** 0.5)
    sigma = param_bound * 0.1
    for _ in range(200):
        m = RTBM(nv, nh, init_max_param_bound=param_bound, random_bound=random_bound,
                 diagonal_T=True, mode=RTBM.Mode.LogProbability)
        if np.all(np.diag(m.t) > sigma) and np.all(np.diag(m.q) > sigma):
            return m
    return m
```

---

## 10. Summary of Changes

| Parameter | Old | New | Reason |
|-----------|-----|-----|--------|
| `diagonal_T` | `True` (workaround) | `False` + $W$ zeroed post-init | `diagonal_T=True` permanently restricts $T$; full $T$ captures feature correlations |
| CMA bounds | `param_bound` for all params | `max(param_bound, max_abs_init) × 1.2` | With full $T$, Schur-init diagonal entries ≈ $2\,\text{random\_bound}^2 \gg \text{param\_bound}$; fixed bounds cause immediate crash |
| `random_bound` | `1` | `max(2, sqrt(param_bound))` | Keeps $\mathbb{E}[T_{ii}]/\sigma \approx 3.3$ for any `param_bound`; previous fixed value of 2 caused systematic Schur failures at `param_bound > 5` |
| RTBM mode | `Probability` | `LogProbability` | Avoids float64 underflow in theta ratio as model trains |
| Cost function | `logarithmic` ($-\sum \log P$) | `sum` ($-\sum \log P$ via log-space input) | Consistent with LogProbability mode |
| `tolflatfitness` | `1` (default) | `maxiter` | Prevents premature stop during early flat landscape |
| `tolfun` | `1e-5` | `0` (disabled) | Range of all-penalty population = 0 < 1e-5 stops training after 1 iter for $N_h \geq 3$ |
| `param_bound` default | `20.0` | `5.0` | $\sigma=0.5$ keeps Schur complement well-separated in generation 1; avoids $10^9$ cost trap |
| Hyperopt search | `param_bound` $\in [5, 50]$ | `param_bound` $\in [1, 15]$ | Full range safe now that `random_bound` scales with `param_bound` |
| Hyperopt `n_hidden` | `{1, 2, 3}` | `{2, 3, 4}` | $N_h=1$ too inexpressive; extend upper range for better density fit |
| `SEARCH_MAXITER` | `80` | `150` | Longer per-trial budget so $N_h=3,4$ trials actually converge during hyperopt |
| Model creation | direct `RTBM(...)` | `make_rtbm(...)` | Retry loop ensures $T_{ii}, Q_{ii} > \sigma$ before handing off to CMA-ES |
| Error handling | none | `try/except LinAlgError` | Guards against marginal-Schur Cholesky crash in main process after training |
| $x_\text{vis}$ preprocessing | linear in $[0,1]$ | logit transform | Maps bounded flat distribution to logistic (Gaussian-like); initial model fits shape from generation 0 |
| $\eta$ preprocessing | $(η-2.5)/5$ | unchanged | Logit fails: events pile up at the lower acceptance boundary $\eta \approx -2.5$, creating a spike at $-9.2$ in logit space |
