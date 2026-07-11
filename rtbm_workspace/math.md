# RTBM Training: Numerical Issues and Fixes

## 1. The RTBM Probability Formula

The RTBM assigns probability density to a visible vector $v \in \mathbb{R}^{N_v}$ via:

$$P(v) = \sqrt{\frac{\det T}{(2\pi)^{N_v}}} \cdot \underbrace{e^{-\frac{1}{2}v^T T v - B_v^T v - \frac{1}{2}B_v^T T^{-1} B_v}}_{\text{Gaussian envelope}} \cdot \underbrace{\frac{\theta\!\left(\frac{v^T W + B_h^T}{2\pi i} \;\Big|\; \frac{-Q}{2\pi i}\right)}{\theta\!\left(\frac{B_h^T - B_v^T T^{-1} W}{2\pi i} \;\Big|\; \frac{-Q + W^T T^{-1} W}{2\pi i}\right)}}_{\text{theta ratio}}$$

The Gaussian envelope captures the bulk shape. The **theta ratio** captures non-Gaussian structure via the hidden units: the numerator is event-dependent (through $v^T W$), while the denominator is a normalisation constant.

The Riemann theta function itself is:

$$\theta(z \mid \Omega) = \sum_{n \in \mathbb{Z}^{N_h}} \exp\!\left(2\pi i \left(\tfrac{1}{2} n^T \Omega n + n^T z\right)\right)$$

In Phase I (the setting used here), $z$ is purely imaginary and $\Omega$ is purely imaginary with positive-definite imaginary part, so the sum becomes a real decaying exponential and is always convergent and positive. The library factors this as $\theta = e^u \cdot v$ where $u$ is the dominant exponential growth and $v$ is the bounded oscillatory part.

### Theta ratio at $W=0$: convergence and the Schur distinction

Substituting the period matrix $\Omega = -Q/(2\pi i) = iQ/(2\pi)$ and argument $z(v) = (W^T v + B_h)/(2\pi i)$ into the lattice sum:

$$\theta\!\left(z(v)\,\bigg|\,\frac{iQ}{2\pi}\right) = \sum_{n \in \mathbb{Z}^{N_h}} \exp\!\left(-\tfrac{1}{2}n^T Q n + n^T(W^T v + B_h)\right)$$

At **$W=0$, $B_h=0$** the argument is identically zero for every event $v$, so numerator equals denominator:

$$\tilde{\theta}(v) = \frac{\theta(0\,|\,iQ/2\pi)}{\theta(0\,|\,iQ/2\pi)} = 1 \qquad \forall\, v$$

and the model reduces to a pure Gaussian. This is why zeroing $W$ after `random_init` (Section 3) gives a numerically stable starting point.

**Convergence of $\theta(0\,|\,iQ/2\pi)$.** The series $\sum_n e^{-\frac{1}{2}n^T Q n}$ converges whenever $Q > 0$ (positive definite). Since

$$n^T Q n \geq \lambda_{\min}(Q)\,|n|^2 \to \infty \quad\text{as } |n| \to \infty,$$

each term decays like $e^{-\frac{\lambda_{\min}}{2}|n|^2}$ — a lattice Gaussian — and the series is absolutely summable. $Q > 0$ is guaranteed at initialisation by construction, so the theta function is always well-defined and finite at $W=0$.

**Convergence at a single $v$ with $W \neq 0$.** For any fixed event $v$, the numerator $\sum_n e^{-\frac{1}{2}n^T Q n + n^T(W^T v + B_h)}$ also converges: the linear term $n^T(W^T v + B_h)$ grows as $|n|$ but the quadratic $-\frac{1}{2}n^T Q n$ grows as $|n|^2$, so the quadratic always wins for large $|n|$. The theta series is finite at every fixed $v$ as long as $Q > 0$, regardless of $W$.

**The Schur complement is about normalisability, not pointwise convergence.** These two conditions are separate:

| Condition | What it guarantees |
|---|---|
| $Q > 0$ | $\theta(z(v)\,|\,iQ/2\pi) < \infty$ at each fixed $v$ |
| $Q - W^T T^{-1} W > 0$ (Schur) | $Z = \int P(v)\,dv < \infty$ — the model is normalisable |

With $W \neq 0$, the Schur complement controls whether integrating $P(v)$ over *all* $v$ converges. Completing the square in $h$ and then integrating over $v$ yields an effective Gaussian in $v$ with precision $T - WQ^{-1}W^T$. If this matrix is not positive definite, $P(v) \to \infty$ in some direction as $|v| \to \infty$ and no proper distribution exists. $Q > 0$ alone cannot prevent this.

**Why $W \neq 0$ is also computationally expensive.** With non-zero $W$ the lattice peak shifts to $n^* = Q^{-1}(W^T v + B_h)$, and the theta library must enumerate all lattice points within radius $R$ of the origin, where $R$ grows with the displacement:

$$R \sim \frac{|W^T v|}{\sqrt{\lambda_{\min}(Q)}}$$

The number of lattice points scales as $R^{N_h}$. At $W=0$ the peak is always at $n=0$, $R$ is small, and the sum is fast and cheap for every event — an additional reason (beyond the theta ratio = 1 initialisation benefit) to start from $W=0$ before releasing $W$ to CMA-ES training.

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

## 9. Fix: Small-$T_{ii}$ Initialisation Trap and `make_rtbm` Retry Loop

### The distribution of `T_ii` at initialisation

`random_init` with `diagonal_T=True` draws a diagonal matrix with entries $x_i \sim \text{Uniform}(-b, b)$ and then sets $T_{ii} = x_i^2$. The CDF of $T_{ii} = x_i^2$ follows from $P(x_i^2 \leq y) = P(|x_i| \leq \sqrt{y}) = \sqrt{y}/b$, giving PDF:

$$f(y) = \frac{1}{2b\sqrt{y}}, \quad y \in [0, b^2]$$

This is **monotonically decreasing** — it diverges at $y=0$ and falls to $1/(2b^2)$ at $y=b^2$, so the mass is concentrated near 0 only. In particular:

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

The key requirement is that $\mathbb{E}[T_{ii}]/\sigma$ stays **constant regardless of `param_bound`**, so the Schur safety margin does not degrade as `param_bound` grows. Setting $b = \sqrt{\texttt{param\_bound}}$ achieves this in both the diagonal and full-$T$ cases:

**Diagonal $T$** (`diagonal_T=True`): $T_{ii} = x_i^2$ for a single $x_i \sim \text{Uniform}(-b,b)$, so $\mathbb{E}[T_{ii}] = b^2/3$:

$$\frac{\mathbb{E}[T_{ii}]}{\sigma} = \frac{b^2/3}{\texttt{param\_bound} \times 0.1} = \frac{\texttt{param\_bound}/3}{\texttt{param\_bound} \times 0.1} = \frac{10}{3} \approx 3.3$$

**Full $T$** (`diagonal_T=False`, the actual implementation): $T_{ii} = \sum_{j=1}^{N_v+N_h} X_{ji}^2$ is a sum of $N_v+N_h = 6$ squared uniforms, so $\mathbb{E}[T_{ii}] = 6 \times b^2/3 = 2b^2$:

$$\frac{\mathbb{E}[T_{ii}]}{\sigma} = \frac{2b^2}{\texttt{param\_bound} \times 0.1} = \frac{2\,\texttt{param\_bound}}{\texttt{param\_bound} \times 0.1} = 20$$

In both cases the ratio is a fixed constant once $b = \sqrt{\texttt{param\_bound}}$. The full-$T$ margin (20) is larger than the diagonal-$T$ margin (3.3) because $T_{ii}$ is the sum of six independent squared terms rather than one. The first-generation Schur complement estimate (full-$T$ case):

$$Q - W^T T^{-1} W \approx \mathbb{E}[Q_{ii}] - N_v \frac{\sigma^2}{\mathbb{E}[T_{ii}]} \approx 20\sigma\cdot\tfrac{1}{10} - 4\times\frac{\sigma^2}{20\sigma\cdot\tfrac{1}{10}} \approx 2\sigma - 0.2\sigma = 1.8\sigma > 0$$

remains well-positive in expectation.

**2. Retry loop until $T$ and $Q$ diagonals exceed $\sigma$**

Even with the scaling above, the arcsine variance means individual draws can still be near zero. The retry loop explicitly checks:

$$\forall i:\; T_{ii} > \sigma \quad\text{and}\quad Q_{ii} > \sigma$$

before accepting the initialisation. The probability of passing per attempt:

$$P(\text{pass}) = \left(1 - \frac{\sqrt{\sigma}}{b}\right)^{N_v} \cdot \left(1 - \frac{\sqrt{\sigma}}{b}\right)^{N_h} = \left(1 - \sqrt{\frac{\texttt{param\_bound} \times 0.1}{\texttt{param\_bound}}}\right)^{N_v + N_h} = \left(1 - \sqrt{0.1}\right)^{N_v + N_h}$$

For $N_v = 4, N_h \in \{2,3,4\}$: $P(\text{pass}) \approx (0.684)^{6\text{–}8} \approx 0.10\text{–}0.08$. The expected number of retries is $\sim 10$–$13$ — negligible overhead compared to a CMA-ES iteration.

```python
def make_rtbm(nv, nh, param_bound, max_tries=200):
    random_bound = param_bound ** 0.5          # keeps E[T_ii]/sigma constant
    sigma = param_bound * 0.1
    for _ in range(max_tries):
        m = RTBM(nv, nh, init_max_param_bound=param_bound, random_bound=random_bound,
                 diagonal_T=False, mode=RTBM.Mode.LogProbability)
        if np.all(np.diag(m.t) > sigma) and np.all(np.diag(m.q) > sigma):
            params = np.real(m.get_parameters()).copy()
            params[nv + nh : nv + nh + nv * nh] = 0.0   # zero W
            if m.set_parameters(params):
                actual_max = float(np.max(np.abs(params)))
                m.set_bounds(max(param_bound, actual_max) * 1.2)
                return m
    raise RuntimeError(f"make_rtbm: no valid initialisation in {max_tries} tries")
```

**3. Widening CMA bounds for the full $T$ matrix**

The theta library's `RTBM.set_bounds(param\_bound)` sets the CMA search box as $[-\texttt{param\_bound},\, +\texttt{param\_bound}]$ **uniformly for all parameters**. This was designed for `diagonal_T=True`, where each T entry is $T_{ii} = x_i^2$ with $x_i \sim \text{Uniform}(-b,\, b)$ and $b = \sqrt{\texttt{param\_bound}}$, so $T_{ii} \in [0, \texttt{param\_bound}]$ — safely inside the bounds.

With `diagonal_T=False` the T matrix is the bottom-right block of $A = X^T X$ (a full 6×6 PSD matrix). Its diagonal entries are now **sums of squares** across all six rows of $X$:

$$T_{ii} = \sum_{j=1}^{6} X_{ji}^2, \qquad X_{ji} \sim \text{Uniform}(-b,\, b)$$

With $b = \sqrt{5} \approx 2.24$ for $\texttt{param\_bound} = 5$: each squared term is at most $5$, and the sum of six gives $\mathbb{E}[T_{ii}] \approx 2b^2 = 10$, with the maximum around $6b^2 = 30$. The initial T diagonal is therefore **$2\times$ to $6\times$ larger than param\_bound**, placing the initial solution outside the CMA bounds before training even starts. CMA raises a `ValueError` immediately.

The fix is to measure the actual largest parameter in the initial vector and set bounds wide enough to contain it:

```python
actual_max = float(np.max(np.abs(params)))
m.set_bounds(max(param_bound, actual_max) * 1.2)
```

A natural concern is that widening the bounds also increases sigma — since `train_rtbm` uses `sigma = max(bounds) * 0.1` — potentially causing Schur violations in the first generation. The two effects cancel precisely because they share the same root cause: a larger full-matrix $T$ means both larger initial parameter values (hence wider bounds, hence larger sigma) **and** a larger Schur complement margin. Concretely:

$$Q - W^T T^{-1} W \approx 10 - \frac{4 \times 1.5^2}{10} \approx 9.1 \gg 0$$

With full $T \approx 10$ and $\sigma \approx 1.5$, the Schur complement is $\sim 9$, compared to $\sim 0.6$ with diagonal $T \approx 1.3$ and $\sigma = 0.5$. The enlarged sigma is not only safe — it is appropriate: the initial T needs to travel from $\sim 10$ down to the optimal $\sim 1$ (for standardised data), a distance of $\sim 9$ covered in about 6 sigma steps. The sigma and the parameter scale are commensurate by construction.

---

## 10. Fix: `gen_timeout` Prevents Riemann Theta Lattice-Sum Hangs

### A new failure mode at higher $N_h$: valid parameters, unbounded computation

All prior failure modes (Sections 2, 7) produce a clean, fast signal: `set_parameters()` returns `False`, or the cost evaluates to a finite-but-huge `NAN_PENALTY`-worthy value. During a 45-run sweep over $(N_h, \text{param\_bound})$, a qualitatively different failure appeared at $N_h = 4$: a single CMA-ES candidate caused one worker process to spin at 100% CPU for over 20 minutes without crashing, returning, or raising any exception — silently blocking the entire `multiprocessing.Pool.map()` call (and with it, the whole sweep) indefinitely.

### Why this happens

The theta library evaluates $\theta(z \mid \Omega)$ by truncating the infinite lattice sum to points within a radius $R$ of the origin, where $R$ is computed by `radius()` from the requested accuracy $\varepsilon = 10^{-8}$ and the Cholesky factor of $\text{Im}(\Omega)$. Schematically, $R$ grows as the smallest eigenvalue of $\text{Im}(\Omega)$ shrinks:

$$R \sim \sqrt{\frac{-\log \varepsilon}{\lambda_{\min}(\text{Im}\,\Omega)}}$$

The number of lattice points enumerated by `integer_points_python(g, R, T)` within that radius scales as $R^g$, where $g = N_h$ is the genus of the theta function. A candidate with $T$ or $Q$ positive definite but **numerically close to singular** (passes `check_pos_def`'s `eigenvalues > 0` check with an eigenvalue at, say, $10^{-6}$) gives a large $R$. For $g = 2$ this still enumerates a modest number of points; for $g = 4$, $R^4$ versus $R^2$ is a difference that can turn a millisecond computation into one that does not finish in any practical time.

This is distinct from the Schur-collapse failure mode: there, `Q - W^T T^{-1} W$ is **not** positive definite and `set_parameters()` rejects the candidate before the theta function is ever evaluated. Here, the candidate is **valid** — it just summons an enormous, but finite, computation.

### Why `multiprocessing.Pool` cannot recover on its own

`pool.map()` is synchronous: it waits for *all* dispatched tasks to return before yielding control back to the caller. With `popsize` candidates split across `ncores` workers, one stuck worker blocks the entire batch — the other 13 workers in a 14-worker pool can finish in seconds and then sit idle, while the main process waits forever for the 14th. `Pool` provides no built-in mechanism to cancel an in-flight task.

### Fix: a per-generation timeout with forced pool termination

```python
try:
    fits = pool.map_async(worker_compute, candidates).get(timeout=gen_timeout)
except mp.TimeoutError:
    raise TrainingTimeout(...)
finally:
    pool.terminate()   # forceful kill, not pool.close() (which waits for pending tasks)
    pool.join()
```

`map_async(...).get(timeout=...)` raises `multiprocessing.TimeoutError` if results aren't ready in time, without needing to know which specific worker is stuck. The `finally: pool.terminate()` then forcibly kills *all* workers (including the hung one) rather than `pool.close()`, which would itself wait for the stuck task to finish — defeating the purpose. `TrainingTimeout` is a `RuntimeError` subclass, so it is caught by the same `except Exception` blocks that already handle Schur-failure crashes in `sweep.py` and the hyperopt objective in `training.py` — a single pathological $(N_h, \text{param\_bound})$ combination now fails that one run within `gen_timeout` seconds (default 60s, configurable via `sweep.py --gen_timeout`) instead of hanging forever.

The single-core path (`ncores=1`) uses `signal.alarm` for the same effect, since there is no worker process to terminate — the alarm simply interrupts the current call stack with a `TrainingTimeout`.

### Sub-timeout cost: the same mechanism degrades "successful" runs too

With `gen_timeout` in place (no hangs), a 45-run sweep still showed $N_h=4$'s wall-clock time varying **over 10x** across `param_bound` — `param_bound=1` runs took 28–56s/core, `param_bound=8` runs took 5–7s/core, at the same `ncores=8` and the same number of generations (`maxiter` is fixed by `feval_budget // popsize`, independent of `param_bound`). This is not noise: sorted by `param_bound`, the times form a clean monotonic trend, not a scatter.

The saved `valid_fraction` history of the slowest run (`pb=1`, 55.8s/core) versus a fast one (`pb=8`, 7.3s/core), both 200 generations:

| | mean valid fraction | time / core |
|---|---|---|
| `pb=1` | 0.978 | 55.8s |
| `pb=8` | 0.697 | 7.3s |

The valid-fraction ratio (1.4×) accounts for only a fraction of the 7.6× time difference. The rest comes from **how expensive each valid evaluation is**, which depends on how close to singular the evaluated $T/Q$ actually are — not merely on whether they pass the `eigenvalues > 0` check.

**Mechanism.** At small `param_bound`, $\sigma = \text{param\_bound} \times 0.1$ is tiny, so CMA-ES barely moves from its starting point for the entire run. That starting point was accepted by `make_rtbm`'s retry loop on a loose bar ($\text{diag}(T) > \sigma = 0.1$ at $\text{pb}=1$) — i.e. only just barely non-singular. Since CMA cannot escape that neighbourhood with such a small step, essentially every candidate it evaluates for the *whole run* sits near that same marginally-singular region, inflating $R$ (and hence the $R^{N_h}$ lattice-point count, Section 10 above) on nearly every single evaluation. At large `param_bound`, CMA explores broadly; a valid candidate reached via a large random jump is statistically far less likely to land exactly near the singular boundary than one reached via a tiny jump that started right next to it — so when CMA does land on a valid candidate, it tends to be cheaper to evaluate, even though fewer candidates qualify as valid at all.

**Why this compounds specifically at high $N_h$.** From the $R^g$ scaling above ($g = N_h$), the same "trapped near a marginal boundary" effect that costs $N_h=2$ a factor of $R^2$ costs $N_h=4$ a factor of $R^4$. A given increase in $R$ from sitting near-singular compounds dramatically more as $N_h$ grows. This is why pooling timing data across `param_bound` for `compute_scaling.png` — the statistically correct choice, since compute cost should generically depend on $N_h$ rather than `param_bound` — surfaces a genuinely **bimodal** cost distribution at $N_h=4$ that does not average out even at $n=28$: it is not sampling noise that a larger sample would shrink, it is two qualitatively different regimes (cheap-and-exploring vs. expensive-and-trapped) whose population split shifts with $N_h$.

### Practical takeaway

Higher $N_h$ is not free even when CMA-ES navigates the Schur constraint successfully — the theta function's intrinsic cost can blow up combinatorially in $N_h$ for borderline-singular parameters that occur naturally during exploration, both catastrophically (the hang above) and gradually (the 10x cost spread above). Both failure modes get worse at *small* `param_bound`, not large — counter-intuitively, the "safer", smaller step size is also the one that traps CMA-ES near a marginal, expensive-to-evaluate boundary for the entire run. Sweeps including $N_h \geq 4$ should always run with a finite `gen_timeout`, and timing comparisons across $N_h$ should pool across `param_bound` rather than fix it, since the cost distribution itself is part of what differs between architectures.

---

## 11. Fix: `param_bound` Silently Had No Effect Below ~4

### Symptom

A sweep with an expanded `param_bound` grid (`1, 3, 5, 8, 10`) and more seeds (5 instead of 3) produced heatmap and scaling plots that looked unexpectedly flat across $N_h \in \{2, 3\}$. Direct inspection of `sweep_results.csv` showed something far more specific than statistical smoothing: for the same `(n_hidden, seed)`, **`param_bound = 1.0`, `2.0`, and `3.0` produced bit-for-bit identical `val_NLL`** (e.g. `5.13235764507767` exactly twice), and identical *crashes* (all three giving the `1e9` sentinel together). `param_bound \geq 5` varied normally.

### Root cause: two compounding overrides

**(a) The `random_bound` floor.** `make_rtbm` set:

```python
random_bound = max(2.0, param_bound ** 0.5)
```

For any $\texttt{param\_bound} \leq 4$, $\sqrt{\texttt{param\_bound}} \leq 2$, so `random_bound` was **pinned at the floor value of 2.0** regardless of the actual requested `param_bound`. Since `random_init` draws from $\text{Uniform}(-\text{random\_bound}, +\text{random\_bound})$, the same seed produced the **identical initial model** for every `param_bound` in $[0, 4]$.

**(b) The bounds-widening override.** `make_rtbm` then set:

```python
m.set_bounds(max(param_bound, actual_max) * 1.2)
```

Because the initial model from (a) is identical across this range, `actual_max` (the largest absolute parameter value) is identical too — and for a full-$T$ matrix (Section 3), $\texttt{actual\_max}$ routinely exceeds 3–4 (diagonal entries are sums of six squared draws). So $\max(\texttt{param\_bound}, \texttt{actual\_max})$ evaluates to $\texttt{actual\_max}$ **regardless of the nominal `param_bound`**, making the CMA bounds — and therefore `train_rtbm`'s `sigma = \max(\text{bounds}) \times 0.1` — identical as well.

Identical initial model + identical bounds + identical sigma + identical CMA seed $\Rightarrow$ identical optimisation trajectory. `param_bound \in \{1, 2, 3\}$ were silently testing the exact same configuration in *every* sweep run conducted so far, including the very first one — the duplication was only noticed once a wider grid made it visually obvious in the heatmap.

### Fix

**Remove the floor.** `random_bound = param_bound ** 0.5` (no `max(2.0, ...)`). This is safe: the Schur-margin invariant from Section 9 is preserved automatically, because both `random_bound` and `sigma` scale with `param_bound` — for a full-$T$ matrix, $\mathbb{E}[T_{ii}] \approx 2 \cdot \texttt{random\_bound}^2 = 2\,\texttt{param\_bound}$, while $\sigma = 0.1\,\texttt{param\_bound}$, giving a constant ratio $\mathbb{E}[T_{ii}]/\sigma = 20$ independent of `param_bound`.

**Decouple `sigma` from the widened bounds.** `train_rtbm` now accepts an explicit `init_sigma` parameter; callers (`sweep.py`, `training.py`) pass `param_bound * 0.1` directly instead of letting `sigma` be inferred from `max(model.get_bounds()[1])`. The box-constraint bounds still need widening to contain the initial point (Section 9.3), but the *exploration step size* now always reflects the literal requested `param_bound`, never silently overridden by however much the bounds needed to grow.

Verified directly: with the fix, `(n_hidden=2, seed=1)` at `param_bound = 1.0, 2.0, 3.0` now gives three different initial models (max $|param|$ = 3.8, 7.6, 11.5 respectively) and three different `val_NLL` after training.

### Secondary issue: `sweep.py` appending duplicate rows

Independent of the bug above, re-running `sweep.py` with an overlapping `--pb`/`--seeds` grid against an existing `--out` CSV appended duplicate `(n_hidden, param_bound, seed)` rows rather than skipping them (the file is opened in append mode so an interrupted sweep — Section 10 — can resume without re-running completed combos). Fixed by reading existing rows on startup and skipping any combo already present.

### Practical impact

Every `param_bound` sweep run prior to this fix had effectively only 2-3 distinct `param_bound` values represented (the low end collapsed into one point, only the points above the collapse threshold were genuinely distinct). `best_pb_per_nh`'s selection and any conclusions drawn about the *shape* of the `param_bound` response (as opposed to just "low vs high") from that data should be treated as unreliable. The summary.md results comparing $N_h$ remain valid in their qualitative conclusion (nh=3 as the reliable middle ground), since the crash-rate finding was independent of which specific low `param_bound` value was tested, but the `heatmap_pb_nh.png` shape at low `param_bound` does not reflect genuine variation and should be regenerated from a fresh sweep.

---

## 12. Why `param_bound` < 10: Competing Effects on Training Stability

The choice of `param_bound` is governed by two competing failure modes that pull in opposite directions, establishing a practical upper limit around 8–10.

### Large `param_bound` → Gaussian envelope degenerates

The Gaussian factor in $P(v)$ is $\exp(-\frac{1}{2} v^T T v)$. As CMA-ES explores large `param_bound` values, $T$ entries grow large and the Gaussian becomes extremely narrow — for standardised data with unit-scale features, large $T_{ii}$ assigns near-zero probability to almost every training event, driving NLL → ∞. Training then consists almost entirely of `NAN_PENALTY` candidates: `mean_valid_fraction` collapses and CMA-ES cannot find a gradient direction to improve.

Separately, a large search box means the fraction of parameter space that corresponds to physically reasonable distributions (moderate $T_{ii}$, well-separated Schur complement) is tiny. CMA-ES must find this small region from a starting point that is already outside it (full-$T$ diagonal entries initialise at $\sim 2\,\texttt{param\_bound}$, far from the optimal $\sim O(1)$ for standardised data), spending most of its evaluation budget navigating back.

### Small `param_bound` → Riemann theta lattice sum hangs

The theta lattice sum converges over a radius (Section 10):

$$R \sim \sqrt{\frac{-\log \varepsilon}{\lambda_{\min}(\text{Im}\,\Omega)}}$$

where $\lambda_{\min}$ is the smallest eigenvalue of $T$ (roughly). When $T$ entries are small — from initialisation or because CMA drifts toward zero during training — $\lambda_{\min} \to 0$ and $R \to \infty$. The number of lattice points grows as $R^{N_h}$, which for $N_h = 4$ can turn a millisecond evaluation into an indefinite hang (Section 10). This is exactly the `TrainingTimeout` failure mode.

Even without hanging, small `param_bound` means $\sigma = \texttt{param\_bound} \times 0.1$ is tiny. The `make_rtbm` retry loop accepts initialisations where $T_{ii}$ just barely exceeds $\sigma$, so the entire CMA-ES run explores a neighbourhood near the Schur boundary — marginally non-singular, expensive to evaluate. As Section 10 shows, `pb=1` runs take $\sim 7\times$ longer per generation than `pb=8` runs at the same `maxiter`, precisely because every evaluation sits near that expensive boundary.

### The natural scale and the sweet spot

For standardised data (zero mean, unit variance), the optimal $T$ entries are $O(1)$: the Gaussian envelope should match the data covariance, which is approximately $I_{N_v}$ after standardisation. The `param_bound` sweep (Section 11) explores $\{1, 2, 3, 5, 8\}$; values in the range $[2, 5]$ consistently give the best AUC and `mean_valid_fraction`. Beyond $\sim 8$ the valid fraction degrades sharply (too much parameter space is infeasible); below $\sim 2$ the theta evaluation slows dramatically (training sits near the singularity for the whole run).

### Connection to the `gen_timeout`

The same asymmetry explains why `gen_timeout` matters more at small `param_bound` than large. At `pb=1`, valid candidates are rare but cheap to produce via rejection — they just cluster near the Schur boundary, where the theta sum is slow. At `pb=8`, valid candidates are rarer still (low `mean_valid_fraction`) but those that do appear are typically far from the boundary and evaluate quickly. The hanging risk (a single candidate making $R^{N_h}$ blow up) is paradoxically higher at small `param_bound`, where borderline-singular parameters are the norm rather than the exception.

---

## 13. Summary of Changes

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
| `train_rtbm` generation timeout | none | `gen_timeout=60s` via `pool.map_async().get(timeout=...)` + forced `pool.terminate()` | A near-singular (but valid) $T/Q$ at $N_h \geq 4$ can make the theta lattice sum hang a worker indefinitely; `pool.map()` has no built-in cancellation, so one stuck candidate blocks the whole sweep forever without a timeout |
| `random_bound` floor | `max(2, sqrt(param_bound))` | `sqrt(param_bound)` (no floor) | Floor pinned the initial draw identical for any param_bound $\leq 4$, silently collapsing those sweep points into duplicates |
| `train_rtbm` sigma source | `max(model.get_bounds()[1]) * 0.1` | explicit `init_sigma = param_bound * 0.1` | Widened bounds (Sec. 9.3) could make `sigma` identical across different param_bound values too, compounding the collapse above |
| `sweep.py` resume behaviour | blind append | skip `(nh, pb, seed)` already in CSV | Re-running with an overlapping grid against an existing `--out` file silently duplicated rows |
