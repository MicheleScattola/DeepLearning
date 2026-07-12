import numpy as np
import matplotlib.pyplot as plt

# PDF(y=x^2)
fig1, ax1 = plt.subplots()
bounds = [2, 4, 6]
for b in bounds:
    x2 = np.linspace(0.01, 3)
    ax1.plot(x2, 1/(2*b*np.sqrt(x2)), label=rf'$y \in [0,{b}^2]$')
ax1.set_title(r'$f(y)=\frac{1}{2b\sqrt{y}}$ with $y\in[0,b^2]$')
ax1.set_xlabel(r'$y$')
ax1.set_ylabel(r'$f(y)=f(x^2)$')
ax1.legend()
fig1.savefig('T_ii.png', dpi=300)

# Montecarlo for T block
Nv, Nh = 4, 2
n  = Nv + Nh          # matrix size: 6x6
b  = np.sqrt(2.0)     # random_bound for param_bound=5
N_trials = 50_000

diag_vals = []
offdiag_vals = []

for _ in range(N_trials):
    X = np.random.uniform(-b, b, (n, n))
    A = X.T @ X
    T = A[Nh:, Nh:]           # bottom-right Nv x Nv block
    for i in range(Nv):
        diag_vals.append(T[i, i])
    for i in range(Nv):
        for j in range(Nv):
            if i != j:
                offdiag_vals.append(T[i, j])

diag_vals   = np.array(diag_vals)
offdiag_vals = np.array(offdiag_vals)

fig2, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(diag_vals, bins=80, density=True, color='steelblue', edgecolor='none')
axes[0].axvline(np.mean(diag_vals), color='red', linestyle='--',
                label=rf'$\mathbb{{E}}[T_{{ii}}] = {np.mean(diag_vals):.2f}$')
axes[0].set_title(r'Diagonal entries $T_{ii}$')
axes[0].set_xlabel(r'$T_{ii}$')
axes[0].set_ylabel('density')
axes[0].legend()

axes[1].hist(offdiag_vals, bins=80, density=True, color='darkorange', edgecolor='none')
axes[1].axvline(np.mean(offdiag_vals), color='red', linestyle='--',
                label=rf'$\mathbb{{E}}[T_{{ij}}] = {np.mean(offdiag_vals):.2f}$')
axes[1].set_title(r'Off-diagonal entries $T_{ij}$, $i \neq j$')
axes[1].set_xlabel(r'$T_{ij}$')
axes[1].legend()

fig2.suptitle(
    rf'$A = X^T X$, $X \sim \mathrm{{Unif}}(-b,+b)$, $b=\sqrt{{{5}}}$'
    rf' — $T$ block ($N_v={Nv}$, $N_h={Nh}$), $n={N_trials}$ trials'
)
fig2.tight_layout()
fig2.savefig('T_ii_block.png', dpi=300)
print(f'Diagonal:    mean={np.mean(diag_vals):.3f}, std={np.std(diag_vals):.3f}, '
      f'min={np.min(diag_vals):.3f}')
print(f'Off-diagonal: mean={np.mean(offdiag_vals):.3f}, std={np.std(offdiag_vals):.3f}')