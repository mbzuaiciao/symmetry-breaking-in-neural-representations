# Tutorial 18: When Symmetry Constraints Help: Low-Data Regimes

## 1. Inductive Bias and Sample Efficiency

When the true data-generating process $P(X, Y)$ is strictly invariant under group $G$, enforcing $G$-equivariance restricts the hypothesis space from $\mathbb{R}^{\mathcal{X}}$ to the invariant subspace $\mathbb{R}^{\mathcal{X} / G}$.

The effective volume of the input domain is reduced by a factor of $|G|$:
$$
\text{Sample Complexity}_{\text{sym}} \approx \frac{1}{|G|} \text{Sample Complexity}_{\text{unconstrained}}
$$

---

## 2. Empirical Validation (Experiment E)

In our experiments across sample sizes $N \in [24, 384]$:
- At $N = 24$ samples: Invariant MLP achieves $\approx 92\%$ accuracy, while the standard MLP struggles around $\approx 68\%$.
- At $N \ge 200$ samples: Standard MLP catches up to $\approx 95\%+$.

Hard-coded symmetry constraints provide their greatest advantage in **data-scarce regimes**.
