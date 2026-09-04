# Tutorial 12: Intrinsic Dimension and Representation Collapse

## 1. Ambient vs. Intrinsic Dimension

Neural representations live in a high-dimensional ambient space $\mathbb{R}^d$ (e.g., $d = 512$ or $4096$). However, the data distribution and task constraints confine the activations to a low-dimensional submanifold of intrinsic dimension $d_{\text{int}} \ll d$.

---

## 2. Estimating Intrinsic Dimension

### 1. Spectral Estimators (Participation Ratio & Effective Rank)
- **Effective Rank**: $\text{erank}(X) = \exp\left(-\sum p_i \ln p_i\right)$
- **Participation Ratio**: $\text{PR}(\Sigma) = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$

### 2. Nearest Neighbor / Manifold Estimators (TWO-NN)
Based on the ratio of distances to the second and first nearest neighbors:
$$
\mu_i = \frac{r_{i, 2}}{r_{i, 1}}
$$
Under the assumption of local Poisson point processes, the cumulative distribution function of $\mu$ yields maximum likelihood estimates of local manifold dimension.

---

## 3. Connection to Symmetry Breaking and Collapse

- **Isotropic State**: Before symmetry breaking, variance is distributed evenly across symmetric dimensions ($d_{\text{int}} \ge 2$).
- **Symmetry Breaking Collapse**: Optimization forces the representation into an anisotropic lower-dimensional manifold ($d_{\text{int}} \to 1$), collapsing irrelevant orthogonal variations.
