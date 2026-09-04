# Tutorial 08: PCA, Eigenspectra, and Anisotropy

## 1. What Problem Are We Trying to Solve?
How do we mathematically quantify the geometry and dimensionality of a high-dimensional representation matrix $H \in \mathbb{R}^{N \times d}$?

---

## 2. Mathematical Metrics

### 1. Roy & Vetterli Effective Rank
Let $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ be the singular values of $H$. Define the normalized singular value probability distribution:
$$
p_i = \frac{\sigma_i}{\sum_{j=1}^r \sigma_j}
$$
The Shannon entropy of $p$ is $H(p) = -\sum_{i=1}^r p_i \ln p_i$. The **effective rank** is:
$$
\text{erank}(H) = \exp(H(p))
$$
- Bounds: $1 \le \text{erank}(H) \le \min(N, d)$.
- Continuous, differentiable, and robust to small noise singular values.

### 2. Participation Ratio (PR)
Let $\lambda_1, \dots, \lambda_d$ be the eigenvalues of the sample covariance matrix $\Sigma$:
$$
\text{PR}(\Sigma) = \frac{\left( \sum_{i=1}^d \lambda_i \right)^2}{\sum_{i=1}^d \lambda_i^2} = \frac{\text{Tr}(\Sigma)^2}{\text{Tr}(\Sigma^2)}
$$
- If variance is evenly distributed across all $d$ directions: $\text{PR} = d$.
- If variance is concentrated along 1 direction: $\text{PR} = 1$.

---

## 3. Symmetry Breaking as Eigenvalue Splitting

When a system possesses a symmetric 2D invariant subspace, the covariance matrix restricted to that subspace is isotropic:
$$
\Sigma = \begin{pmatrix} \sigma^2 & 0 \\ 0 & \sigma^2 \end{pmatrix} \implies \lambda_1 = \lambda_2 = \sigma^2, \quad \text{PR} = 2
$$
When spontaneous symmetry breaking occurs, the network selects one direction over the other:
$$
\Sigma = \begin{pmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{pmatrix} \text{ with } \sigma_1^2 \gg \sigma_2^2 \implies \text{PR} \to 1
$$
Thus, symmetry breaking directly corresponds to **spectral degeneracy lifting** and **anisotropy growth**.

---

## 4. Python Implementation

```python
import torch
from symmetry_reps.geometry import effective_rank, participation_ratio, covariance_anisotropy

H = torch.randn(100, 16)
print("Effective Rank:", effective_rank(H))
print("Participation Ratio:", participation_ratio(H))
print("Anisotropy Index:", covariance_anisotropy(H))
```
