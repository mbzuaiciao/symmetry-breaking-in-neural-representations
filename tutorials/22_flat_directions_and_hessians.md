# Tutorial 22: Flat Directions and Loss Hessians

## 1. The Loss Hessian Matrix

Let $\mathcal{L}(\theta)$ be the empirical loss. The Hessian matrix $H_{\mathcal{L}} \in \mathbb{R}^{P \times P}$ (where $P = \dim(\theta)$) measures local curvature:
$$
H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}
$$
Eigendecomposition: $H_{\mathcal{L}} v_k = \lambda_k v_k$.

---

## 2. Spectrum Structure: Sharp vs. Flat Directions

In modern overparameterized neural networks, the Hessian spectrum consists of:
1. **Outlier Sharp Modes**: A small number of large positive eigenvalues corresponding to dominant task constraints.
2. **The Zero Bulk**: A vast majority ($\ge 80\%$) of eigenvalues clustered tightly around zero ($\lambda_k \approx 0$).

---

## 3. Origins of Flat Directions: Symmetry vs Overparameterization

> [!IMPORTANT]
> **Scientific Discipline**: Not every flat direction arises from continuous symmetry!
> - Continuous parameter symmetries (e.g. ReLU scaling directions) provide *exact* zero curvature directions ($\lambda = 0$).
> - Overparameterization ($P > N$) guarantees that the empirical Hessian has at least $P - N$ algebraic zero eigenvalues, regardless of symmetry.
> - Flat basins also arise from inactive neurons (dead ReLUs) and redundant features.
