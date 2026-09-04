# Tutorial 11: CCA and Subspace Similarity

## 1. What Question Does Each Similarity Metric Answer?

When comparing representation matrices $X_1, X_2 \in \mathbb{R}^{N \times d}$, different metrics test different invariances:

| Metric | Group Invariance | Geometric Meaning | Best Used When |
| :--- | :--- | :--- | :--- |
| **PCA Subspace Angles** | Unitary $U(k)$ within top-$k$ subspace | Principal angles between dominant variance planes | Comparing principal variance directions |
| **Orthogonal Procrustes** | Orthogonal $O(d)$ | Minimum Euclidean point-to-point discrepancy under rigid rotation | Measuring isometric point cloud alignment |
| **Linear CKA** | $O(d) \times \mathbb{R}_{>0}$ (Rotations + Isotropic Scale) | Correlation of centered relational Gram kernels | Comparing layer-to-layer and cross-architecture representations |
| **Canonical Correlation (CCA)** | General Linear $GL(d)$ | Maximally correlated linear projections | Finding shared 1D latent semantic features |

---

## 2. Canonical Correlation Analysis (CCA) Formulation

Given centered $X_1 \in \mathbb{R}^{N \times d_1}$ and $X_2 \in \mathbb{R}^{N \times d_2}$, CCA finds projection vectors $w_1 \in \mathbb{R}^{d_1}, w_2 \in \mathbb{R}^{d_2}$ maximizing Pearson correlation:
$$
\max_{w_1, w_2} \frac{w_1^T X_1^T X_2 w_2}{\sqrt{(w_1^T X_1^T X_1 w_1)(w_2^T X_2^T X_2 w_2)}}
$$

---

## 3. Subspace Principal Angles

Let $U_1 \in \mathbb{R}^{d \times k}$ and $U_2 \in \mathbb{R}^{d \times k}$ be orthonormal bases for the top-$k$ principal subspaces. The singular values of $M = U_1^T U_2$ are the cosines of the **principal angles**:
$$
\cos(\theta_i) = \sigma_i(U_1^T U_2), \quad i = 1, \dots, k
$$
- $\theta_i = 0$: Subspaces completely coincide.
- $\theta_i = \pi / 2$: Subspaces are mutually orthogonal.
