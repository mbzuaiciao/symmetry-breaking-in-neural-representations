# Tutorial 10: Orthogonal Procrustes Alignment

## 1. What Problem Are We Trying to Solve?
If two independently trained neural networks learn identical representation geometries that differ only by an arbitrary latent rotation or reflection $R$, how do we find the optimal transformation to align them?

---

## 2. Mathematical Formulation of the Orthogonal Procrustes Problem

Given two centered representation matrices $X_1, X_2 \in \mathbb{R}^{N \times d}$, find the orthogonal matrix $R \in O(d)$ (satisfying $R^T R = I$) that minimizes the Frobenius distance:
$$
\min_{R \in O(d)} \|X_1 R - X_2\|_F^2
$$

Expanding the Frobenius norm squared:
$$
\|X_1 R - X_2\|_F^2 = \text{Tr}(R^T X_1^T X_1 R) - 2 \text{Tr}(X_2^T X_1 R) + \text{Tr}(X_2^T X_2)
$$
Because $\text{Tr}(R^T X_1^T X_1 R) = \text{Tr}(X_1^T X_1)$ is constant for orthogonal $R$, minimizing the distance is equivalent to **maximizing the cross-covariance trace**:
$$
\max_{R \in O(d)} \text{Tr}((X_1^T X_2)^T R)
$$

---

## 3. Exact Closed-Form SVD Solution

Let $M = X_1^T X_2 \in \mathbb{R}^{d \times d}$ be the cross-covariance matrix. Compute its Singular Value Decomposition (SVD):
$$
M = U \Sigma V^T
$$
The optimal orthogonal Procrustes rotation matrix is:
$$
R^* = U V^T
$$

The **normalized Procrustes disparity** is:
$$
\mathcal{D}(X_1, X_2) = \frac{\|X_1 R^* - X_2\|_F}{\|X_2\|_F}
$$

---

## 4. Python Implementation

```python
import torch
from symmetry_reps.geometry import orthogonal_procrustes

X1 = torch.randn(100, 4)
# Apply known rotation
R_true = torch.tensor(
    [[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
X2 = X1 @ R_true

R_est, disparity, X1_aligned = orthogonal_procrustes(X1, X2)
print("Estimated Disparity:", disparity)  # < 1e-6
```
