# Tutorial 09: Representation Similarity and Linear CKA

## 1. What Problem Are We Trying to Solve?
How do we compare the internal representations $X \in \mathbb{R}^{N \times d_1}$ and $Y \in \mathbb{R}^{N \times d_2}$ learned by two independently trained neural networks (or different layers of the same network) when their neuron counts and coordinate systems differ?

---

## 2. Why Linear CKA (Centered Kernel Alignment)?

Naive metrics such as Canonical Correlation Analysis (CCA) or raw Euclidean activation distance suffer from severe limitations:
- Standard CCA is invariant to any invertible linear transformation, making it overly sensitive to small directions of meaningless noise when $d > N$.
- Raw Euclidean distance depends heavily on the coordinate basis.

**Linear CKA** (Kornblith et al., 2019) provides an optimal balance:
- Invariant to orthogonal transformations (rotations and reflections): $\text{CKA}(X R_1, Y R_2) = \text{CKA}(X, Y)$ for $R_1 \in O(d_1), R_2 \in O(d_2)$.
- Invariant to isotropic scaling: $\text{CKA}(c_1 X, c_2 Y) = \text{CKA}(X, Y)$ for $c_1, c_2 > 0$.
- Not invariant to arbitrary non-orthogonal linear transformations (preserves geometric shape).

---

## 3. Mathematical Formula and Derivation

Let $\tilde{X} = X - \frac{1}{N} \mathbf{1}\mathbf{1}^T X$ and $\tilde{Y} = Y - \frac{1}{N} \mathbf{1}\mathbf{1}^T Y$ be the column-centered representation matrices.

The Hilbert-Schmidt Independence Criterion (HSIC) for linear kernels $K = X X^T, L = Y Y^T$ is:
$$
\text{HSIC}(K, L) = \frac{1}{(N - 1)^2} \text{Tr}(K H L H) = \frac{1}{(N - 1)^2} \|\tilde{X}^T \tilde{Y}\|_F^2
$$

Linear CKA is defined as:
$$
\text{CKA}(X, Y) = \frac{\text{HSIC}(X X^T, Y Y^T)}{\sqrt{\text{HSIC}(X X^T, X X^T) \cdot \text{HSIC}(Y Y^T, Y Y^T)}} = \frac{\|\tilde{X}^T \tilde{Y}\|_F^2}{\|\tilde{X}^T \tilde{X}\|_F \|\tilde{Y}^T \tilde{Y}\|_F}
$$

---

## 4. Empirical Usage and Proof of Orthogonal Invariance

```python
import torch
from symmetry_reps.metrics import linear_cka

X1 = torch.randn(100, 16)
X2 = torch.randn(100, 16)

# Generate random orthogonal rotation
Q, _ = torch.linalg.qr(torch.randn(16, 16))
X1_rotated = X1 @ Q

print("CKA(X1, X2):", linear_cka(X1, X2))
print("CKA(X1_rot, X2):", linear_cka(X1_rotated, X2))
# CKA(X1, X2) == CKA(X1_rot, X2) exactly!
```
