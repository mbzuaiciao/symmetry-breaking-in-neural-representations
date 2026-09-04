# Tutorial 01: Representations as Geometry

## 1. What Problem Are We Trying to Solve?
To understand how a neural network processes information, inspecting individual neuron weights $W_{ij}$ is frequently uninformative. We need a coordinate-free, holistic mathematical framework to quantify the internal geometric relationships formed by hidden layer activations.

## 2. Why Is the Naive Approach Insufficient?
Comparing raw activation vectors directly across models fails because:
- Hidden units can be permuted without altering network function ($\Pi \in S_H$).
- Networks may learn identical functional representations rotated by an arbitrary orthogonal transformation $R \in O(d)$.

## 3. Mathematical Objects & Definitions

### Representation Matrix
Let $\mathcal{D} = \{x_1, \dots, x_N\}$ be a dataset of $N$ inputs. The representation matrix at hidden layer $l$ is:
$$
H \in \mathbb{R}^{N \times d} = \begin{bmatrix} h(x_1)^T \\ \vdots \\ h(x_N)^T \end{bmatrix}
$$

### Centered Covariance Matrix
Let $\tilde{H} = H - \frac{1}{N} \mathbf{1}\mathbf{1}^T H$ denote the mean-centered activation matrix. The sample covariance is:
$$
\Sigma = \frac{1}{N - 1} \tilde{H}^T \tilde{H} \in \mathbb{R}^{d \times d}
$$

### Gram (Kernel) Matrix
The sample Gram matrix measures pairwise inner products between sample representations:
$$
K = H H^T \in \mathbb{R}^{N \times N}, \quad K_{ij} = \langle h(x_i), h(x_j) \rangle
$$

### Principal Component Analysis (PCA) & Eigenspectra
Eigendecomposition of the covariance matrix:
$$
\Sigma = V \Lambda V^T = \sum_{k=1}^d \lambda_k v_k v_k^T
$$
where $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$ are the eigenvalues, and $v_k$ are orthonormal principal directions.

---

## 4. Geometric Meaning of the Equations
- **Covariance $\Sigma$** characterizes the spread, elongation, and principal axes of the representation point cloud in feature space.
- **Gram Matrix $K$** captures all pairwise relational geometry (distances and angles) between datapoints, completely independent of the feature basis.
- **Eigenvalue Spectrum $\Lambda$** measures the variance captured along orthogonal directions. A steep spectral decay indicates that the representation lives on a low-dimensional subspace.

---

## 5. What Does It Mean in a Neural Network?
During learning, optimization shapes the eigenspectrum of $H$:
- Early training: Representations are often diffuse and isotropic ($\lambda_1 \approx \dots \approx \lambda_d$).
- Late training: Task-relevant directions expand while redundant directions contract, compressing representations into low-dimensional task manifolds.

---

## 6. How to Test It Empirically

```python
import torch
from symmetry_reps.geometry import compute_covariance, compute_pca_spectrum, effective_rank

# Sample activations (N=200, D=64)
H = torch.randn(200, 64)
cov = compute_covariance(H)
eigenvalues, explained_var, components = compute_pca_spectrum(H)
erank = effective_rank(H)

print(f"Top 3 Eigenvalues: {eigenvalues[:3]}")
print(f"Effective Rank: {erank:.2f}")
```

---

## 7. Failure Modes & Scientific Caveats
- **Coordinate Bias**: Looking at single-neuron tuning curves can fool researchers into believing a neuron has a distinct semantic role, when it is merely a projection of a distributed subspace onto an arbitrary basis.
- **Sample Size Artifacts**: If $N < d$, the sample covariance has rank at most $N-1$, producing apparent spectral collapse that is purely a finite-sample artifact.
