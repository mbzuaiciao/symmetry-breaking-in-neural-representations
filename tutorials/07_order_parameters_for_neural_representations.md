# Tutorial 07: Order Parameters for Neural Representations

## 1. What Problem Are We Trying to Solve?
To scientifically quantify symmetry breaking, we cannot rely on qualitative visual inspection of scatter plots. We require formal, scalar mathematical **order parameters** that can be evaluated on any representation matrix $H \in \mathbb{R}^{N \times d}$.

---

## 2. Four Concrete Order Parameters

### 1. Mean Signed Activation Projection
For binary classification with labels $y_i \in \{-1, +1\}$ and symmetry-breaking axis $v \in \mathbb{R}^d$:
$$
m_{\text{proj}} = \frac{1}{N} \sum_{i=1}^N y_i \langle h(x_i), v \rangle
$$
- $m = 0$: Unpolarized / symmetric representation.
- $m \ne 0$: Polarized representation aligned along axis $v$.

### 2. Representation Imbalance / Cluster Center Distance
Let $\mu_+ = \frac{1}{N_+} \sum_{y_i = +1} h(x_i)$ and $\mu_- = \frac{1}{N_-} \sum_{y_i = -1} h(x_i)$.
$$
\Delta \mu = \|\mu_+\|_2 - \|\mu_-\|_2
$$

### 3. Covariance Eigenvalue Splitting
In a fully symmetric state, orthogonal symmetric modes share degenerate eigenvalues ($\lambda_1 = \lambda_2$). Symmetry breaking lifts this degeneracy:
$$
\Delta \lambda = \lambda_1 - \lambda_2 > 0
$$

### 4. Covariance Anisotropy Index
$$
\mathcal{A} = \frac{\lambda_{\max} - \bar{\lambda}}{\sum_{i=1}^d \lambda_i} \in [0, 1]
$$
Measures concentration of variance along the dominant symmetry-breaking mode.

---

## 3. What Makes a Good Order Parameter?
1. **Invariance under Irrelevant Gauge Shifts**: An order parameter should be invariant to hidden unit permutations and global translations.
2. **Monotonicity with Symmetry Violation**: $m \to 0$ continuously as the representation becomes perfectly symmetric.
3. **Statistical Power**: High signal-to-noise ratio across finite datasets.
