# Tutorial 17: Hard-Coded Equivariance vs. Learned Symmetry

## 1. Two Paradigms for Symmetry in Machine Learning

1. **Hard-Coded Geometric Inductive Bias (Group Equivariant Networks)**:
   The hypothesis space $\mathcal{H}$ is restricted strictly to functions satisfying $f(\rho(g)x) = \rho'(g) f(x)$ by construction (e.g. G-CNNs, Steerable CNNs, Reynolds symmetrization).
   - Symmetry error $E_{\text{sym}} \equiv 0$ identically for all parameter values $\theta$.
2. **Learned Symmetry (Unconstrained Optimization)**:
   The hypothesis space $\mathcal{H}$ contains unconstrained functions (standard MLPs, Transformers). The network must discover symmetric representations via data and gradient descent.

---

## 2. Comparison Dimensions

| Dimension | Hard-Coded Equivariant | Unconstrained Standard |
| :--- | :--- | :--- |
| **Symmetry Error** | 0.0 (by design) | $> 0$ (approximate, dependent on $N$) |
| **Sample Complexity** | $O(1/|G|)$ sample reduction | Standard sample complexity |
| **Flexibility to Asymmetry**| Zero flexibility (underfitting) | High flexibility (adapts to data) |
| **Optimization Landscape**| Constrained parameter subspace | High-dimensional loss landscape |
