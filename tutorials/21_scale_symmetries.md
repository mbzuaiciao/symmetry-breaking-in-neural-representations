# Tutorial 21: Scale Symmetries in Positive Homogeneous Networks

## 1. Positive Homogeneity and Scaling Invariance

An activation function $\sigma$ is **positive homogeneous of degree 1** if:
$$
\sigma(c \cdot z) = c \cdot \sigma(z), \quad \forall c > 0
$$
Examples: ReLU ($\max(0, z)$), LeakyReLU.

---

## 2. Continuous Layer-to-Layer Rescaling Group

For an $L$-layer ReLU network, any positive diagonal matrix $D = \text{diag}(s_1, \dots, s_H) \in (\mathbb{R}_{>0})^H$ defines a continuous transformation on weights:
$$
W_l \mapsto D W_l, \quad b_l \mapsto D b_l, \quad W_{l+1} \mapsto W_{l+1} D^{-1}
$$
The output is identically preserved:
$$
f_{\text{rescaled}}(x) = W_{l+1} D^{-1} \sigma(D W_l x + D b_l) = W_{l+1} D^{-1} D \sigma(W_l x + b_l) = f(x)
$$

---

## 3. Implications for Optimization & Loss Landscapes

- **Degenerate Submanifolds**: The loss landscape $\mathcal{L}(\theta)$ contains continuous connected submanifolds of constant loss $(\mathbb{R}_{>0})^{\sum H_l}$.
- **Hessian Eigenvalues & Ill-Conditioning**: Moving along scaling trajectories leaves the loss unchanged ($\nabla \mathcal{L} = 0$, $\nabla^2 \mathcal{L} = 0$). By scaling $D \to \infty$, the Hessian eigenvalues along orthogonal non-scale directions can be made arbitrarily large or small without changing the underlying function!
