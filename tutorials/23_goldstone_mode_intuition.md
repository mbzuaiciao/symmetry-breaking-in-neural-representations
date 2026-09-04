# Tutorial 23: Goldstone Modes: Physics Analogy and Neural Limits

## 1. Goldstone's Theorem in Field Theory

In theoretical physics, **Goldstone's Theorem** (1961) states:
> *Whenever a continuous global symmetry of a Lagrangian is spontaneously broken, there necessarily exist massless (zero-energy), dispersion-free excitations called Goldstone bosons.*

Geometrically, moving along the bottom of the Mexican hat potential costs zero energy because the potential is rotationally symmetric.

---

## 2. Translating to Neural Networks: What Holds?

1. **Continuous Parameter Symmetries**:
   If the parameterization possesses an exact continuous Lie group symmetry $G$ (such as $SO(2)$ latent rotations or diagonal scale transformations), the loss $\mathcal{L}(\theta)$ is constant along the group orbit.
   - The tangent vectors to the orbit $t_a = \left.\frac{d}{d\alpha} (e^{\alpha T_a} \cdot \theta)\right|_{\alpha=0}$ satisfy:
     $$
     H_{\mathcal{L}} t_a = 0 \implies \text{Exact zero eigenvalue (Goldstone mode in parameter space)}.
     $$

---

## 3. What Does NOT Hold (Scientific Caveats)

- **Discrete Symmetries**: Discrete groups ($\mathbb{Z}_2$, $C_n$, $S_n$) have disconnected group elements. Spontaneous breaking of discrete symmetries produces *isolated domain walls* and distinct basins, **not** continuous Goldstone zero-modes.
- **Representation-Level Symmetries**: When an unconstrained network breaks symmetry in representation space, the loss landscape in parameter space does not necessarily possess a continuous manifold of zero modes unless the parameterization itself is continuously invariant.
