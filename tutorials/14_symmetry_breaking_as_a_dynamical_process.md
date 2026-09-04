# Tutorial 14: Symmetry Breaking as a Dynamical Process

## 1. When Does Symmetry Breaking Occur?

Does symmetry breaking happen immediately at epoch 0, or is it a delayed transition?

Empirical tracking across multiple seeds (Experiment B) reveals a characteristic **two-stage dynamic**:
1. **Symmetric Fitting Stage ($0 \le t \le t^*$)**:
   The network fits coarse components of the data distribution while maintaining an unpolarized representation ($m(t) \approx 0$). The loss Hessian at this point possesses near-zero flat modes along symmetry-breaking directions.
2. **Spontaneous Bifurcation ($t > t^*$)**:
   Optimization noise or minute gradient imbalances amplify exponentially along the unstable mode, causing $m(t)$ to rapidly branch toward $+m^*$ or $-m^*$.

---

## 2. Bifurcation Diagram and Mathematical Modeling

We can model the effective gradient flow of the order parameter near the critical point as a pitchfork bifurcation:
$$
\frac{dm}{dt} = \alpha(t) m - \beta m^3 + \xi(t)
$$
where $\alpha(t)$ is the curvature along the symmetry-breaking direction, $\beta > 0$ provides saturation, and $\xi(t)$ is SGD gradient noise.
- When $\alpha(t) < 0$: $m = 0$ is a stable fixed point (symmetric phase).
- When $\alpha(t) > 0$: $m = 0$ becomes unstable, and two stable fixed points $m^* = \pm \sqrt{\alpha / \beta}$ emerge.
