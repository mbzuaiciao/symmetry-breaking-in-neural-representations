# Tutorial 28: Symmetry Breaking in World Models: Detection and Failure Modes

## 1. Introducing Perturbations into Dynamical Systems

Consider a rotationally symmetric limit cycle $\dot{r} = r(1 - r^2), \dot{\theta} = 1$. We introduce an asymmetric damping force along the x-axis:
$$
\dot{x} = -y + x(1 - x^2 - y^2) - \delta \cdot x, \quad \dot{y} = x + y(1 - x^2 - y^2)
$$

---

## 2. Model Comparison Under Perturbation

- **Unconstrained Latent Model**: Directly learns the deformed elliptic trajectory. The covariance anisotropy $\mathcal{A}$ grows smoothly with $\delta$.
- **Strictly Equivariant Model**: Fails to represent the asymmetric contraction because its transition operator cannot break $SO(2)$ symmetry, resulting in systematic prediction error along the deformed axis.

---

## 3. Takeaway for World Model Design
Symmetry constraints are invaluable for out-of-distribution generalization in ideal physical systems (e.g. molecular dynamics), but fragile when the real-world environment contains subtle environmental asymmetries (e.g. wind resistance, friction).
