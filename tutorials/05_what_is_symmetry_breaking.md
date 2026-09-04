# Tutorial 05: What Is Symmetry Breaking?

## 1. What Problem Are We Trying to Solve?
When an optimization problem has multiple equivalent global minimizers related by a symmetry transformation $g \in G$, an algorithm initialized randomly will typically select a single minimizer. We need to formalize what "symmetry breaking" means in the context of gradient descent and neural network representations.

---

## 2. Explicit vs. Spontaneous Symmetry Breaking

### Explicit Symmetry Breaking
- Occurs when the governing equations, dataset, or loss objective contain an explicit term that violates the symmetry.
- *Physics Example*: Applying an external magnetic field $B$ to a ferromagnet breaks spin-flip symmetry.
- *Neural Net Example*: Adding an asymmetric inductive bias or training with an imbalanced dataset ($P(x) \ne P(\rho(g)x)$).

### Spontaneous Symmetry Breaking (SSB)
- Occurs when the objective function $\mathcal{L}(\theta)$ is *strictly symmetric* under $G$, but the selected solution $\theta^*$ is *not* invariant:
  $$
  \mathcal{L}(g \cdot \theta) = \mathcal{L}(\theta), \quad \text{yet} \quad g \cdot \theta^* \ne \theta^*
  $$
- The ground-state manifold forms a degenerate orbit $\mathcal{M} = \{ g \cdot \theta^* : g \in G \}$.

---

## 3. Order Parameters and Phase Transitions

In Landau theory, an **order parameter** $m$ is a macroscopic observable that:
- Equals zero ($m = 0$) in the symmetric / disordered phase.
- Assumes a non-zero value ($m \ne 0$) in the broken / ordered phase.

In neural networks:
- At initialization with mean-zero symmetric weights, representations are unpolarized ($m \approx 0$).
- As optimization proceeds past a critical threshold, the network chooses a specific branch ($m > 0$ or $m < 0$), breaking the symmetry.

---

## 4. Analogy vs Mathematical Fact
> [!WARNING]
> While statistical physics provides powerful intuition (e.g. Mexican hat potential, Ising ferromagnetism), deep neural networks operate out of equilibrium with finite sample sizes and non-conservative gradient flow. We treat physical concepts as rigorous analogies and define explicit mathematical order parameters tailored to activation geometry.
