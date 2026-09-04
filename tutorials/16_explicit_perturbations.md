# Tutorial 16: Explicit Perturbations: External Symmetry-Breaking Fields

## 1. Physical Analogy: External Magnetic Fields

In physical ferromagnetism, applying a microscopic magnetic field $h_{\text{ext}} > 0$ breaks the $\pm$ spin degeneracy, tilting the free energy landscape such that the system reliably magnetizes along the direction of the field.

---

## 2. Mathematical Modeling in Neural Networks

We introduce an external symmetry-breaking field $\epsilon \ge 0$ into the training objective or input distribution:
$$
\tilde{x}_i = x_i + \epsilon \cdot v_{\text{bias}}
$$
We sweep $\epsilon \in \{0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ and measure:
- Probability of selecting positive orientation: $P(m > 0)$.
- Susceptibility:
  $$
  \chi = \left. \frac{d \langle m \rangle}{d \epsilon} \right|_{\epsilon=0}
  $$

---

## 3. Empirical Results from Experiment C

- At $\epsilon = 0$: $P(m > 0) \approx 0.50$ (unbiased random selection).
- At $\epsilon = 10^{-3}$: $P(m > 0) \approx 0.85$.
- At $\epsilon \ge 10^{-2}$: $P(m > 0) = 1.00$ (deterministic capture).

This demonstrates high susceptibility: minute data imbalances in real-world training sets deterministically dictate internal representational alignment.
