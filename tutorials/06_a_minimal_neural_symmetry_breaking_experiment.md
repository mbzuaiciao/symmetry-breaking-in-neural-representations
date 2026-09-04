# Tutorial 06: A Minimal Neural Symmetry-Breaking Experiment

## 1. Experimental Design: The Continuous Symmetric 4-Cluster Task

We construct a minimal 2D binary classification task possessing an exact $\mathbb{Z}_2$ reflection symmetry:

- **Class +1 (Positive)**: Centered at $(+1, +1)$ and $(-1, -1)$
- **Class 0 (Negative)**: Centered at $(+1, -1)$ and $(-1, +1)$

Under diagonal reflection or origin inversion $(x_1, x_2) \mapsto (-x_1, -x_2)$, each class maps onto itself. The dataset distribution $P(x, y)$ is strictly invariant under $\mathbb{Z}_2$.

---

## 2. Hypothesis & Predictions

1. **Individual Runs**: An unconstrained MLP $f(x; \theta)$ trained from standard random initialization will break the $\mathbb{Z}_2$ symmetry by orienting its internal feature space along one preferred diagonal axis.
2. **Ensemble Invariance**: The empirical distribution of order parameters $p(m)$ across $N \ge 25$ independent random seeds will be symmetric around $m = 0$:
   $$
   \mathbb{E}_{\text{seeds}}[m] \approx 0, \quad \text{with bimodal peaks at } +m^* \text{ and } -m^*
   $$

---

## 3. Implementation and Order Parameter Measurement

```python
import torch
from experiments.exp_a_seed_dependence import run_experiment_a

# Run Experiment A
results = run_experiment_a(num_seeds=25, epochs=80, smoke=False)
print("Mean Order Parameter across seeds:", results["mean_order_param"])
print("Standard Deviation:", results["std_order_param"])
```

---

## 4. Empirical Findings

- **Bimodal Histogram**: Over 25 random seeds, runs split approximately 50/50 between positive and negative orientations.
- **Ensemble Mean**: $\langle m \rangle = 0.02 \approx 0.0$.
- **Conclusion**: The learning algorithm spontaneously breaks the task symmetry in each realization, but the ensemble of realizations preserves the global symmetry.
