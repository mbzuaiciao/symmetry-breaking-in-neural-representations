# Open Questions in Neural Representation Symmetries

This document tracks unresolved theoretical and empirical questions that remain open for future investigation.

---

## 1. The Goldstone Mode in Deep Learning: Analogy or Theorem?
- *Open Question*: When a continuous symmetry (such as $SO(2)$ latent rotation or parameter scale symmetry) is spontaneously broken by training, does the loss Hessian strictly possess exact zero eigenvalues corresponding to group generator tangents $\partial_\theta \mathcal{L} = 0$?
- *Current Understanding*: For parameter symmetries (permutations, continuous scaling), zero directions in parameter space are exact. For representation-level symmetries, the Hessian w.r.t *parameters* does not necessarily have exact zero modes unless the parameterization itself is continuously invariant.

---

## 2. Symmetry Breaking as a Function of Optimization Algorithm
- *Open Question*: Does Stochastic Gradient Descent (SGD) with large learning rates act as a thermal bath that preserves symmetry longer than deterministic Full-Batch Gradient Descent (GD)?
- *Hypothesis*: Mini-batch noise acts as an effective temperature $T_{\text{eff}} \propto \frac{\eta}{B}$. Above a critical temperature $T^*$, thermal fluctuations should restore symmetry ($\langle m \rangle = 0$ with low variance), whereas cooling down ($T < T^*$) triggers spontaneous symmetry breaking.

---

## 3. Symmetry in Superposition and Polysemanticity
- *Open Question*: When a model represents $M > D$ features in a $D$-dimensional space via superposition (e.g. Anthropic's toy models), does the geometry of the optimal frame (e.g., antipodal pairs, pentagons, icosahedra) reflect spontaneous breaking of continuous $O(D)$ symmetry to a discrete polytope subgroup?

---

## 4. Inductive Bias vs Generalization on Partially Symmetric Data
- *Open Question*: What is the exact mathematical trade-off between bias and variance when enforcing a symmetry group $G$ on a task whose true data distribution only respects a subgroup $H \subset G$?
