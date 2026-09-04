# NeurReps Workshop Paper Proposals

This document outlines 4 concrete, scientifically rigorous workshop paper proposals derived directly from the experiments and theoretical framework in this repository.

---

## Proposal 1: Spontaneous Symmetry Breaking in Representation Space: Order Parameters, Dynamical Transitions, and Procrustes Invariance
- **Track/Venue**: NeurIPS Workshop on Symmetry and Representation (NeurReps) / UniReps
- **Target Format**: 4–8 pages + Appendix

### Abstract
While modern geometric deep learning emphasizes hard-coded equivariance, standard unconstrained neural networks frequently solve symmetric tasks by discovering asymmetric internal representations. In this work, we formalize spontaneous symmetry breaking in representation learning by introducing representation-level order parameters. Through controlled experiments on discrete $\mathbb{Z}_2$ and cyclic $C_n$ symmetric manifolds, we show: (1) individual training trajectories undergo a symmetry-breaking bifurcation at the onset of loss convergence; (2) while individual models break symmetry, the ensemble distribution over random seeds is strictly symmetric; and (3) independently trained asymmetric representations are pairwise isometric under Orthogonal Procrustes alignment. Our results establish that symmetry is preserved at the level of the hypothesis ensemble and geometry, even when violated coordinate-wise by individual models.

### Evaluation Criteria
| Criterion | Rating | Justification |
| :--- | :---: | :--- |
| **Novelty** | **High** | Translates statistical physics concepts into precise representation geometry without vague hand-waving. |
| **Feasibility** | **Very High** | Fully reproducible on laptop CPU/Apple Silicon in under 5 minutes. |
| **Compute Req.** | **Low** | Synthetic 2D/4D datasets and shallow MLPs. |
| **Scientific Clarity**| **High** | Direct mathematical falsification criteria via order parameters and Procrustes disparity. |
| **Risk** | **Low** | Strong empirical effects with minimal noise. |

---

## Proposal 2: When Does Equivariance Hurt? Phase Transitions in Asymmetric Data Regimes
- **Track/Venue**: NeurReps / ICLR GDL Workshop

### Abstract
Equivariance is widely treated as an unalloyed inductive benefit. However, real-world data often exhibits *approximate* rather than exact symmetries. We investigate the trade-off between hard-coded equivariant architectures and unconstrained MLPs as a function of data asymmetry $\delta$. We identify a critical threshold $\delta^*$ below which unconstrained networks retain the approximate symmetry, but above which they rapidly adapt to the asymmetry, outperforming strictly equivariant models which suffer structural underfitting. We characterize this transition via effective rank and CKA geometry.

### Evaluation Criteria
| Criterion | Rating | Justification |
| :--- | :---: | :--- |
| **Novelty** | **Medium-High**| Critical counter-narrative to naive equivariance enforcement. |
| **Feasibility** | **Very High** | Sweep over $\delta \in [0, 1]$ and sample sizes. |
| **Compute Req.** | **Low** | Runs in minutes. |
| **Scientific Clarity**| **Very High** | Clear Pareto frontier curves of test error vs data asymmetry. |
| **Risk** | **Low** | Clear positive/negative contrast. |

---

## Proposal 3: Neural Collapse as Spontaneous Permutation Symmetry Breaking
- **Track/Venue**: NeurReps / Workshop on Emergent Geometries in Deep Learning

### Abstract
Neural Collapse (NC) describes the geometric convergence of penultimate activations into a Simplex Equiangular Tight Frame (ETF) in the terminal phase of training. We re-examine NC through the lens of representation symmetry. The unconstrained loss landscape exhibits continuous orthogonal $O(d)$ and discrete permutation $S_C$ symmetries. We analyze how optimization selects an ETF orientation, tracking within-class collapse (NC1), angle equiangularity (NC2), and weight-activation duality (NC3) as a symmetry-breaking progression.

### Evaluation Criteria
| Criterion | Rating | Justification |
| :--- | :---: | :--- |
| **Novelty** | **Medium-High**| Connects Papyan et al.'s NC framework to group representation theory. |
| **Feasibility** | **High** | Multi-class synthetic and small image classification tasks. |
| **Compute Req.** | **Medium-Low** | Small MLP and ResNet architectures. |
| **Scientific Clarity**| **High** | Closed-form formulas for Simplex ETF angles and Hessian eigenspectra. |
| **Risk** | **Medium** | Careful distinction between mathematical facts and physical analogies. |

---

## Proposal 4: Representation Anisotropy and Implicit Regularization in Latent Dynamics
- **Track/Venue**: NeurReps

### Abstract
Latent world models trained on Hamiltonian or rotational dynamical systems frequently develop anisotropic representations. We show that $L_2$ regularization (weight decay) acts as an isotropic pressure that controls the effective rank and prevents degenerate representation collapse, while unregularized networks collapse onto low-dimensional manifolds that align with the dominant dynamical modes.

### Evaluation Criteria
| Criterion | Rating | Justification |
| :--- | :---: | :--- |
| **Novelty** | **Medium** | Extends representation geometry to latent transition dynamics. |
| **Feasibility** | **High** | 2D dynamical systems (pendulum / harmonic oscillator). |
| **Compute Req.** | **Low** | Fast training. |
| **Scientific Clarity**| **High** | Direct trajectory MSE and participation ratio metrics. |
| **Risk** | **Low** | Clear empirical demonstration. |
