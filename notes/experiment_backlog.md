# Experiment Backlog

A roadmap of completed, in-progress, and prospective experiments on representation geometry and symmetry breaking.

---

## Core Experimental Suite (Completed)

- [x] **Exp A: Seed Dependence & Ensemble Symmetry**
  - Verify if $\langle m \rangle_{\text{seeds}} = 0$ while individual runs have $|m| > 0$.
- [x] **Exp B: Training Trajectories & Transition Timing**
  - Track $m(t)$, $\text{erank}(t)$, loss, and CKA across training epochs.
- [x] **Exp C: External Symmetry-Breaking Field**
  - Measure response curve $P(m > 0)$ as a function of perturbation $\epsilon \in [0, 10^{-1}]$.
- [x] **Exp D: Representation Alignment & Procrustes Isometry**
  - Prove that distinct seed representations are isometric under $O(d)$ Procrustes rotation.
- [x] **Exp E: Architecture Comparison (Hard-Coded vs Learned)**
  - Compare sample efficiency and symmetry error across sample sizes.
- [x] **Exp F: Ground-Truth Data Asymmetry Sweep**
  - Characterize representation adaptation when data symmetry is broken by $\delta \in [0, 1]$.
- [x] **Exp G: Network Width Sweep**
  - Track order parameter variance and effective rank across $W \in [4, 128]$.
- [x] **Exp H: Regularization and Weight Decay**
  - Assess how weight decay parameter $\lambda$ impacts anisotropy and symmetry breaking.

---

## Extended Experiment Backlog

1. **Continuous Rotation Manifolds ($SO(2)$ / $S^1$)**
   - *Description*: Replace discrete $\mathbb{Z}_2$ clusters with a continuous circle $S^1$. Measure phase-angle order parameter $\theta_{\text{orient}} = \text{atan2}(\bar{y}, \bar{x})$.
   - *Target Metric*: Circular variance of learned phase angles across seeds.

2. **Hessian Mode Alignment at the Symmetry-Breaking Point**
   - *Description*: Compute the Hessian eigenspectrum precisely at the epoch $t^*$ where $m(t)$ bifurcates away from 0.
   - *Target Metric*: Overlap between the leading negative/near-zero Hessian eigenvector and the symmetry-breaking parameter gradient $\nabla_{\theta} m$.

3. **Multi-Class Neural Collapse & Equiangular Tight Frame Dynamics**
   - *Description*: Extend to $C = 10$ classes on synthetic Gaussian simplexes and MNIST.
   - *Target Metric*: Track NC1, NC2, NC3 throughout extended training beyond zero training error (Terminal Phase of Training).

4. **Superposition and Orthogonal Disentanglement**
   - *Description*: Study toy models of superposition (Anthropic style) under group symmetries.
   - *Target Metric*: Feature activation sparsity and Gram matrix off-diagonal interference.

5. **SGD Batch Noise vs. Full-Batch Gradient Descent**
   - *Description*: Compare symmetry-breaking trajectories under deterministic gradient descent (GD) vs. mini-batch stochastic gradient descent (SGD) with varying batch sizes $B \in \{4, 16, 64, \infty\}$.
   - *Target Metric*: Critical epoch of bifurcation $t^*(B)$.
