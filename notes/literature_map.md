# Literature Map: Symmetry, Representation Geometry, and Symmetry Breaking

This document maps foundational and recent literature relevant to symmetry breaking in neural representations.

---

## 1. Geometric Deep Learning & Equivariant Networks
- **Bronstein et al. (2021)**: *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. Comprehensive blueprint of symmetries in machine learning.
- **Cohen & Welling (2016)**: *Group Equivariant Convolutional Networks (G-CNNs)*. Introduced discrete group convolution for rot/reflection groups.
- **Kondor & Trivedi (2018)**: *On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups*. Formulated universal representation-theoretic conditions for equivariance.
- **Weiler & Cesa (2019)**: *General E(2)-Equivariant Steerable CNNs*. Steerable filter representations.

---

## 2. Representation Geometry, Similarity, and Alignment
- **Kornblith et al. (2019)**: *Similarity of Neural Network Representations Revisited*. Introduced Centered Kernel Alignment (CKA) and proved orthogonal invariance.
- **Williams et al. (2021)**: *Generalized Shape Metrics on Neural Representations*. Procrustes distance, geodesics on representation manifolds.
- **Kriegeskorte et al. (2008)**: *Representational Similarity Analysis - Connecting the Branches of Systems Neuroscience*. Foundational RSA framework.
- **Raghu et al. (2017)**: *SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability*. Canonical angles between hidden representations.
- **Roy & Vetterli (2007)**: *The Effective Rank: A Measure of Effective Dimensionality*. Shannon entropy of normalized singular values.

---

## 3. Loss Landscapes, Parameter Symmetries, and Flat Minima
- **Entezari et al. (2021)**: *The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks*. Proved that permutation symmetries explain apparent barriers in loss landscapes.
- **Ainsworth et al. (2022)**: *Git Re-Basin: Merging Models modulo Permutation Symmetries*. Algorithms for aligning hidden unit permutations across independently trained networks.
- **Dinh et al. (2017)**: *Sharp Minima Can Generalize For Deep Nets*. Scaling symmetries in ReLU networks and Hessian reparameterization.
- **Sagun et al. (2017)**: *Empirical Analysis of the Hessian of Over-Parameterized Neural Networks*. Massive zero-eigenvalue bulk in loss Hessians.

---

## 4. Symmetry Breaking and Phase Transitions in Machine Learning
- **Landau & Lifshitz (1980)**: *Statistical Physics*. Classical theory of continuous phase transitions and order parameters.
- **Anderson (1972)**: *More is Different*. Philosophical foundation of broken symmetry and emergent properties.
- **Goldstone (1961)**: *Field Theories with Superconductor Solutions*. Spontaneous breaking of continuous symmetries and zero-energy massless modes (Goldstone bosons).
- **Advani et al. (2020)**: *High-Dimensional Dynamics of Generalization Error in Neural Networks*. Statistical mechanics of learning curves.

---

## 5. Neural Collapse and Geometric Emergence
- **Papyan, Han, & Donoho (2020)**: *Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training*. Discovered NC1 (variability collapse), NC2 (equiangular tight frames), NC3 (classifier duality), and NC4 (nearest class-center decision rule).
- **Mixon et al. (2020)**: *Neural Collapse with Unconstrained Features*. Mathematical analysis of the Simplex ETF as the global minimizer of cross-entropy under cross-feature constraints.
- **Zhu et al. (2021)**: *A Geometric Analysis of Neural Collapse with Cross-Entropy Loss*. Rigorous proof of ETF geometry emergence.
