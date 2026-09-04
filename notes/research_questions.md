# Research Questions: Symmetry Breaking in Neural Representations

## Central Research Question
> **When does a neural network preserve the symmetry of a task, and when does optimization select an asymmetric internal representation?**

---

## Tier 1: Core Mechanics of Spontaneous Symmetry Breaking

1. **Ensemble Symmetry vs. Instance Polarization (Hypothesis 1)**
   - *Question*: Given an exact global discrete symmetry $G$ in the data distribution and loss objective, does an individual network trained from random initialization break $G$ by choosing a specific orientation in representation space, while the ensemble of models across seeds remains $G$-symmetric?
   - *Order Parameter*: $m = \frac{1}{N} \sum_{i=1}^N y_i \langle h(x_i), v_{\text{sym}} \rangle$.
   - *Falsification Condition*: If $\forall \text{seed}, m \approx 0$ (networks learn symmetric representations directly) or if all seeds consistently collapse to a single biased orientation regardless of initialization.

2. **Dynamical Trajectory of Symmetry Breaking (Hypothesis 2)**
   - *Question*: Does symmetry breaking occur early during the initial spectral fitting phase, or does it emerge via a sharp bifurcation as training error approaches zero?
   - *Metrics*: $\frac{dm}{dt}$, effective rank $\text{erank}(t)$, and leading covariance eigenvalue splitting $\Delta \lambda_{12}(t)$.

3. **Susceptibility to External Fields (Hypothesis 3)**
   - *Question*: How sensitive is representation orientation to a tiny symmetry-breaking perturbation $\epsilon$? Is there a critical susceptibility $\chi = \left.\frac{\partial \langle m \rangle}{\partial \epsilon}\right|_{\epsilon=0}$?

---

## Tier 2: Representation Geometry and Latent Equivalence

4. **Isometric Equivalence under Procrustes Alignment**
   - *Question*: When two networks select different symmetry orientations, are their internal geometries fundamentally distinct, or are they related by an exact orthogonal transformation $R \in O(d)$?
   - *Metric*: Normalized Procrustes disparity $\mathcal{D}(H_1, H_2) = \min_{R} \frac{\|H_1 R - H_2\|_F}{\|H_2\|_F}$ vs. raw CKA.

5. **Width and Dimensionality Collapse**
   - *Question*: Does increasing width $W \to \infty$ suppress symmetry-breaking variance (as in the Neural Tangent Kernel infinite-width limit) or enhance representational diversity?

6. **Regularization and Implicit Bias**
   - *Question*: Does weight decay ($L_2$ regularization) drive representations toward maximal symmetry (higher entropy/isotropic) or accelerate rank collapse?

---

## Tier 3: Connections to Downstream Phenomena

7. **World Models and Latent Dynamics**
   - *Question*: In continuous dynamical systems with rotational symmetry, how does an unconstrained latent transition model break symmetry compared to an $SO(2)$-steerable architecture?

8. **Neural Collapse as Equiangular Symmetry Emergence**
   - *Question*: Can the terminal phase of training (Neural Collapse into Simplex Equiangular Tight Frames) be framed as an emergent maximal discrete symmetry ($S_C$) breaking problem in feature space?
