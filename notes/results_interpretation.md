# Interpretation of Experimental Results

A synthesis and scientific discussion of empirical observations from Experiments A through H.

---

## 1. Experiment A: Seed Dependence & Ensemble Symmetry
- **Empirical Finding**: Training 25 MLPs on a $\mathbb{Z}_2$-symmetric continuous task yields individual models with polarized order parameters $|m| \approx 3.5 - 5.0$, while the ensemble distribution $\langle m \rangle \approx 0.0$ and exhibits a clear bimodal distribution centered symmetrically around zero.
- **Scientific Interpretation**: This confirms the core hypothesis of spontaneous symmetry breaking in representation learning. Random initialization selects a basin of attraction corresponding to one orientation. The system possesses a degenerate manifold of optimal solutions.

---

## 2. Experiment B: Training Dynamics and Bifurcation Timing
- **Empirical Finding**: During the initial epochs ($t < 10$), the order parameter remains close to zero ($m \approx 0$), while the loss decreases rapidly. Symmetry breaking occurs only after initial coarse fitting, marked by a rapid divergent bifurcation in $m(t)$ toward positive or negative saturation.
- **Scientific Interpretation**: Symmetry breaking is not instantaneous at step 0; it requires the optimization trajectory to exit the isotropic saddle region near the origin.

---

## 3. Experiment C: External Perturbation Response (Susceptibility)
- **Empirical Finding**: Even a minute perturbation $\epsilon = 10^{-3}$ tilts the orientation selection probability from $50\%$ to over $85\%$, and $\epsilon = 10^{-2}$ yields deterministic $100\%$ selection of the biased orientation.
- **Scientific Interpretation**: Representation selection exhibits high susceptibility near the symmetric critical point, explaining why subtle dataset imbalances strongly bias internal representations.

---

## 4. Experiment D: Representation Alignment & Procrustes Isometry
- **Empirical Finding**: Independently trained models with opposite order parameters ($m_1 > 0, m_2 < 0$) have high CKA similarity ($\text{CKA} > 0.95$) and achieve near-zero disparity ($< 0.08$) after Orthogonal Procrustes rotation.
- **Scientific Interpretation**: Apparent representational differences across runs are predominantly coordinate artifacts (gauge degrees of freedom). The underlying geometric structure is invariant and shared.

---

## 5. Experiment E: Architecture Comparison & Inductive Bias
- **Empirical Finding**: On small sample sizes ($N < 50$), the $\mathbb{Z}_2$-invariant architecture significantly outperforms the unconstrained MLP. As sample size grows ($N > 200$), the unconstrained MLP closes the performance gap.
- **Scientific Interpretation**: Hard-coded symmetry is an effective sample-complexity regularizer for small data regimes, but unconstrained models can learn equivalent representations given sufficient data.

---

## 6. Experiment F: Data Asymmetry & Phase Transitions
- **Empirical Finding**: As the ground truth is shifted by $\delta \in [0, 1]$, the order parameter responds smoothly for small $\delta$ before saturating at large $\delta$.
- **Scientific Interpretation**: The transition is a continuous cross-over rather than a discontinuous first-order jump in this continuous Gaussian cluster regime.

---

## 7. Experiment G: Network Width Scaling
- **Empirical Finding**: Increasing width $W$ from 4 to 128 increases representation polarization magnitude while effective rank stabilizes at a low intrinsic dimension.
- **Scientific Interpretation**: Wider networks do not disperse activations uniformly throughout the ambient space; rather, overparameterization facilitates sharp compression onto low-dimensional linear manifolds.

---

## 8. Experiment H: Regularization and Anisotropy
- **Empirical Finding**: Increasing weight decay parameter $\lambda$ reduces covariance anisotropy and compresses feature norms, enforcing a more isotropic latent geometry.
- **Scientific Interpretation**: Weight decay acts as a geometric regularizer that penalizes excessive polarization along single principal axes.
