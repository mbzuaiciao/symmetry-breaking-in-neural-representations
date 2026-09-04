# Tutorial 15: Initialization and Symmetry

## 1. Comparing Initialization Schemes

How does initialization constrain the symmetry of the learning trajectory?

1. **Exactly Symmetric Initialization**:
   If weights are initialized such that $W_1 = -W_1$ (or identical paired rows $W_{1, 2k} = -W_{1, 2k+1}$), and exact full-batch gradient descent is used, the gradients remain strictly symmetric:
   $$
   \nabla_{\theta} \mathcal{L} \text{ respects } G \implies \theta(t) \text{ remains trapped in the symmetric subspace forever!}
   $$
2. **Standard Random Initialization (Kaiming / Xavier)**:
   Random normal entries break the exact coordinate symmetry microscopically, acting as an initial seed perturbation that selects a basin.
3. **Deliberately Asymmetric Initialization**:
   Explicitly adding a positive bias to one neuron family deterministically drives the network into that orientation ($P(m > 0) = 1.0$).

---

## 2. Numerical Trapping vs Finite-Precision Perturbation

In exact arithmetic, symmetric initializations cannot break symmetry without external noise. In floating-point arithmetic (IEEE 754 float32), round-off errors at $\sim 10^{-7}$ can eventually act as tiny perturbations, but the escape time $\tau_{\text{escape}} \propto \frac{1}{\lambda_{\max}} \ln(1 / \epsilon_{\text{float}})$ can be very long.
