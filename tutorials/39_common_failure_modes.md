# Tutorial 39: Common Failure Modes in Representation Geometry

## 1. Top Seven Pitfalls to Avoid

1. **Treating 2D PCA/t-SNE/UMAP as Proof**: 2D projections discard $>90\%$ of high-dimensional variance; apparent visual clusters can be projection artifacts.
2. **Confusing Parameter Symmetry with Representation Symmetry**: Functional equivalence of permuted weights does not imply the internal activation manifold is symmetric.
3. **Ignoring Seed Dependence**: A single run with $m = +3.2$ might lead an investigator to claim "the model always chooses orientation A," missing the fact that orientation B is chosen 50% of the time.
4. **Using CKA without Understanding Invariance Constraints**: Linear CKA is invariant to rotations and isotropic scaling, but sensitive to non-uniform scaling; choose metrics aligned with your exact hypothesis.
5. **Inferring Phase Transitions from Noisy Curves**: Claiming a sharp thermodynamic transition from a single noisy learning curve without finite-size scaling.
6. **Overusing Physics Terminology**: Throwing terms like "Goldstone bosons," "gauge fields," and "Higgs mechanism" around without defining the explicit differential equations and state spaces.
7. **Neglecting Negative Controls**: Failing to check whether an unconstrained random network at initialization shows the same effect.
