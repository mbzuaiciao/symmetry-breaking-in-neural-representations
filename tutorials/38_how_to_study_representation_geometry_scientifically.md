# Tutorial 38: How to Study Representation Geometry Scientifically

## 1. Principles of Rigorous Representation Science

1. **Pre-Specify Hypotheses and Metrics**: Define exact mathematical order parameters and falsification conditions *before* plotting exploratory figures.
2. **Control Random Seeds**: Never draw conclusions from a single training run. Run $\ge 20$ independent seeds and report means, confidence intervals, and full empirical distributions.
3. **Use Coordinate-Free Invariant Metrics**: Ensure metrics are invariant to arbitrary hidden unit permutations and orthogonal rotations (e.g. CKA, Procrustes disparity, eigenspectra).
4. **Negative Controls & Perturbation Tests**: Verify that symmetric baselines produce zero order parameters, and sweep perturbation parameters $\epsilon$ to verify continuous response.
