# Tutorial 19: When Symmetry Constraints Hurt: Approximate Symmetries and Inductive Bias

## 1. Symmetry as an Inductive Bias, Not an Absolute Good

> **Core Principle:**
> *Symmetry is a hypothesis about the world. When the hypothesis is wrong, enforcing it causes structural underfitting and permanent information loss.*

---

## 2. Approximate Symmetries in the Real World

In reality, few natural systems possess *exact* mathematical symmetries:
- Natural images have vertical gravity gradients (sky is up, ground is down); vertical flip equivariance is harmful.
- Medical MRI scans possess anatomical left-right asymmetry (heart is on the left); enforcing bilateral reflection destroys diagnostic signals.

---

## 3. Mathematical Demonstration of Information Loss

Let the true label be $y(x) = f_{\text{sym}}(x) + \delta \cdot g_{\text{asym}}(x)$.
- A hard-coded invariant model projects all inputs onto the group orbit: $f_{\text{inv}}(x) = \frac{1}{|G|} \sum_{g \in G} f(gx)$.
- The asymmetric signal $\delta \cdot g_{\text{asym}}(x)$ is averaged to zero:
  $$
  f_{\text{inv}}(x) = f_{\text{sym}}(x) + \delta \cdot \underbrace{\left(\frac{1}{|G|} \sum g_{\text{asym}}(gx)\right)}_{= 0}
  $$
- The invariant model has an irreducible approximation error $O(\delta^2)$ that cannot be eliminated by any amount of training data!
