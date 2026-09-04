# Tutorial 02: Symmetry from First Principles

## 1. What Problem Are We Trying to Solve?
In machine learning and physics, the world exhibits regularities under transformations: an object rotated in space remains the same object, and physical laws do not depend on the laboratory's spatial coordinates. We seek a formal mathematical language to define when a system, function, or neural network respects a transformation.

## 2. Why Is the Naive Approach Insufficient?
Informal notions like "invariance" and "equivariance" are often conflated. For example, a classifier predicting object category should be *invariant* to translation, whereas a segmentation or dynamics model predicting future velocity vectors must be *equivariant* (the predicted vectors must rotate when the scene rotates). Conflating these leads to incorrect architectural choices.

---

## 3. Mathematical Definitions: Invariance and Equivariance

Let $\mathcal{X}$ be the input space, $\mathcal{Y}$ the output space, and $G$ a group of transformations.

Let $\rho_{\mathcal{X}}(g)$ be the transformation (group action) on $\mathcal{X}$, and $\rho_{\mathcal{Y}}(g)$ the action on $\mathcal{Y}$.

### Invariance
A function $f: \mathcal{X} \to \mathcal{Y}$ is **invariant** under $G$ if transforming the input leaves the output completely unchanged:
$$
f(\rho_{\mathcal{X}}(g) x) = f(x), \quad \forall g \in G, \; \forall x \in \mathcal{X}
$$

### Equivariance
A function $f: \mathcal{X} \to \mathcal{Y}$ is **equivariant** under $G$ if transforming the input transforms the output by the corresponding action:
$$
f(\rho_{\mathcal{X}}(g) x) = \rho_{\mathcal{Y}}(g) f(x), \quad \forall g \in G, \; \forall x \in \mathcal{X}
$$

Note that invariance is a special case of equivariance where $\rho_{\mathcal{Y}}(g) = \text{Id}$ (the trivial representation).

---

## 4. Concrete Transformation Examples

### Reflection ($\mathbb{Z}_2$)
- $\rho(-1) x = -x$ (sign inversion) or reflection across an axis $\rho(-1) (x_1, x_2) = (-x_1, x_2)$.
- Invariance: $f(-x) = f(x)$ (e.g. Euclidean norm $\|x\|$).
- Equivariance: $f(-x) = -f(x)$ (e.g. odd polynomials, vector field flows).

### 2D Rotation ($SO(2)$ / $C_n$)
- $\rho(\theta) \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$
- Invariant function: $f(x) = x_1^2 + x_2^2 = \|x\|^2$.

### Permutations ($S_n$)
- $\rho(\pi) (x_1, \dots, x_n) = (x_{\pi(1)}, \dots, x_{\pi(n)})$.
- Invariant function: $f(x) = \sum_{i=1}^n x_i$ or $\max(x)$.
- Equivariant function: Transformer self-attention without positional encodings ($f(\pi x) = \pi f(x)$).

---

## 5. What Does It Mean in a Neural Network?
A neural network is a composition of layers: $f = f_L \circ f_{L-1} \circ \dots \circ f_1$.
- If each intermediate layer $f_l$ is equivariant ($\rho_{l}(g) f_l(h) = f_l(\rho_{l-1}(g) h)$), the entire network is equivariant.
- Placing a pooling/invariant layer at the end yields an end-to-end invariant network.

---

## 6. How to Test It Empirically

```python
import torch
from symmetry_reps.symmetry import CyclicGroup, check_equivariance, check_invariance

c4 = CyclicGroup(4)
f_rot = lambda x: 2.0 * x  # Linear scaling is equivariant to rotations

x_sample = torch.randn(10, 2)
mean_err, is_eq = check_equivariance(f_rot, c4, x_sample)
print(f"Equivariance Error: {mean_err:.2e}, Equivariant: {is_eq}")
```

---

## 7. Failure Modes & Scientific Caveats
- **Testing on symmetric inputs**: If $x$ is invariant under $g$ ($\rho(g)x = x$), then $f(\rho(g)x) = f(x)$ holds trivially even if $f$ is not invariant. Always test invariance on generic asymmetric inputs!
- **Floating-point tolerance**: Due to float32 rounding in trigonometric rotations, exact zero is rarely attained; use numerical tolerance $\epsilon \sim 10^{-5}$.
