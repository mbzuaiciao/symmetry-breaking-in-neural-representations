# Tutorial 03: Groups and Group Actions

## 1. What Problem Are We Trying to Solve?
To mathematically reason about symmetries, we need a rigorous algebraic structure that formalizes transformations, their compositions, and how they manipulate representation vectors.

## 2. Why Is the Naive Approach Insufficient?
Treating transformations as ad-hoc Python functions without group axioms makes it impossible to guarantee closure, invertibility, or coordinate consistency when chaining multiple transformations.

---

## 3. Mathematical Foundations of Group Theory

A **group** $(G, \cdot)$ is a set $G$ equipped with an operation $\cdot : G \times G \to G$ satisfying 4 fundamental axioms:
1. **Closure**: $\forall a, b \in G, \; a \cdot b \in G$.
2. **Associativity**: $\forall a, b, c \in G, \; (a \cdot b) \cdot c = a \cdot (b \cdot c)$.
3. **Identity Element**: $\exists e \in G \text{ such that } \forall a \in G, \; e \cdot a = a \cdot e = a$.
4. **Inverse Element**: $\forall a \in G, \; \exists a^{-1} \in G \text{ such that } a \cdot a^{-1} = a^{-1} \cdot a = e$.

---

## 4. Group Actions and Linear Representations

A **group action** of $G$ on a vector space $V = \mathbb{R}^d$ is a map $\phi: G \times V \to V$, denoted $g \cdot v$, satisfying:
- $\phi(e, v) = v$
- $\phi(g_1, \phi(g_2, v)) = \phi(g_1 \cdot g_2, v)$

When the action is linear, it is a **linear representation**:
$$
\rho: G \to GL(d, \mathbb{R}), \quad \rho(g_1 g_2) = \rho(g_1) \rho(g_2)
$$

### Orbits and Stabilizers
- **Orbit of $x$**: The set of all points reachable under group action:
  $$
  \mathcal{O}_x = G \cdot x = \{ \rho(g) x : g \in G \}
  $$
- **Stabilizer of $x$**: The subgroup leaving $x$ invariant:
  $$
  G_x = \{ g \in G : \rho(g) x = x \}
  $$
- **Orbit-Stabilizer Theorem**: For finite groups, $|G| = |\mathcal{O}_x| \cdot |G_x|$.

---

## 5. Concrete Examples

### Cyclic Group $\mathbb{Z}_n$ / $C_n$
The discrete rotational group on regular $n$-gons. For $C_4$:
$$
\rho(1) = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \quad (\text{rotation by } 90^\circ)
$$

### Permutation Group $S_n$
The group of all $n!$ permutations of $n$ elements. Each $\pi \in S_n$ is represented by an orthogonal permutation matrix $P_{\pi}$.

---

## 6. Python Implementation

```python
import torch
from symmetry_reps.symmetry import CyclicGroup, PermutationGroup, compute_orbit

c4 = CyclicGroup(4)
x = torch.tensor([[1.0, 0.0]])
orbit = compute_orbit(c4, x)
print("Orbit under C_4:", [pt.numpy().tolist() for pt in orbit])
```

---

## 7. Scientific Caveats
- **Non-commutative groups**: While cyclic groups $C_n$ and $\mathbb{Z}_2$ are abelian ($g_1 g_2 = g_2 g_1$), permutation groups $S_n$ ($n \ge 3$) and 3D rotations $SO(3)$ are non-abelian. Order of composition matters!
