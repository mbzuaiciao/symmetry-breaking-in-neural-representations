# Tutorial 37: Superposition, Polysemanticity, and Discrete Symmetries

## 1. What Is Superposition?

**Superposition** (Elhage et al., 2022; Anthropic) describes the phenomenon where a neural network represents $M > d$ features in a $d$-dimensional activation space by exploiting feature sparsity:
$$
h = \sum_{i=1}^M c_i f_i \in \mathbb{R}^d
$$
where $f_i$ are quasi-orthogonal unit feature vectors.

---

## 2. Geometric Configurations of Superposition

In small dimensions ($d = 2, 3$):
- $M = 2$ features in $d = 2$: Orthogonal frame ($\mathbb{Z}_2^2$ symmetry).
- $M = 3$ features in $d = 2$: Equilateral triangle / antipodal pairs ($D_3$ dihedral symmetry).
- $M = 5$ features in $d = 2$: Regular pentagon ($D_5$ dihedral symmetry).
- $M = 20$ features in $d = 3$: Regular icosahedron ($I_h$ symmetry).

The emergence of regular polytopes in superposition is a direct consequence of energy minimization (Thomson problem on spheres), where continuous rotation symmetry $SO(d)$ spontaneously breaks to a discrete polytope subgroup!
