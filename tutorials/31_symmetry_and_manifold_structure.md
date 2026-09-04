# Tutorial 31: Symmetry and Manifold Structure: Group Orbits

## 1. Group Actions Generate Homogeneous Manifolds

When a continuous Lie group $G$ acts transitively on a point $x_0 \in \mathbb{R}^d$, the orbit forms a smooth homogeneous manifold:
$$
\mathcal{M} = G \cdot x_0 \cong G / G_{x_0}
$$
where $G_{x_0}$ is the stabilizer subgroup.

Examples:
- $SO(2)$ acting on $(1, 0) \in \mathbb{R}^2$ generates the circle $S^1$.
- $SO(3)$ acting on $(1, 0, 0) \in \mathbb{R}^3$ with stabilizer $SO(2)$ generates the 2-sphere $S^2 \cong SO(3) / SO(2)$.
- 2D translations $\mathbb{R}^2$ modulo periodic lattice $\mathbb{Z}^2$ generate the torus $T^2 \cong \mathbb{R}^2 / \mathbb{Z}^2$.

---

## 2. Symmetry Breaking on Manifolds

When a neural representation breaks continuous $SO(2)$ symmetry, the circular orbit $S^1$ is distorted into an elongated ellipse or broken into discrete clusters (e.g. $C_4$ or $C_8$). 

Visualizing group orbits in representation space allows direct inspection of whether the manifold preserves the Lie group topology.
