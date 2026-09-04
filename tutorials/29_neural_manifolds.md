# Tutorial 29: Neural Manifolds and the Manifold Hypothesis

## 1. The Manifold Hypothesis

The **Manifold Hypothesis** posits that real-world high-dimensional data (e.g. natural images in $\mathbb{R}^{3 \times 256 \times 256}$) concentrates near smooth, low-dimensional submanifolds $\mathcal{M} \subset \mathbb{R}^D$ of dimension $d \ll D$.

---

## 2. Canonical Manifold Topologies

- **1D Line ($\mathbb{R}^1$)**: Continuous scalar intensity or age.
- **Circle ($S^1$)**: Periodic variables (heading angle, phase of periodic oscillator).
- **2-Sphere ($S^2$)**: 3D orientation without roll.
- **Torus ($T^2 = S^1 \times S^1$)**: 2D periodic spatial position (e.g. grid cells in rodent navigation).

---

## 3. Representation Warping Through Layers

Deep networks untangle complex, entangled manifolds by applying non-linear transformations:
$$
\mathcal{M}_0 \xrightarrow{f_1} \mathcal{M}_1 \xrightarrow{f_2} \dots \xrightarrow{f_L} \mathcal{M}_L
$$
In early layers, manifolds for distinct classes are tightly intertwined. In penultimate layers, they are separated and flattened into linearly separable clusters.
