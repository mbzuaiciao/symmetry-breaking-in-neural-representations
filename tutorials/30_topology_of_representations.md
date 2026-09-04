# Tutorial 30: Topology of Representations: Betti Numbers and Persistent Homology

## 1. Geometric vs. Topological Properties

- **Geometry** measures metric properties: distances, angles, curvatures, and volumes. (Changes under stretching or bending).
- **Topology** measures invariant qualitative connectivity: connected components, loops, and voids. (Invariant under continuous homeomorphisms).

---

## 2. Betti Numbers

The $k$-th Betti number $\beta_k$ counts the number of $k$-dimensional topological features:
- $\beta_0$: Number of **connected components**.
- $\beta_1$: Number of **1D loops / circular tunnels** (e.g. circle $S^1$ has $\beta_0 = 1, \beta_1 = 1$).
- $\beta_2$: Number of **2D enclosed voids / cavities** (e.g. sphere $S^2$ has $\beta_0 = 1, \beta_1 = 0, \beta_2 = 1$; torus $T^2$ has $\beta_0 = 1, \beta_1 = 2, \beta_2 = 1$).

---

## 3. Persistent Homology Intuition

Persistent Homology tracks topological features across multiple spatial resolution scales $\epsilon > 0$ by building Vietoris-Rips simplicial complexes:
- Short-lived features (birth $\approx$ death) correspond to sampling noise.
- Long-lived features (persisting across large scale ranges $\Delta \epsilon$) capture the true global topology of the neural representation.
