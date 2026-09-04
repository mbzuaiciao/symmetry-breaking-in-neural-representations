# Tutorial 00: What This Repository Is About

## 1. Motivation: Symmetries and Representations in Deep Learning

Deep neural networks do not simply map inputs to outputs as black-box look-up tables; they construct intermediate **representations**—geometric configurations of activations in high-dimensional vector spaces. 

When a learning task possesses an intrinsic symmetry (such as reflection, rotation, translation, or permutation), a fundamental question arises at the intersection of machine learning, physics, and representation theory:

> **The Central Research Question:**
> *When does a neural network preserve the symmetry of a task, and when does optimization select an asymmetric internal representation?*

---

## 2. Why Representation Geometry Matters

Studying weights alone is often notoriously uninformative due to continuous scaling invariances, hidden unit permutations, and non-convex loss landscapes. In contrast, examining the **geometry of representations**—the pairwise distances, angles, eigenspectra, and intrinsic dimensionality of activation vectors across samples—reveals the coordinate-free invariant structure learned by the network.

---

## 3. The Phenomenon of Symmetry Breaking

In physics, **spontaneous symmetry breaking** occurs when a physical system governed by symmetric laws settles into an asymmetric ground state (such as ferromagnetism or crystallization). 

In neural networks:
- The task loss function $\mathcal{L}(\theta)$ may be invariant under a group action $G$.
- Multiple distinct, degenerate minima exist across parameter space.
- Training an individual network from random initialization breaks the symmetry by settling into a specific oriented basin.
- However, the *ensemble* distribution of learned models across random seeds preserves the underlying symmetry of the task.

---

## 4. Why This Matters Across Disciplines

1. **Representation Learning & World Models**: Understanding whether latent dynamics models faithfully preserve continuous Lie group symmetries ($SO(2), SE(3)$) or suffer coordinate drift.
2. **Mechanistic Interpretability**: Distinguishing between intrinsic semantic directions and arbitrary coordinate rotations selected by optimization noise.
3. **Systems Neuroscience**: Comparing biological population codes (e.g. grid cells, visual cortex) with artificial networks using Representational Similarity Analysis (RSA).

---

## 5. Repository Roadmap

The repository is structured into 12 progressive parts:

```
Part I: Foundations (Tutorials 00–04)
   ↓
Part II: Symmetry Breaking (Tutorials 05–07)
   ↓
Part III: Representation Geometry (Tutorials 08–12)
   ↓
Part IV: Training Dynamics (Tutorials 13–16)
   ↓
Part V: Architecture and Equivariance (Tutorials 17–19)
   ↓
Part VI: Loss Landscapes and Parameter Symmetry (Tutorials 20–23)
   ↓
Part VII: Neural Collapse and Emergent Geometry (Tutorials 24–25)
   ↓
Part VIII: World Models and Dynamics (Tutorials 26–28)
   ↓
Part IX: Topology and Manifolds (Tutorials 29–31)
   ↓
Part X: Neuroscience Connections (Tutorials 32–34)
   ↓
Part XI: Language and Modern Representation Learning (Tutorials 35–37)
   ↓
Part XII: Research Methodology (Tutorials 38–40)
```

Each tutorial is paired with executable Python implementations in `src/symmetry_reps/` and empirical validations in `experiments/`.
