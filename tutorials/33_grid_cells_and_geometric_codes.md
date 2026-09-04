# Tutorial 33: Grid Cells and Geometric Neural Codes

## 1. Hexagonal Symmetry in Entorhinal Cortex

Grid cells in the medial entorhinal cortex (MEC) fire at the vertices of a periodic 2D triangular/hexagonal lattice as an animal navigates an open arena.

Why do biological representations spontaneously develop a 6-fold ($C_6$) hexagonal symmetry?

---

## 2. Theory of Optimal Geometric Coding

1. **Continuous Attractor Neural Networks (CANNs)**:
   Recurrent inhibition and excitation on a neural sheet with translational invariance create a periodic activity bump whose manifold topology is a **torus $T^2$**.
2. **Sphere Packing and Fisher Information**:
   In 2D Euclidean space, the hexagonal lattice provides the densest sphere packing and maximizes spatial Fisher information per neuron.
3. **Emergence in Artificial Networks**:
   When training recurrent networks or autoencoders on path integration (predicting position $(x, y)$ from velocity inputs $(\dot{x}, \dot{y})$), hexagonal grid patterns spontaneously emerge without any hard-coded lattice constraints (Banino et al., 2018; Cueva & Wei, 2018).
