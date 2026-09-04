# Tutorial 26: Symmetry in Dynamical Systems

## 1. Symmetries of Differential Equations

Consider an autonomous continuous-time dynamical system:
$$
\dot{s}(t) = F(s(t)), \quad s \in \mathbb{R}^D
$$
A group $G$ is a **symmetry of the dynamics** if the vector field $F$ is equivariant under $G$:
$$
F(\rho(g) s) = \rho(g) F(s), \quad \forall g \in G
$$
This implies that if $s(t)$ is a valid trajectory, then $\tilde{s}(t) = \rho(g) s(t)$ is also a valid trajectory for all $g \in G$.

---

## 2. Discrete Transition Dynamics

In discrete time:
$$
s_{t+1} = \mathcal{T}(s_t), \quad \text{with } \mathcal{T}(\rho(g) s_t) = \rho(g) \mathcal{T}(s_t)
$$
Example: 2D harmonic oscillator or rotational flow in polar coordinates $(\dot{r}, \dot{\theta}) = (0, \omega)$. The dynamics are invariant under continuous rotations $SO(2)$.
