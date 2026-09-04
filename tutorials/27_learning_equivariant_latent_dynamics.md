# Tutorial 27: Learning Equivariant Latent Dynamics

## 1. Latent World Models

A neural latent world model consists of:
1. **Encoder**: $z_t = E(x_t)$
2. **Transition Model**: $\hat{z}_{t+1} = T(z_t)$
3. **Decoder / Predictor**: $\hat{x}_{t+1} = D(\hat{z}_{t+1})$

---

## 2. When Does the Latent Space Preserve Symmetry?

When observations $x_t$ possess rotational symmetry $SO(2)$:
- **Equivariant Model**: The latent space $z$ carries an explicit representation $\rho_Z(g)$ (e.g. $C_8$ steerable filters), guaranteeing $T(\rho_Z(g) z) = \rho_Z(g) T(z)$.
- **Unconstrained Model**: The network is free to construct an arbitrary latent coordinate system. Even if it successfully predicts trajectories, the latent representation will typically break the continuous rotational symmetry into an anisotropic ellipse.

---

## 3. Code Example

```python
import torch
from symmetry_reps.models import LatentDynamicsModel

model_std = LatentDynamicsModel(state_dim=2, hidden_dim=32, model_type="standard")
model_eq = LatentDynamicsModel(state_dim=2, hidden_dim=32, model_type="equivariant")

z = torch.randn(5, 2)
print("Standard output shape:", model_std(z).shape)
print("Equivariant output shape:", model_eq(z).shape)
```
