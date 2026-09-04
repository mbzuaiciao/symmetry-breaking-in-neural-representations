# Tutorial 04: Symmetry in Neural Networks

## 1. The Five Levels of Symmetry in Deep Learning

A crucial source of confusion in machine learning literature is failing to distinguish *where* a symmetry lives. We must strictly separate:

1. **Symmetry of the Data Generating Process**: The underlying physical or mathematical distribution $P(X)$ satisfies $P(\rho(g)X) = P(X)$.
2. **Symmetry of the Task**: The target label function satisfies $y(\rho(g)x) = y(x)$ (invariant) or $y(\rho(g)x) = \rho'(g)y(x)$ (equivariant).
3. **Symmetry of the Architecture**: The functional model satisfies $f(\rho(g)x; \theta) = \rho'(g) f(x; \theta)$ for all weights $\theta$.
4. **Symmetry of the Parameterization (Gauge Symmetries)**: Distinct weight configurations $\theta_1 \ne \theta_2$ produce identical functions $f(\cdot; \theta_1) \equiv f(\cdot; \theta_2)$ (e.g., hidden unit permutation $S_H$ or ReLU scale invariance).
5. **Symmetry of the Learned Representation**: The empirical activation distribution $H = \{h(x_i)\}$ satisfies $\Sigma = \rho(g) \Sigma \rho(g)^T$.

---

## 2. Parameter Symmetries: Permutation and Scaling

### Hidden Unit Permutation Symmetry ($S_H$)
In a standard fully connected layer $h = \sigma(W_1 x + b_1), \quad y = W_2 h$:
Applying any permutation matrix $P_{\pi}$ to the hidden units:
$$
\tilde{W}_1 = P_{\pi} W_1, \quad \tilde{b}_1 = P_{\pi} b_1, \quad \tilde{W}_2 = W_2 P_{\pi}^T
$$
yields an identical output for all inputs:
$$
\tilde{y} = (W_2 P_{\pi}^T) \sigma(P_{\pi} W_1 x + P_{\pi} b_1) = W_2 \sigma(W_1 x + b_1) = y
$$

### Continuous Scale Symmetry in ReLU Networks
Because $\text{ReLU}(c \cdot z) = c \cdot \text{ReLU}(z)$ for $c > 0$, for any positive diagonal matrix $D = \text{diag}(s_1, \dots, s_H)$:
$$
\tilde{W}_1 = D W_1, \quad \tilde{W}_2 = W_2 D^{-1} \implies \tilde{f}(x) = f(x)
$$

---

## 3. Empirical Demonstration

```python
import torch
from symmetry_reps.models import StandardMLP, permute_hidden_units

model = StandardMLP(input_dim=4, hidden_dim=8, output_dim=2)
x = torch.randn(5, 4)
orig_out = model(x)

# Apply permutation to hidden units
perm = [7, 6, 5, 4, 3, 2, 1, 0]
perm_model = permute_hidden_units(model, layer_idx=0, permutation=perm)
perm_out = perm_model(x)

print("Max difference after permutation:", torch.max(torch.abs(orig_out - perm_out)).item())
```

---

## 4. Scientific Discipline: What Can Fool Us?
- Confusing **parameter symmetry** with **task symmetry**: Just because two weight vectors $\theta_1, \theta_2$ yield the same loss does not mean the network has learned a symmetric representation of the task data!
