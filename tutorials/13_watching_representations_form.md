# Tutorial 13: Watching Representations Form: Representation Checkpointing

## 1. Instrumenting Neural Network Training

To analyze representation geometry dynamically, we must checkpoint activation matrices at discrete steps $t_0, t_1, \dots, t_T$ throughout optimization.

At each checkpoint $t$, we evaluate on a fixed validation set $\mathcal{D}_{\text{val}}$ and compute:
1. **Task Loss & Accuracy**: $\mathcal{L}(t), \text{Acc}(t)$.
2. **Order Parameter**: $m(t) = \frac{1}{N} \sum y_i \langle h_t(x_i), v \rangle$.
3. **Effective Rank & PR**: $\text{erank}(H_t), \text{PR}(\Sigma_t)$.
4. **CKA Similarity to Initialization**: $\text{CKA}(H_t, H_0)$.
5. **CKA Similarity to Final Convergence**: $\text{CKA}(H_t, H_T)$.

---

## 2. Typical Dynamical Regimes

1. **Initial Diffusion Phase**: Rapid loss descent, high effective rank, $m \approx 0$, $\text{CKA}(H_t, H_0)$ decays rapidly.
2. **Bifurcation Phase**: Loss plateau or saddle exit, rapid divergence of order parameter $m(t) \to \pm m^*$, effective rank contracts.
3. **Terminal Alignment Phase**: Minimal change in loss/accuracy, representations undergo rigid rotation or subtle contraction ($\text{CKA}(H_t, H_T) \to 1.0$).

---

## 3. Code Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from symmetry_reps.models import StandardMLP
from symmetry_reps.training import train_model

X = torch.randn(100, 2)
y = (X[:, 0] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=20)

model = StandardMLP(input_dim=2, hidden_dim=8, output_dim=2)
traj = train_model(
    model, loader, torch.optim.Adam(model.parameters(), lr=0.01), nn.CrossEntropyLoss(), epochs=30
)

print("Checkpoints recorded:", len(traj.epochs))
```
