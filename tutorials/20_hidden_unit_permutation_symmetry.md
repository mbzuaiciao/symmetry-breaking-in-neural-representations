# Tutorial 20: Hidden-Unit Permutation Symmetry

## 1. Permutation Degeneracy in Deep Networks

Consider an $L$-layer fully connected network with hidden layer widths $H_1, H_2, \dots, H_{L-1}$.

For each layer $l$, there exist $H_l!$ discrete permutations of hidden units. The total number of functionally identical parameter configurations is:
$$
|\mathcal{G}_{\text{perm}}| = \prod_{l=1}^{L-1} H_l!
$$
For a tiny MLP with 2 hidden layers of width 64:
$$
(64!)^2 \approx (1.27 \times 10^{89})^2 \approx 1.6 \times 10^{178} \text{ identical global minima!}
$$

---

## 2. Permutation Mechanism

For layer $l$ with weight $W_l \in \mathbb{R}^{H_l \times H_{l-1}}$ and bias $b_l \in \mathbb{R}^{H_l}$, and next layer $W_{l+1} \in \mathbb{R}^{H_{l+1} \times H_l}$:
$$
W_l \leftarrow P_{\pi} W_l, \quad b_l \leftarrow P_{\pi} b_l, \quad W_{l+1} \leftarrow W_{l+1} P_{\pi}^T
$$
where $P_{\pi}$ is a permutation matrix.

Because element-wise activation functions $\sigma(\cdot)$ commute with permutations ($\sigma(P_{\pi} z) = P_{\pi} \sigma(z)$), the output remains exactly identical for all $x \in \mathbb{R}^{d_{\text{in}}}$.
