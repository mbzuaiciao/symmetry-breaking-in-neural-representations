"""Neural network architectures, equivariant layers, latent dynamical models,

and symmetry transformation operations on model parameters.
"""

from __future__ import annotations

import copy
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(act_name: str) -> nn.Module:
    name = act_name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu":
        return nn.SiLU()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation: {act_name}")


class StandardMLP(nn.Module):
    """Standard multi-layer perceptron with feature extraction capabilities."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "relu",
        use_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers

        layers = []
        in_d = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_d, hidden_dim, bias=use_bias))
            layers.append(_get_activation(activation))
            in_d = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=use_bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate layer activations."""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)
        return self.classifier(h)


class Z2InvariantMLP(nn.Module):
    r"""Exact \mathbb{Z}_2 invariant neural network: f(-x) = f(x) (or sign-flip invariance).

    Implements exact Reynolds group-averaging:
        f(x) = \frac{1}{2} (g(x) + g(-x))
    where g is an unconstrained MLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.base_mlp = StandardMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h_pos = self.base_mlp.forward_features(x)
        h_neg = self.base_mlp.forward_features(-x)
        return 0.5 * (h_pos + h_neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_pos = self.base_mlp(x)
        out_neg = self.base_mlp(-x)
        return 0.5 * (out_pos + out_neg)


class Z2EquivariantMLP(nn.Module):
    r"""Exact \mathbb{Z}_2 equivariant neural network: f(-x) = -f(x).

    Implements exact Reynolds odd group projection:
        f(x) = \frac{1}{2} (g(x) - g(-x))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 2,
        activation: str = "tanh",
    ):
        super().__init__()
        self.base_mlp = StandardMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            use_bias=False,
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h_pos = self.base_mlp.forward_features(x)
        h_neg = self.base_mlp.forward_features(-x)
        return 0.5 * (h_pos - h_neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_pos = self.base_mlp(x)
        out_neg = self.base_mlp(-x)
        return 0.5 * (out_pos - out_neg)


class CyclicEquivariantMLP(nn.Module):
    r"""Exact C_n rotation-equivariant group-convolutional network on 2D inputs."""

    def __init__(
        self,
        order: int = 4,
        hidden_dim: int = 32,
        output_dim: int = 2,
    ):
        super().__init__()
        self.order = order
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Filter parameter in R^{hidden_dim x 2}
        self.weight1 = nn.Parameter(torch.randn(hidden_dim, 2) / np.sqrt(2))
        self.weight2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim))
        self.out_weight = nn.Parameter(torch.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim))

    def _get_rotation_matrices(self) -> torch.Tensor:
        thetas = [2.0 * np.pi * k / self.order for k in range(self.order)]
        mats = []
        for th in thetas:
            mats.append(
                torch.tensor(
                    [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=torch.float32
                )
            )
        return torch.stack(mats, dim=0)  # (K, 2, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        # Group conv 1: lift to functions on G
        # R: (K, 2, 2)
        rot_mats = self._get_rotation_matrices().to(x.device)  # (K, 2, 2)
        # rotated inputs: (B, K, 2) = x @ rot_mats^T
        x_rot = torch.einsum("bi,kij->bkj", x, rot_mats)
        # features: (B, K, H) = relu(x_rot @ W1^T)
        h = F.relu(torch.einsum("bkj,hj->bkh", x_rot, self.weight1))
        # Group conv 2:
        h2 = F.relu(torch.einsum("bkh,mh->bkm", h, self.weight2))
        return h2.mean(dim=1)  # (B, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Equivariant map back to R^2
        rot_mats = self._get_rotation_matrices().to(x.device)  # (K, 2, 2)
        x_rot = torch.einsum("bi,kij->bkj", x, rot_mats)
        h = F.relu(torch.einsum("bkj,hj->bkh", x_rot, self.weight1))
        h2 = F.relu(torch.einsum("bkh,mh->bkm", h, self.weight2))
        out_k = torch.einsum("bkm,dm->bkd", h2, self.out_weight)  # (B, K, 2)
        # Un-rotate and average:
        # R^{-1} * out_k
        inv_rot = rot_mats.transpose(-1, -2)  # (K, 2, 2)
        unrot = torch.einsum("bkj,kjd->bkd", out_k, inv_rot)
        return unrot.mean(dim=1)


class LatentDynamicsModel(nn.Module):
    """Dynamics predictor (x_{t+1}, y_{t+1}) = f(x_t, y_t) with optional symmetry constraints."""

    def __init__(
        self,
        state_dim: int = 2,
        hidden_dim: int = 64,
        model_type: Literal["standard", "equivariant"] = "standard",
    ):
        super().__init__()
        self.state_dim = state_dim
        self.model_type = model_type

        if model_type == "equivariant":
            self.net = CyclicEquivariantMLP(order=8, hidden_dim=hidden_dim, output_dim=state_dim)
        else:
            self.net = StandardMLP(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                output_dim=state_dim,
                num_hidden_layers=2,
                activation="tanh",
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def forward_features(self, state: torch.Tensor) -> torch.Tensor:
        return self.net.forward_features(state)


def permute_hidden_units(
    model: StandardMLP,
    layer_idx: int,
    permutation: list[int] | np.ndarray | torch.Tensor,
) -> StandardMLP:
    r"""Apply a permutation \pi \in S_H to hidden layer `layer_idx` of a StandardMLP.

    Transforms weight matrices:
        W_l \leftarrow P_{\pi} W_l
        b_l \leftarrow P_{\pi} b_l
        W_{l+1} \leftarrow W_{l+1} P_{\pi}^T

    Preserves exact functional equivalence: f_{\pi}(x) = f(x) for all x.
    """
    new_model = copy.deepcopy(model)
    perm = torch.as_tensor(permutation, dtype=torch.long)

    # Extract linear layers from feature_extractor and classifier
    linear_layers: list[nn.Linear] = []
    for mod in new_model.feature_extractor:
        if isinstance(mod, nn.Linear):
            linear_layers.append(mod)
    linear_layers.append(new_model.classifier)

    if layer_idx < 0 or layer_idx >= len(linear_layers) - 1:
        raise ValueError(
            f"layer_idx must be between 0 and {len(linear_layers) - 2}, got {layer_idx}"
        )

    layer_curr = linear_layers[layer_idx]
    layer_next = linear_layers[layer_idx + 1]

    # Permute outgoing rows of current layer
    with torch.no_grad():
        layer_curr.weight.copy_(layer_curr.weight[perm])
        if layer_curr.bias is not None:
            layer_curr.bias.copy_(layer_curr.bias[perm])

        # Permute incoming columns of next layer
        layer_next.weight.copy_(layer_next.weight[:, perm])

    return new_model


def rescale_relu_weights(
    model: StandardMLP,
    layer_idx: int,
    scale_factors: torch.Tensor | np.ndarray | list[float],
) -> StandardMLP:
    r"""Apply positive diagonal scaling D = diag(s_1, ..., s_H) to a ReLU hidden layer.

    Since ReLU(c * z) = c * ReLU(z) for c > 0:
        W_l \leftarrow D W_l
        b_l \leftarrow D b_l
        W_{l+1} \leftarrow W_{l+1} D^{-1}

    Preserves exact functional equivalence f_D(x) = f(x).
    """
    new_model = copy.deepcopy(model)
    scales = torch.as_tensor(scale_factors, dtype=torch.float32)

    if torch.any(scales <= 0):
        raise ValueError("Scale factors must be strictly positive for ReLU scaling invariance.")

    linear_layers: list[nn.Linear] = []
    for mod in new_model.feature_extractor:
        if isinstance(mod, nn.Linear):
            linear_layers.append(mod)
    linear_layers.append(new_model.classifier)

    if layer_idx < 0 or layer_idx >= len(linear_layers) - 1:
        raise ValueError(
            f"layer_idx must be between 0 and {len(linear_layers) - 2}, got {layer_idx}"
        )

    layer_curr = linear_layers[layer_idx]
    layer_next = linear_layers[layer_idx + 1]

    with torch.no_grad():
        # Multiply row i by scales[i]
        layer_curr.weight.copy_(layer_curr.weight * scales.unsqueeze(1))
        if layer_curr.bias is not None:
            layer_curr.bias.copy_(layer_curr.bias * scales)

        # Divide column i of next layer by scales[i]
        layer_next.weight.copy_(layer_next.weight / scales.unsqueeze(0))

    return new_model
