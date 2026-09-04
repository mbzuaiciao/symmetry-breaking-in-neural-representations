"""Unit tests for neural network architectures, equivariant models, and parameter symmetries."""

import torch

from symmetry_reps.models import (
    CyclicEquivariantMLP,
    LatentDynamicsModel,
    StandardMLP,
    Z2EquivariantMLP,
    Z2InvariantMLP,
    permute_hidden_units,
    rescale_relu_weights,
)
from symmetry_reps.symmetry import CyclicGroup, Z2Group, check_equivariance, check_invariance


def test_standard_mlp_forward():
    model = StandardMLP(input_dim=4, hidden_dim=16, output_dim=2, num_hidden_layers=2)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 2)
    feat = model.forward_features(x)
    assert feat.shape == (8, 16)


def test_z2_invariant_mlp():
    model = Z2InvariantMLP(input_dim=4, hidden_dim=16, output_dim=1)
    x = torch.randn(20, 4)
    out_pos = model(x)
    out_neg = model(-x)
    assert torch.allclose(out_pos, out_neg, atol=1e-5)

    z2 = Z2Group()
    err, is_inv = check_invariance(model, z2, x)
    assert is_inv
    assert err < 1e-5


def test_z2_equivariant_mlp():
    model = Z2EquivariantMLP(input_dim=4, hidden_dim=16, output_dim=4)
    x = torch.randn(20, 4)
    out_pos = model(x)
    out_neg = model(-x)
    assert torch.allclose(out_neg, -out_pos, atol=1e-5)

    z2 = Z2Group()
    err, is_eq = check_equivariance(model, z2, x)
    assert is_eq
    assert err < 1e-5


def test_cyclic_equivariant_mlp():
    c4 = CyclicGroup(4)
    model = CyclicEquivariantMLP(order=4, hidden_dim=16, output_dim=2)
    x = torch.randn(15, 2)
    err, is_eq = check_equivariance(model, c4, x)
    assert is_eq
    assert err < 1e-4


def test_latent_dynamics_model():
    model_std = LatentDynamicsModel(state_dim=2, hidden_dim=16, model_type="standard")
    model_eq = LatentDynamicsModel(state_dim=2, hidden_dim=16, model_type="equivariant")

    x = torch.randn(10, 2)
    out_std = model_std(x)
    out_eq = model_eq(x)
    assert out_std.shape == (10, 2)
    assert out_eq.shape == (10, 2)


def test_permute_hidden_units_functional_equivalence():
    torch.manual_seed(42)
    model = StandardMLP(input_dim=4, hidden_dim=8, output_dim=2, num_hidden_layers=2)
    x = torch.randn(20, 4)
    out_orig = model(x)

    # Permute hidden layer 0
    perm = torch.randperm(8).tolist()
    perm_model = permute_hidden_units(model, layer_idx=0, permutation=perm)

    out_perm = perm_model(x)
    assert torch.allclose(out_orig, out_perm, atol=1e-5)


def test_rescale_relu_weights_functional_equivalence():
    torch.manual_seed(42)
    model = StandardMLP(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        num_hidden_layers=2,
        activation="relu",
    )
    x = torch.randn(20, 4)
    out_orig = model(x)

    # Positive scale factors
    scales = torch.tensor([0.5, 2.0, 1.5, 3.0, 0.2, 1.1, 0.9, 2.5])
    scaled_model = rescale_relu_weights(model, layer_idx=0, scale_factors=scales)

    out_scaled = scaled_model(x)
    assert torch.allclose(out_orig, out_scaled, atol=1e-4)
