"""Additional comprehensive tests for group homomorphisms, CKA invariance,

and representation geometry properties.
"""

import pytest
import torch

from symmetry_reps.geometry import (
    covariance_anisotropy,
    effective_rank,
    orthogonal_procrustes,
    participation_ratio,
)
from symmetry_reps.metrics import linear_cka, neural_collapse_metrics
from symmetry_reps.models import StandardMLP, permute_hidden_units, rescale_relu_weights
from symmetry_reps.symmetry import CyclicGroup, PermutationGroup, Z2Group


def test_cyclic_group_homomorphism():
    c6 = CyclicGroup(6)
    dim = 4
    for g1 in c6.elements():
        for g2 in c6.elements():
            rho_g1 = c6.action_matrix(g1, dim)
            rho_g2 = c6.action_matrix(g2, dim)
            g12 = c6.compose(g1, g2)
            rho_g12 = c6.action_matrix(g12, dim)
            assert torch.allclose(rho_g1 @ rho_g2, rho_g12, atol=1e-5)


def test_z2_group_homomorphism():
    z2 = Z2Group()
    dim = 5
    for g1 in z2.elements():
        for g2 in z2.elements():
            rho_g1 = z2.action_matrix(g1, dim)
            rho_g2 = z2.action_matrix(g2, dim)
            g12 = z2.compose(g1, g2)
            rho_g12 = z2.action_matrix(g12, dim)
            assert torch.allclose(rho_g1 @ rho_g2, rho_g12, atol=1e-5)


def test_permutation_group_homomorphism():
    s3 = PermutationGroup(3)
    dim = 3
    for g1 in s3.elements():
        for g2 in s3.elements():
            rho_g1 = s3.action_matrix(g1, dim)
            rho_g2 = s3.action_matrix(g2, dim)
            g12 = s3.compose(g1, g2)
            rho_g12 = s3.action_matrix(g12, dim)
            assert torch.allclose(rho_g1 @ rho_g2, rho_g12, atol=1e-5)


def test_cka_random_orthogonal_invariance_multi_dim():
    torch.manual_seed(123)
    N, D = 80, 12
    X1 = torch.randn(N, D)
    X2 = torch.randn(N, D)

    for _ in range(5):
        # Generate random Haar-distributed orthogonal matrix via QR
        Q1, _ = torch.linalg.qr(torch.randn(D, D))
        Q2, _ = torch.linalg.qr(torch.randn(D, D))

        cka_orig = linear_cka(X1, X2)
        cka_rot = linear_cka(X1 @ Q1, X2 @ Q2)
        assert cka_orig == pytest.approx(cka_rot, abs=1e-4)


def test_procrustes_identity():
    X = torch.randn(50, 4)
    R, disparity, aligned = orthogonal_procrustes(X, X)
    assert disparity < 1e-6
    assert torch.allclose(R, torch.eye(4), atol=1e-5)


def test_procrustes_reflection():
    X = torch.randn(50, 3)
    # Reflect z axis
    R_ref = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])
    X_ref = X @ R_ref

    R_est, disparity, aligned = orthogonal_procrustes(X, X_ref)
    assert disparity < 1e-5
    assert torch.allclose(aligned, X_ref, atol=1e-5)


def test_anisotropy_pure_diagonal():
    # Diagonal matrix with [10, 1, 1, 1]
    X = torch.zeros(100, 4)
    X[:, 0] = torch.randn(100) * 10.0
    X[:, 1:] = torch.randn(100, 3) * 1.0

    aniso = covariance_anisotropy(X)
    # Expected: (100 - 25.75) / 103 ~ 0.72
    assert aniso > 0.65


def test_effective_rank_rank_deficient():
    # Rank 2 subspace embedded in R^10
    X_sub = torch.randn(100, 2)
    proj = torch.randn(2, 10)
    X_embed = X_sub @ proj

    erank = effective_rank(X_embed)
    assert 1.8 <= erank <= 2.1


def test_participation_ratio_decay():
    # Steep exponential decay of eigenvalues
    evals = torch.tensor([100.0, 10.0, 1.0, 0.1])
    pr = participation_ratio(evals, is_eigenvalues=True)
    # sum = 111.1, sum_sq = 10000 + 100 + 1 + 0.01 = 10101.01
    # pr = (111.1^2) / 10101.01 ~ 1.22
    assert 1.15 <= pr <= 1.30


def test_neural_collapse_two_class():
    H1 = torch.randn(30, 4) * 0.01 + torch.tensor([1.0, 0, 0, 0])
    H2 = torch.randn(30, 4) * 0.01 - torch.tensor([1.0, 0, 0, 0])
    H = torch.cat([H1, H2], dim=0)
    y = torch.cat([torch.zeros(30), torch.ones(30)], dim=0).long()

    nc = neural_collapse_metrics(H, y)
    assert nc["nc1"] < 0.01
    # In 2-class, target angle is -1 / (2-1) = -1.0 (antipodal)
    assert nc["nc2_target_diff"] < 1e-3


def test_multi_layer_permutation():
    torch.manual_seed(99)
    model = StandardMLP(input_dim=3, hidden_dim=6, output_dim=2, num_hidden_layers=3)
    x = torch.randn(10, 3)
    out_orig = model(x)

    perm0 = torch.randperm(6)
    m1 = permute_hidden_units(model, layer_idx=0, permutation=perm0)

    perm1 = torch.randperm(6)
    m2 = permute_hidden_units(m1, layer_idx=1, permutation=perm1)

    out_perm = m2(x)
    assert torch.allclose(out_orig, out_perm, atol=1e-5)


def test_relu_rescaling_composition():
    torch.manual_seed(99)
    model = StandardMLP(
        input_dim=3, hidden_dim=4, output_dim=2, num_hidden_layers=2, activation="relu"
    )
    x = torch.randn(10, 3)
    out_orig = model(x)

    s1 = torch.tensor([2.0, 0.5, 3.0, 1.5])
    m_scaled = rescale_relu_weights(model, layer_idx=0, scale_factors=s1)

    s2 = torch.tensor([0.5, 2.0, 1.0 / 3.0, 1.0 / 1.5])
    m_restored = rescale_relu_weights(m_scaled, layer_idx=0, scale_factors=s2)

    out_scaled = m_scaled(x)
    out_restored = m_restored(x)

    assert torch.allclose(out_orig, out_scaled, atol=1e-4)
    assert torch.allclose(out_orig, out_restored, atol=1e-4)
