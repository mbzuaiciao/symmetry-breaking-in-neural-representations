"""Unit tests for representation metrics, CKA invariance properties, order parameters,

and neural collapse metrics.
"""

import numpy as np
import pytest
import torch

from symmetry_reps.metrics import (
    linear_cka,
    neural_collapse_metrics,
    order_parameter_projection,
    representation_distance,
    subspace_angles,
)


def test_cka_self_similarity():
    torch.manual_seed(42)
    X = torch.randn(50, 8)
    cka_val = linear_cka(X, X)
    assert cka_val == pytest.approx(1.0, abs=1e-5)


def test_cka_orthogonal_invariance():
    torch.manual_seed(42)
    X1 = torch.randn(100, 10)
    X2 = torch.randn(100, 10)

    # Random orthogonal matrices via QR decomposition
    Q1, _ = torch.linalg.qr(torch.randn(10, 10))
    Q2, _ = torch.linalg.qr(torch.randn(10, 10))

    X1_rot = X1 @ Q1
    X2_rot = X2 @ Q2

    cka_orig = linear_cka(X1, X2)
    cka_rot = linear_cka(X1_rot, X2_rot)

    assert cka_orig == pytest.approx(cka_rot, abs=1e-4)


def test_cka_isotropic_scaling_invariance():
    torch.manual_seed(42)
    X1 = torch.randn(50, 6)
    X2 = torch.randn(50, 6)

    c1 = 7.5
    c2 = 0.02

    cka_orig = linear_cka(X1, X2)
    cka_scaled = linear_cka(c1 * X1, c2 * X2)

    assert cka_orig == pytest.approx(cka_scaled, abs=1e-4)


def test_order_parameter_projection():
    # Construct 2 polarized clusters
    X_pos = torch.randn(50, 4) + torch.tensor([5.0, 0.0, 0.0, 0.0])
    X_neg = torch.randn(50, 4) - torch.tensor([5.0, 0.0, 0.0, 0.0])

    X = torch.cat([X_pos, X_neg], dim=0)
    y = torch.cat([torch.ones(50), -torch.ones(50)], dim=0)

    op_val = order_parameter_projection(X, y)
    # Since clusters are well-separated along x[0] with +5 and -5, op should be large positive
    assert op_val > 4.5

    # If representations are mixed / unseparated:
    X_unsep = torch.randn(100, 4)
    op_unsep = order_parameter_projection(X_unsep, y)
    assert abs(op_unsep) < 0.5


def test_subspace_angles():
    # Identical subspaces -> angles should all be 0
    torch.manual_seed(42)
    X1 = torch.randn(100, 4)
    angles_identical = subspace_angles(X1, X1, k=2)
    assert np.allclose(angles_identical, [0.0, 0.0], atol=1e-3)

    # Orthogonal subspaces in R^4
    X_a = torch.zeros(100, 4)
    X_a[:, :2] = torch.randn(100, 2)

    X_b = torch.zeros(100, 4)
    X_b[:, 2:] = torch.randn(100, 2)

    angles_ortho = subspace_angles(X_a, X_b, k=2)
    assert np.allclose(angles_ortho, [np.pi / 2, np.pi / 2], atol=1e-4)


def test_representation_distance():
    X1 = torch.ones(10, 2)
    X2 = torch.zeros(10, 2)
    d_frob = representation_distance(X1, X2, metric="frobenius")
    assert d_frob == pytest.approx(np.sqrt(20), abs=1e-4)


def test_neural_collapse_metrics():
    # Create synthetic Simplex ETF for 3 classes in R^3
    C = 3
    # Simplex ETF in 3D: vertices of equilateral triangle centered at origin
    means = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3) / 2, 0.0],
            [-0.5, -np.sqrt(3) / 2, 0.0],
        ]
    )

    features = []
    labels = []
    for c in range(C):
        # Small noise around class mean
        h_c = means[c : c + 1] + torch.randn(20, 3) * 0.01
        features.append(h_c)
        labels.append(torch.full((20,), c, dtype=torch.long))

    H = torch.cat(features, dim=0)
    Y = torch.cat(labels, dim=0)

    # Perfect classifier weights matching class means
    W = means.clone()

    metrics = neural_collapse_metrics(H, Y, classifier_weights=W)

    # NC1 should be small
    assert metrics["nc1"] < 0.05
    # NC2 angle variance should be very small
    assert metrics["nc2_angle_var"] < 1e-4
    # Ideal Simplex angle diff: off-diagonal cos is -1/(3-1) = -0.5
    assert metrics["nc2_target_diff"] < 1e-3
    # NC3 duality distance should be near 0
    assert metrics["nc3"] < 1e-3
