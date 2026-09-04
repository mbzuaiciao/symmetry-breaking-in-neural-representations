"""Unit tests for mathematical groups, representations, orbits, and symmetry checks."""

import numpy as np
import torch

from symmetry_reps.symmetry import (
    ContinuousRotation2D,
    CyclicGroup,
    PermutationGroup,
    Z2Group,
    check_equivariance,
    check_invariance,
    compute_orbit,
    symmetry_error,
)


def test_z2_group_properties():
    g = Z2Group()
    assert g.name == "Z_2"
    assert g.order == 2
    assert g.identity() == 1
    assert g.elements() == [1, -1]
    assert g.compose(1, -1) == -1
    assert g.compose(-1, -1) == 1
    assert g.inverse(-1) == -1

    mat1 = g.action_matrix(1, dim=3)
    assert torch.allclose(mat1, torch.eye(3))

    mat_neg = g.action_matrix(-1, dim=3)
    assert torch.allclose(mat_neg, -torch.eye(3))

    ref = g.reflection_matrix(dim=3, axis=1)
    expected = torch.tensor([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    assert torch.allclose(ref, expected)


def test_cyclic_group_properties():
    c4 = CyclicGroup(4)
    assert c4.name == "C_4"
    assert c4.order == 4
    assert c4.identity() == 0
    assert c4.compose(1, 3) == 0
    assert c4.compose(2, 3) == 1
    assert c4.inverse(1) == 3

    # Action matrix orthogonality
    for elem in c4.elements():
        mat = c4.action_matrix(elem, dim=4)
        assert torch.allclose(mat @ mat.T, torch.eye(4), atol=1e-5)
        # 2D determinant is +1
        sub_2d = mat[:2, :2]
        det = torch.linalg.det(sub_2d)
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)


def test_permutation_group_properties():
    s3 = PermutationGroup(3)
    assert s3.name == "S_3"
    assert s3.order == 6
    assert s3.identity() == (0, 1, 2)

    # Test composition
    g1 = (1, 0, 2)  # swap 0 and 1
    g2 = (0, 2, 1)  # swap 1 and 2
    # g1(g2(0)) = g1(0) = 1
    # g1(g2(1)) = g1(2) = 2
    # g1(g2(2)) = g1(1) = 0
    assert s3.compose(g1, g2) == (1, 2, 0)
    assert s3.compose(g1, s3.inverse(g1)) == s3.identity()

    for p in s3.elements():
        mat = s3.action_matrix(p, dim=3)
        assert torch.allclose(mat @ mat.T, torch.eye(3))


def test_continuous_rotation():
    so2 = ContinuousRotation2D()
    assert so2.name == "SO(2)"
    assert so2.identity() == 0.0

    theta1 = np.pi / 4
    theta2 = np.pi / 2
    composed = so2.compose(theta1, theta2)
    assert np.isclose(composed, 3 * np.pi / 4)

    mat = so2.action_matrix(theta1, dim=2)
    assert torch.allclose(mat @ mat.T, torch.eye(2), atol=1e-5)


def test_compute_orbit():
    c4 = CyclicGroup(4)
    x = torch.tensor([[1.0, 0.0]])
    orbit = compute_orbit(c4, x)
    assert len(orbit) == 4
    # Points should be (1,0), (0,1), (-1,0), (0,-1)
    pts = torch.cat(orbit, dim=0)
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    assert torch.allclose(pts, expected, atol=1e-5)


def test_check_invariance():
    z2 = Z2Group()
    # Norm is sign-invariant
    def f_inv(x):
        return torch.norm(x, dim=-1, keepdim=True)
    # Linear function is not sign-invariant
    def f_non_inv(x):
        return x[..., :1]

    x = torch.randn(10, 4)
    err_inv, is_inv = check_invariance(f_inv, z2, x)
    assert is_inv
    assert err_inv < 1e-6

    err_non, is_non = check_invariance(f_non_inv, z2, x)
    assert not is_non
    assert err_non > 1e-2


def test_check_equivariance():
    c4 = CyclicGroup(4)
    # Scaling is rotation equivariant
    def f_equiv(x):
        return 2.5 * x
    # Constant offset is not rotation equivariant
    def f_non_equiv(x):
        return x + torch.tensor([[1.0, 0.0]])

    x = torch.randn(10, 2)
    err_eq, is_eq = check_equivariance(f_equiv, c4, x)
    assert is_eq
    assert err_eq < 1e-6

    err_non, is_non = check_equivariance(f_non_equiv, c4, x)
    assert not is_non
    assert err_non > 1e-2


def test_symmetry_error_dataloader():
    z2 = Z2Group()
    x = torch.randn(30, 4)
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    def f_inv(t):
        return torch.sum(t**2, dim=-1, keepdim=True)
    err = symmetry_error(f_inv, z2, loader, is_equivariant=False)
    assert err < 1e-5
