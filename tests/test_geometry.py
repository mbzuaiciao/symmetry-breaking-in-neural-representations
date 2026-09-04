"""Unit tests for covariance, Gram matrix, PCA spectra, effective rank,

participation ratio, anisotropy, and Procrustes alignment.
"""

import numpy as np
import pytest
import torch

from symmetry_reps.geometry import (
    compute_covariance,
    compute_gram_matrix,
    compute_pca_spectrum,
    covariance_anisotropy,
    effective_rank,
    orthogonal_procrustes,
    participation_ratio,
)


def test_compute_covariance():
    torch.manual_seed(42)
    X = torch.randn(100, 4)
    cov = compute_covariance(X)
    assert cov.shape == (4, 4)
    # Symmetry: cov = cov^T
    assert torch.allclose(cov, cov.T, atol=1e-6)
    # Positive semi-definiteness
    evals = torch.linalg.eigvalsh(cov)
    assert torch.all(evals >= -1e-6)

    # Compare with numpy
    cov_np = np.cov(X.numpy(), rowvar=False)
    assert np.allclose(cov.numpy(), cov_np, atol=1e-5)


def test_compute_gram_matrix():
    X = torch.randn(20, 5)
    K = compute_gram_matrix(X)
    assert K.shape == (20, 20)
    assert torch.allclose(K, K.T)
    evals = torch.linalg.eigvalsh(K)
    assert torch.all(evals >= -1e-5)


def test_pca_spectrum():
    # Rank-1 data along x-axis
    X = torch.zeros(50, 3)
    X[:, 0] = torch.randn(50) * 10.0
    evals, explained_var, components = compute_pca_spectrum(X)
    assert len(evals) == 3
    assert explained_var[0] > 0.99
    assert explained_var.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_effective_rank_bounds():
    # 1D line -> effective rank close to 1
    X_1d = torch.zeros(100, 10)
    X_1d[:, 0] = torch.linspace(-5, 5, 100)
    erank_1d = effective_rank(X_1d)
    assert 1.0 <= erank_1d <= 1.05

    # Isotropic high-dimensional sphere -> high effective rank
    torch.manual_seed(0)
    X_iso = torch.randn(500, 10)
    erank_iso = effective_rank(X_iso)
    assert erank_iso > 7.5


def test_participation_ratio():
    # 1D distribution
    evals_1d = torch.tensor([10.0, 0.0, 0.0])
    pr_1d = participation_ratio(evals_1d, is_eigenvalues=True)
    assert pr_1d == pytest.approx(1.0, abs=1e-4)

    # Completely isotropic 4D distribution
    evals_iso = torch.tensor([5.0, 5.0, 5.0, 5.0])
    pr_iso = participation_ratio(evals_iso, is_eigenvalues=True)
    assert pr_iso == pytest.approx(4.0, abs=1e-4)


def test_covariance_anisotropy():
    # Pure anisotropic (rank 1 in 2D)
    torch.tensor([10.0, 0.0])
    # (10 - 5) / 10 = 0.5
    X_aniso = torch.zeros(50, 2)
    X_aniso[:, 0] = torch.randn(50) * 5
    aniso_val = covariance_anisotropy(X_aniso)
    assert aniso_val > 0.45

    # Isotropic in 2D
    torch.manual_seed(42)
    X_iso = torch.randn(1000, 2)
    aniso_iso = covariance_anisotropy(X_iso)
    assert aniso_iso < 0.05


def test_orthogonal_procrustes():
    torch.manual_seed(42)
    X1 = torch.randn(100, 3)

    # Create true orthogonal rotation matrix (e.g. 90 deg around z)
    R_true = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    X2 = X1 @ R_true

    R_est, disparity, X1_aligned = orthogonal_procrustes(X1, X2)

    # Disparity should be 0
    assert disparity < 1e-5
    # R_est should be close to R_true
    assert torch.allclose(R_est, R_true, atol=1e-5)
    # Aligned X1 should match X2
    assert torch.allclose(X1_aligned, X2, atol=1e-5)
