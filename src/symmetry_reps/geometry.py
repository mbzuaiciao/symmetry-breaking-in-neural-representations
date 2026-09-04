"""Representation geometry: Covariance, Gram matrices, PCA spectra, effective rank,

participation ratio, anisotropy, and Procrustes alignment.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_covariance(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Compute sample covariance matrix of activations X of shape (N, D).

    Cov(X) = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \mu)(x_i - \mu)^T
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor of shape (N, D), got {X.shape}")

    N = X.shape[0]
    if N < 2:
        raise ValueError("At least 2 samples required to compute sample covariance.")

    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (N - 1)
    return cov


def compute_gram_matrix(X: torch.Tensor | np.ndarray) -> torch.Tensor:
    r"""Compute sample Gram (kernel) matrix K of shape (N, N).

    K_{ij} = \langle x_i, x_j \rangle = (X X^T)_{ij}
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor of shape (N, D), got {X.shape}")

    return X @ X.T


def compute_pca_spectrum(
    X: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute PCA eigenvalues, explained variance ratio, and principal components.

    Returns:
        eigenvalues: (D,) sorted descending
        explained_variance_ratio: (D,) sum is 1.0
        components: (D, D) eigenvectors as columns
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    cov = compute_covariance(X)
    # eigh for symmetric matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Non-negative clamping for numerical stability
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    total_var = eigenvalues.sum()
    if total_var > 0:
        explained_variance_ratio = eigenvalues / total_var
    else:
        explained_variance_ratio = torch.zeros_like(eigenvalues)

    return eigenvalues, explained_variance_ratio, eigenvectors


def effective_rank(X: torch.Tensor | np.ndarray, eps: float = 1e-12) -> float:
    r"""Compute the Roy & Vetterli (2007) effective rank of representation matrix X.

    Let \sigma_i be singular values of X.
    Normalized distribution: p_i = \sigma_i / \sum_j \sigma_j
    Shannon entropy: H(p) = -\sum_i p_i \ln(p_i)
    Effective rank: erank(X) = \exp(H(p))

    Bounded by 1 <= erank(X) <= min(N, D).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    if X.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

    # Singular values
    s = torch.linalg.svdvals(X)
    s = s[s > eps]

    if len(s) == 0:
        return 0.0

    p = s / s.sum()
    entropy = -torch.sum(p * torch.log(p + eps)).item()
    return float(np.exp(entropy))


def participation_ratio(
    eigenvalues_or_X: torch.Tensor | np.ndarray, is_eigenvalues: bool = False
) -> float:
    r"""Compute the Participation Ratio (PR) of eigenvalues \lambda_i:

        PR = \frac{(\sum_{i=1}^d \lambda_i)^2}{\sum_{i=1}^d \lambda_i^2}

    Measures effective dimensionality occupied by covariance spectrum.
    Ranges from 1 (100% variance along 1 direction) to d (isotropic variance).
    """
    if isinstance(eigenvalues_or_X, np.ndarray):
        arr = torch.from_numpy(eigenvalues_or_X)
    else:
        arr = eigenvalues_or_X

    if is_eigenvalues:
        evals = arr.float()
    else:
        cov = compute_covariance(arr.float())
        evals, _ = torch.linalg.eigh(cov)

    evals = torch.clamp(evals, min=0.0)
    sum_evals = evals.sum()
    sum_sq_evals = (evals**2).sum()

    if sum_sq_evals.item() < 1e-12:
        return 0.0

    pr = (sum_evals**2) / sum_sq_evals
    return float(pr.item())


def covariance_anisotropy(X: torch.Tensor | np.ndarray) -> float:
    r"""Compute anisotropy index of covariance matrix:

        Anisotropy = \frac{\lambda_1 - \bar{\lambda}}{\sum_{i=1}^d \lambda_i}

    where \lambda_1 is top eigenvalue and \bar{\lambda} is average eigenvalue.
    Ranges from 0 (perfectly isotropic / spherical) to ~1 (fully anisotropic / rank-1).
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)

    evals, _, _ = compute_pca_spectrum(X)
    d = len(evals)
    if d <= 1 or evals.sum().item() < 1e-12:
        return 0.0

    lambda_max = evals[0].item()
    lambda_mean = evals.mean().item()
    total_var = evals.sum().item()

    return float((lambda_max - lambda_mean) / total_var)


def orthogonal_procrustes(
    X1: torch.Tensor | np.ndarray,
    X2: torch.Tensor | np.ndarray,
    center: bool = True,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    r"""Solve the Orthogonal Procrustes problem:

        \min_{R \in O(d)} \| X_1 R - X_2 \|_F^2

    where X1 and X2 are representation matrices of shape (N, d).

    Solution via SVD of M = X_1^T X_2 = U \Sigma V^T:
        R^* = U V^T

    Returns:
        R: (d, d) optimal orthogonal rotation/reflection matrix
        disparity: normalized Frobenius distance \| X_1 R - X_2 \|_F / \| X_2 \|_F
        aligned_X1: X1 @ R
    """
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1).float()
    else:
        X1 = X1.float()

    if isinstance(X2, np.ndarray):
        X2 = torch.from_numpy(X2).float()
    else:
        X2 = X2.float()

    if X1.shape != X2.shape:
        raise ValueError(f"Shape mismatch: X1 {X1.shape} vs X2 {X2.shape}")

    if center:
        X1_mean = X1.mean(dim=0, keepdim=True)
        X2_mean = X2.mean(dim=0, keepdim=True)
        X1_c = X1 - X1_mean
        X2_c = X2 - X2_mean
    else:
        X1_mean = torch.zeros(1, X1.shape[1], dtype=X1.dtype, device=X1.device)
        X2_mean = torch.zeros(1, X2.shape[1], dtype=X2.dtype, device=X2.device)
        X1_c = X1
        X2_c = X2

    M = X1_c.T @ X2_c
    U, _, Vh = torch.linalg.svd(M)
    R = U @ Vh

    aligned_X1 = (X1_c @ R) + X2_mean
    norm_X2 = torch.norm(X2, p="fro").item()
    diff_norm = torch.norm(aligned_X1 - X2, p="fro").item()

    disparity = diff_norm / max(norm_X2, 1e-12)

    return R, float(disparity), aligned_X1
