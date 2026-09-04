"""Metrics for representation similarity, order parameters, canonical angles,

and emergent geometric phenomena like neural collapse.
"""

from __future__ import annotations

import numpy as np
import torch


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix K: H K H where H = I - 1/n 1 1^T."""
    n = K.shape[0]
    unit = torch.ones(n, n, dtype=K.dtype, device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    r"""Compute Hilbert-Schmidt Independence Criterion (HSIC) between kernel matrices K and L:

    HSIC(K, L) = \frac{1}{(n - 1)^2} \text{tr}(K H L H)
    """
    K_c = _center_gram(K)
    L_c = _center_gram(L)
    n = K.shape[0]
    return (K_c * L_c.T).sum() / ((n - 1) ** 2)


def linear_cka(
    X1: torch.Tensor | np.ndarray,
    X2: torch.Tensor | np.ndarray,
    debiased: bool = False,
) -> float:
    r"""Compute Linear Centered Kernel Alignment (CKA) between representations X1 and X2.

    Let \tilde{X}_1 = X_1 - \mu_1 and \tilde{X}_2 = X_2 - \mu_2 be column-centered.
    HSIC(X_1 X_1^T, X_2 X_2^T) = \frac{1}{(n-1)^2} \| \tilde{X}_1^T \tilde{X}_2 \|_F^2

    CKA(X_1, X_2) = \frac{\| \tilde{X}_1^T \tilde{X}_2 \|_F^2}{\| \tilde{X}_1^T \tilde{X}_1 \|_F \| \tilde{X}_2^T \tilde{X}_2 \|_F}

    Properties:
    - Invariant to orthogonal transformation / rotation: CKA(X1 R1, X2 R2) = CKA(X1, X2) for R \in O(d)
    - Invariant to isotropic scaling: CKA(c1 X1, c2 X2) = CKA(X1, X2) for c > 0
    - Bounded in [0, 1]
    """
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1).float()
    else:
        X1 = X1.float()

    if isinstance(X2, np.ndarray):
        X2 = torch.from_numpy(X2).float()
    else:
        X2 = X2.float()

    if X1.shape[0] != X2.shape[0]:
        raise ValueError(
            f"Sample size mismatch: X1 has {X1.shape[0]} samples, X2 has {X2.shape[0]} samples"
        )

    N = X1.shape[0]
    if N < 2:
        return 1.0

    X1_c = X1 - X1.mean(dim=0, keepdim=True)
    X2_c = X2 - X2.mean(dim=0, keepdim=True)

    # Compute Frobenius norms of cross-covariance and auto-covariances
    # || X1_c^T X2_c ||_F^2
    cross = X1_c.T @ X2_c
    hsic_cross = torch.sum(cross**2)

    hsic_x1 = torch.sum((X1_c.T @ X1_c) ** 2)
    hsic_x2 = torch.sum((X2_c.T @ X2_c) ** 2)

    denom = torch.sqrt(hsic_x1 * hsic_x2)
    if denom.item() < 1e-12:
        return 0.0

    cka = hsic_cross / denom
    return float(torch.clamp(cka, 0.0, 1.0).item())


def order_parameter_projection(
    representations: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    direction: torch.Tensor | np.ndarray | None = None,
) -> float:
    r"""Compute an order parameter measuring the orientation / symmetry breaking of a representation.

    For a binary symmetric task (labels y \in {-1, +1}):
        m = \frac{1}{N} \sum_{i=1}^N y_i \langle h(x_i), v \rangle

    where v is the unit direction separating the two symmetric clusters (e.g. \mu_+ - \mu_- / \|\mu_+ - \mu_-\|).
    If direction v is not provided, it is estimated from the first principal component of class means.

    m \approx +c or -c indicates broken symmetry (spontaneous selection of one orientation).
    m \approx 0 indicates an unpolarized, symmetric representation.
    """
    if isinstance(representations, np.ndarray):
        H = torch.from_numpy(representations).float()
    else:
        H = representations.float()

    if isinstance(labels, np.ndarray):
        y = torch.from_numpy(labels).float()
    else:
        y = labels.float()

    if y.dim() > 1:
        y = y.squeeze(-1)

    # Map labels to {-1, +1} if binary {0, 1}
    unique_labels = torch.unique(y)
    if len(unique_labels) == 2 and 0.0 in unique_labels:
        y = torch.where(y == 0.0, torch.tensor(-1.0, dtype=y.dtype), y)

    mask_pos = y > 0
    mask_neg = y < 0

    if not torch.any(mask_pos) or not torch.any(mask_neg):
        return 0.0

    mu_pos = H[mask_pos].mean(dim=0)
    mu_neg = H[mask_neg].mean(dim=0)

    if direction is None:
        v = mu_pos - mu_neg
        norm_v = torch.norm(v)
        if norm_v.item() < 1e-12:
            return 0.0
        v = v / norm_v
    else:
        if isinstance(direction, np.ndarray):
            v = torch.from_numpy(direction).float()
        else:
            v = direction.float()
        v = v / torch.norm(v)

    # Compute mean signed projection
    projections = H @ v
    signed_proj = y * projections
    return float(signed_proj.mean().item())


def subspace_angles(
    X1: torch.Tensor | np.ndarray,
    X2: torch.Tensor | np.ndarray,
    k: int = 2,
) -> np.ndarray:
    r"""Compute principal angles \theta_1, ..., \theta_k between top-k principal subspaces of X1 and X2.

    Let U1 \in R^{d x k} and U2 \in R^{d x k} be orthonormal bases for top-k subspaces.
    Singular values of U1^T U2 are cos(\theta_i).
    """
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1).float()
    else:
        X1 = X1.float()

    if isinstance(X2, np.ndarray):
        X2 = torch.from_numpy(X2).float()
    else:
        X2 = X2.float()

    # PCA bases
    cov1 = (X1 - X1.mean(dim=0)).T @ (X1 - X1.mean(dim=0))
    cov2 = (X2 - X2.mean(dim=0)).T @ (X2 - X2.mean(dim=0))

    _, _, V1 = torch.linalg.svd(cov1)
    _, _, V2 = torch.linalg.svd(cov2)

    U1 = V1[:k].T  # (d, k)
    U2 = V2[:k].T  # (d, k)

    M = U1.T @ U2
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, -1.0, 1.0)
    angles_rad = torch.acos(s)

    return angles_rad.cpu().numpy()


def representation_distance(
    X1: torch.Tensor | np.ndarray,
    X2: torch.Tensor | np.ndarray,
    metric: str = "frobenius",
) -> float:
    """Compute distance between two aligned representation sets."""
    if isinstance(X1, np.ndarray):
        X1 = torch.from_numpy(X1).float()
    else:
        X1 = X1.float()

    if isinstance(X2, np.ndarray):
        X2 = torch.from_numpy(X2).float()
    else:
        X2 = X2.float()

    if metric == "frobenius":
        return float(torch.norm(X1 - X2, p="fro").item())
    elif metric == "normalized_frobenius":
        norm_ref = max(torch.norm(X2, p="fro").item(), 1e-12)
        return float(torch.norm(X1 - X2, p="fro").item() / norm_ref)
    elif metric == "cosine":
        cos = torch.nn.functional.cosine_similarity(X1, X2, dim=-1)
        return float((1.0 - cos).mean().item())
    else:
        raise ValueError(f"Unknown metric {metric}")


def neural_collapse_metrics(
    features: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    classifier_weights: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    r"""Compute standard Neural Collapse (NC) metrics (Papyan et al., 2020):

    NC1 (Variability collapse): Tr(\Sigma_W \Sigma_B^\dagger) / C
        Within-class covariance \Sigma_W collapses to zero relative to between-class covariance \Sigma_B.

    NC2 (Equiangular Simplex ETF geometry):
        Variance of cosine similarities between normalized class means \tilde{\mu}_c.
        Ideal Simplex ETF angle is -1 / (C - 1).

    NC3 (Classifier-to-feature duality):
        Average normalized cosine distance between normalized classifier weights w_c and centered class means \mu_c.
    """
    if isinstance(features, np.ndarray):
        H = torch.from_numpy(features).float()
    else:
        H = features.float()

    if isinstance(labels, np.ndarray):
        y = torch.from_numpy(labels).long()
    else:
        y = labels.long()

    if y.dim() > 1:
        y = y.squeeze(-1)

    classes = torch.unique(y)
    C = len(classes)
    N, d = H.shape

    if C < 2:
        return {"nc1": 0.0, "nc2_angle_var": 0.0, "nc3": 0.0}

    # Global mean
    mu_global = H.mean(dim=0, keepdim=True)

    # Class means and within-class covariance
    class_means = torch.zeros(C, d, dtype=H.dtype, device=H.device)
    Sigma_W = torch.zeros(d, d, dtype=H.dtype, device=H.device)

    for i, c in enumerate(classes):
        H_c = H[y == c]
        N_c = H_c.shape[0]
        if N_c > 0:
            mu_c = H_c.mean(dim=0)
            class_means[i] = mu_c
            diff = H_c - mu_c.unsqueeze(0)
            Sigma_W += diff.T @ diff

    Sigma_W = Sigma_W / max(N - C, 1)

    # Between-class covariance
    class_means_centered = class_means - mu_global
    Sigma_B = (class_means_centered.T @ class_means_centered) / C

    # NC1: Tr(Sigma_W @ pinv(Sigma_B)) / C
    try:
        Sigma_B_pinv = torch.linalg.pinv(Sigma_B)
        nc1 = float(torch.trace(Sigma_W @ Sigma_B_pinv).item() / C)
    except Exception:
        nc1 = 0.0

    # NC2: Equiangularity of class means
    norms = torch.norm(class_means_centered, dim=1, keepdim=True).clamp(min=1e-8)
    mu_normalized = class_means_centered / norms
    cos_matrix = mu_normalized @ mu_normalized.T

    # Extract off-diagonal entries
    mask = ~torch.eye(C, dtype=torch.bool, device=H.device)
    off_diag = cos_matrix[mask]
    target_cos = -1.0 / (C - 1)
    nc2_angle_var = float(torch.var(off_diag).item())
    nc2_target_diff = float(torch.abs(off_diag.mean() - target_cos).item())

    # NC3: Classifier weight duality
    nc3 = 0.0
    if classifier_weights is not None:
        if isinstance(classifier_weights, np.ndarray):
            W = torch.from_numpy(classifier_weights).float()
        else:
            W = classifier_weights.float()

        # W of shape (C, d)
        if W.shape == (C, d):
            W_norm = W / torch.norm(W, dim=1, keepdim=True).clamp(min=1e-8)
            cos_duality = (W_norm * mu_normalized).sum(dim=1)
            nc3 = float((1.0 - cos_duality).mean().item())

    return {
        "nc1": nc1,
        "nc2_angle_var": nc2_angle_var,
        "nc2_target_diff": nc2_target_diff,
        "nc3": nc3,
    }
