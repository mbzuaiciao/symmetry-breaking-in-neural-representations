"""Publication-quality minimal Matplotlib visualization tools for symmetry,

representation geometry, dynamics, and alignment.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import torch

from symmetry_reps.geometry import (
    compute_pca_spectrum,
    effective_rank,
    orthogonal_procrustes,
    participation_ratio,
)
from symmetry_reps.metrics import linear_cka

# Set publication style defaults
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "lines.linewidth": 1.75,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


def _ensure_dir(save_path: str | None) -> None:
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)


def plot_symmetry_orbits(
    orbits: list[torch.Tensor | np.ndarray],
    labels: list[str] | None = None,
    title: str = "Group Action Orbits in Representation Space",
    save_path: str | None = None,
) -> Figure:
    """Plot 2D projections of symmetry orbits under group actions."""
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, orbit in enumerate(orbits):
        if isinstance(orbit, torch.Tensor):
            arr = orbit.detach().cpu().numpy()
        else:
            arr = np.asarray(orbit)

        label = labels[i] if labels and i < len(labels) else f"Orbit {i + 1}"
        col = colors[i % len(colors)]

        # Scatter points
        ax.scatter(arr[:, 0], arr[:, 1], color=col, s=40, alpha=0.85, label=label, zorder=3)
        # Closed polygon / trajectory loop
        if len(arr) > 2:
            loop = np.vstack([arr, arr[0:1]])
            ax.plot(loop[:, 0], loop[:, 1], color=col, linestyle=":", alpha=0.6, zorder=2)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_representation_pca(
    representations: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    title: str = "PCA Projection of Latent Representations",
    save_path: str | None = None,
) -> Figure:
    """Project high-dimensional representations to top 2 principal components."""
    if isinstance(representations, torch.Tensor):
        X = representations.detach().cpu().numpy()
    else:
        X = np.asarray(representations)

    if isinstance(labels, torch.Tensor):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)

    if y.ndim > 1:
        y = y.squeeze()

    _, _, components = compute_pca_spectrum(X)
    V2 = components[:, :2].detach().cpu().numpy()

    X_centered = X - X.mean(axis=0, keepdims=True)
    coords = X_centered @ V2

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = np.unique(y)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    for i, lab in enumerate(unique_labels):
        mask = y == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=colors[i % len(colors)],
            label=f"Class {int(lab)}",
            alpha=0.75,
            edgecolors="none",
            s=35,
        )

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_eigenspectra_and_rank(
    representations: torch.Tensor | np.ndarray,
    title: str = "Covariance Eigenspectrum & Effective Dimensionality",
    save_path: str | None = None,
) -> Figure:
    """Plot scree plot of covariance eigenvalues, cumulative variance, and effective rank."""
    if isinstance(representations, torch.Tensor):
        X = representations.detach().cpu()
    else:
        X = torch.from_numpy(representations)

    evals, explained_var, _ = compute_pca_spectrum(X)
    erank = effective_rank(X)
    pr = participation_ratio(X)

    evals_np = evals.numpy()
    cum_var_np = torch.cumsum(explained_var, dim=0).numpy()
    k_range = np.arange(1, len(evals_np) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Eigenspectrum
    ax1.plot(k_range, evals_np, marker="o", markersize=4, color="#1f77b4")
    ax1.set_title("Eigenvalue Decay")
    ax1.set_xlabel("Index k")
    ax1.set_ylabel("Eigenvalue $\\lambda_k$")
    ax1.set_yscale("log")

    # Cumulative variance & rank metrics
    ax2.plot(k_range, cum_var_np, marker="s", markersize=4, color="#2ca02c", label="Cum. Variance")
    ax2.axvline(
        erank,
        color="#d62728",
        linestyle="--",
        label=f"Effective Rank = {erank:.2f}",
    )
    ax2.axvline(
        pr,
        color="#9467bd",
        linestyle=":",
        label=f"Participation Ratio = {pr:.2f}",
    )
    ax2.set_title("Dimensionality Metrics")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=True, framealpha=0.9)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_order_parameter_trajectories(
    trajectories: list[dict[str, list[float]]],
    title: str = "Order Parameter Trajectories Across Seeds",
    save_path: str | None = None,
) -> Figure:
    """Plot order parameter evolution during training across multiple random seeds."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, traj in enumerate(trajectories):
        steps = traj.get("steps", list(range(len(traj["order_parameters"]))))
        ops = traj["order_parameters"]
        ax.plot(steps, ops, alpha=0.6, linewidth=1.2, label=f"Seed {i + 1}" if i < 5 else None)

    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Symmetric State")
    ax.set_title(title)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Order Parameter $m$")
    if len(trajectories) <= 8:
        ax.legend(frameon=True, framealpha=0.9, loc="best")
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_order_parameter_distribution(
    order_parameters: Sequence[float],
    bins: int = 15,
    title: str = "Ensemble Distribution of Order Parameter",
    save_path: str | None = None,
) -> Figure:
    """Plot histogram of final order parameters over many random initialization seeds."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ops = np.asarray(order_parameters)

    ax.hist(
        ops,
        bins=bins,
        color="#3b528b",
        edgecolor="black",
        alpha=0.75,
        density=True,
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Symmetric Unpolarized (m=0)")
    ax.axvline(
        float(np.mean(ops)),
        color="green",
        linestyle="-.",
        linewidth=1.5,
        label=f"Mean = {float(np.mean(ops)):.3f}",
    )

    ax.set_title(title)
    ax.set_xlabel("Order Parameter $m$")
    ax.set_ylabel("Empirical Probability Density")
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_perturbation_response_curve(
    epsilons: Sequence[float],
    positive_fractions: Sequence[float],
    title: str = "Orientation Selection vs External Field $\\epsilon$",
    save_path: str | None = None,
) -> Figure:
    """Plot probability of choosing orientation A as a function of external bias epsilon."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        epsilons,
        positive_fractions,
        marker="o",
        color="#d95f02",
        linewidth=2,
        markersize=6,
        label="Observed $P(m > 0)$",
    )
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="Unbiased Neutral (50%)")
    ax.set_title(title)
    ax.set_xlabel("External Perturbation $\\epsilon$")
    ax.set_ylabel("Probability of Orientation $A$ ($m > 0$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, framealpha=0.9)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_cka_matrix(
    cka_matrix: np.ndarray | torch.Tensor,
    labels: list[str] | None = None,
    title: str = "Linear CKA Similarity Matrix",
    save_path: str | None = None,
) -> Figure:
    """Plot heatmap of pairwise CKA similarities between models/checkpoints."""
    if isinstance(cka_matrix, torch.Tensor):
        mat = cka_matrix.detach().cpu().numpy()
    else:
        mat = np.asarray(cka_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA", rotation=270, labelpad=15)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    ax.set_title(title)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_procrustes_alignment(
    X1: torch.Tensor | np.ndarray,
    X2: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray | None = None,
    title: str = "Orthogonal Procrustes Latent Alignment",
    save_path: str | None = None,
) -> Figure:
    """Visualize two latent representations before and after Procrustes rotation."""
    R, disparity, X1_aligned = orthogonal_procrustes(X1, X2)

    X1_np = X1.detach().cpu().numpy() if isinstance(X1, torch.Tensor) else np.asarray(X1)
    X2_np = X2.detach().cpu().numpy() if isinstance(X2, torch.Tensor) else np.asarray(X2)
    X1_aligned_np = (
        X1_aligned.detach().cpu().numpy()
        if isinstance(X1_aligned, torch.Tensor)
        else np.asarray(X1_aligned)
    )

    # 2D projection via PCA of X2
    _, _, comp2 = compute_pca_spectrum(X2_np)
    V2 = comp2[:, :2].numpy()

    p_x1 = (X1_np - X1_np.mean(axis=0)) @ V2
    p_x2 = (X2_np - X2_np.mean(axis=0)) @ V2
    p_x1_al = (X1_aligned_np - X1_aligned_np.mean(axis=0)) @ V2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cka_before = linear_cka(X1_np, X2_np)
    linear_cka(X1_aligned_np, X2_np)

    # Before alignment
    ax1.scatter(p_x1[:, 0], p_x1[:, 1], color="#1f77b4", alpha=0.6, label="Model 1 (Raw)", s=25)
    ax1.scatter(p_x2[:, 0], p_x2[:, 1], color="#ff7f0e", alpha=0.6, label="Model 2 (Target)", s=25)
    ax1.set_title(f"Before Alignment (CKA = {cka_before:.3f})")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.legend(frameon=True)

    # After alignment
    ax2.scatter(
        p_x1_al[:, 0],
        p_x1_al[:, 1],
        color="#2ca02c",
        alpha=0.6,
        label="Model 1 (Rotated)",
        s=25,
    )
    ax2.scatter(p_x2[:, 0], p_x2[:, 1], color="#ff7f0e", alpha=0.6, label="Model 2 (Target)", s=25)
    ax2.set_title(f"After Alignment (Disparity = {disparity:.3f})")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.legend(frameon=True)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_neural_collapse_metrics(
    metrics_history: dict[str, list[float]],
    title: str = "Emergence of Neural Collapse Geometry",
    save_path: str | None = None,
) -> Figure:
    """Plot evolution of NC1, NC2, and NC3 metrics across training epochs."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))

    epochs = metrics_history.get("epochs", list(range(1, len(metrics_history["nc1"]) + 1)))

    # NC1
    ax1.plot(epochs, metrics_history["nc1"], color="#d62728", marker="o", markersize=3)
    ax1.set_title("NC1: Within-Class Collapse")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("$\\mathrm{Tr}(\\Sigma_W \\Sigma_B^\\dagger)/C$")
    ax1.set_yscale("log")

    # NC2
    ax2.plot(epochs, metrics_history["nc2_angle_var"], color="#1f77b4", marker="s", markersize=3)
    ax2.set_title("NC2: Simplex ETF Variance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Angle Variance")
    ax2.set_yscale("log")

    # NC3
    ax3.plot(epochs, metrics_history["nc3"], color="#2ca02c", marker="^", markersize=3)
    ax3.set_title("NC3: Classifier Duality")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Cosine Distance")

    plt.suptitle(title, y=1.03)
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig


def plot_hessian_spectrum(
    eigenvalues: torch.Tensor | np.ndarray,
    title: str = "Loss Hessian Eigenvalue Spectrum",
    save_path: str | None = None,
) -> Figure:
    """Plot sorted Hessian eigenvalues displaying sharp vs flat / near-zero directions."""
    if isinstance(eigenvalues, torch.Tensor):
        evals = eigenvalues.detach().cpu().numpy()
    else:
        evals = np.asarray(eigenvalues)

    evals = np.sort(evals)[::-1]
    k_range = np.arange(1, len(evals) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_range, evals, color="#9467bd", marker="o", markersize=3)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)

    # Highlight near-zero flat directions
    zero_count = np.sum(np.abs(evals) < 1e-4)
    ax.set_title(f"{title} ({zero_count} Near-Zero Directions)")
    ax.set_xlabel("Parameter Direction Index")
    ax.set_ylabel("Curvature / Eigenvalue $\\lambda$")
    plt.tight_layout()

    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig
