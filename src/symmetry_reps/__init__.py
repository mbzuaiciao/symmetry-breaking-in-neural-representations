"""Symmetry Breaking in Neural Representations.

A research and educational library for exploring symmetry, representation geometry,
and spontaneous symmetry breaking in neural networks.
"""

from symmetry_reps.geometry import (
    compute_covariance,
    compute_gram_matrix,
    compute_pca_spectrum,
    covariance_anisotropy,
    effective_rank,
    orthogonal_procrustes,
    participation_ratio,
)
from symmetry_reps.metrics import (
    linear_cka,
    neural_collapse_metrics,
    order_parameter_projection,
    representation_distance,
    subspace_angles,
)
from symmetry_reps.models import (
    CyclicEquivariantMLP,
    LatentDynamicsModel,
    StandardMLP,
    Z2EquivariantMLP,
    Z2InvariantMLP,
    permute_hidden_units,
    rescale_relu_weights,
)
from symmetry_reps.symmetry import (
    ContinuousRotation2D,
    CyclicGroup,
    Group,
    PermutationGroup,
    Z2Group,
    check_equivariance,
    check_invariance,
    compute_orbit,
    symmetry_error,
)
from symmetry_reps.training import (
    TrainingTrajectory,
    compute_hessian_eigenspectrum,
    estimate_top_hessian_eigenvalues,
    train_model,
)
from symmetry_reps.visualization import (
    plot_cka_matrix,
    plot_eigenspectra_and_rank,
    plot_hessian_spectrum,
    plot_neural_collapse_metrics,
    plot_order_parameter_distribution,
    plot_order_parameter_trajectories,
    plot_perturbation_response_curve,
    plot_procrustes_alignment,
    plot_representation_pca,
    plot_symmetry_orbits,
)

__version__ = "0.1.0"

__all__ = [
    "ContinuousRotation2D",
    "CyclicEquivariantMLP",
    "CyclicGroup",
    "Group",
    "LatentDynamicsModel",
    "PermutationGroup",
    "StandardMLP",
    "TrainingTrajectory",
    "Z2EquivariantMLP",
    "Z2Group",
    "Z2InvariantMLP",
    "check_equivariance",
    "check_invariance",
    "compute_covariance",
    "compute_gram_matrix",
    "compute_hessian_eigenspectrum",
    "compute_orbit",
    "compute_pca_spectrum",
    "covariance_anisotropy",
    "effective_rank",
    "estimate_top_hessian_eigenvalues",
    "linear_cka",
    "neural_collapse_metrics",
    "order_parameter_projection",
    "orthogonal_procrustes",
    "participation_ratio",
    "permute_hidden_units",
    "plot_cka_matrix",
    "plot_eigenspectra_and_rank",
    "plot_hessian_spectrum",
    "plot_neural_collapse_metrics",
    "plot_order_parameter_distribution",
    "plot_order_parameter_trajectories",
    "plot_perturbation_response_curve",
    "plot_procrustes_alignment",
    "plot_representation_pca",
    "plot_symmetry_orbits",
    "rescale_relu_weights",
    "representation_distance",
    "subspace_angles",
    "symmetry_error",
    "train_model",
]
