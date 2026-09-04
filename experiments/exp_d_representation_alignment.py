"""Experiment D: Representation Alignment and Latent Geometry.

Core Question:
Are representations trained from different initializations genuinely different,
or are they isometric / symmetry-equivalent rotations in activation space?
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.exp_a_seed_dependence import generate_z2_dataset
from symmetry_reps.geometry import orthogonal_procrustes
from symmetry_reps.metrics import linear_cka
from symmetry_reps.models import StandardMLP
from symmetry_reps.visualization import plot_cka_matrix, plot_procrustes_alignment


def run_experiment_d(
    num_models: int = 6,
    epochs: int = 80,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if smoke:
        num_models = 4
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_train, y_train = generate_z2_dataset(n_samples=400, seed=42)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    trained_representations = []

    for idx in range(num_models):
        torch.manual_seed(idx * 500 + 99)
        np.random.seed(idx * 500 + 99)

        model = StandardMLP(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_hidden_layers=2,
            activation="tanh",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            feat = model.forward_features(X_train)
            trained_representations.append(feat)

    # Compute pairwise CKA matrix
    cka_mat = np.zeros((num_models, num_models))
    for i in range(num_models):
        for j in range(num_models):
            cka_mat[i, j] = linear_cka(trained_representations[i], trained_representations[j])

    # Perform Procrustes alignment between Model 0 and Model 1
    R, disparity, aligned_0 = orthogonal_procrustes(
        trained_representations[0], trained_representations[1]
    )

    results = {
        "experiment": "Exp D - Representation Alignment",
        "num_models": num_models,
        "mean_pairwise_cka": float(np.mean(cka_mat[np.triu_indices(num_models, k=1)])),
        "procrustes_disparity_0_1": float(disparity),
        "cka_matrix": cka_mat.tolist(),
    }

    json_path = os.path.join(output_dir, "exp_d_representation_alignment.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Visualizations
    fig_cka = os.path.join(assets_dir, "exp_d_cka_matrix.png")
    labels = [f"Model {i + 1}" for i in range(num_models)]
    plot_cka_matrix(
        cka_mat, labels=labels, title="Exp D: Pairwise Representation CKA", save_path=fig_cka
    )

    fig_proc = os.path.join(assets_dir, "exp_d_procrustes_alignment.png")
    plot_procrustes_alignment(
        trained_representations[0],
        trained_representations[1],
        title="Exp D: Orthogonal Procrustes Alignment (Model 1 vs Model 2)",
        save_path=fig_proc,
    )

    print(
        f"[Exp D] Finished. Mean pairwise CKA: {results['mean_pairwise_cka']:.3f}, Procrustes Disparity: {disparity:.3f}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment D: Representation Alignment")
    parser.add_argument("--models", type=int, default=6, help="Number of models to train")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_d(num_models=args.models, epochs=args.epochs, smoke=args.smoke)
