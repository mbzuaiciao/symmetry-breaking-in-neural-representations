"""Experiment A: Seed Dependence and Spontaneous Symmetry Breaking.

Core Question:
Does the ensemble distribution of learned representations respect the original
task symmetry even though individual random training runs break it?
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

from symmetry_reps.metrics import order_parameter_projection
from symmetry_reps.models import StandardMLP
from symmetry_reps.visualization import plot_order_parameter_distribution


def generate_z2_dataset(
    n_samples: int = 400, noise: float = 0.1, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 2D continuous symmetric 4-cluster dataset with Z_2 symmetry.

    Cluster 0: (+1, +1) -> y = +1
    Cluster 1: (-1, -1) -> y = +1
    Cluster 2: (+1, -1) -> y = -1
    Cluster 3: (-1, +1) -> y = -1
    """
    np.random.seed(seed)
    n_per_cluster = n_samples // 4
    centers = np.array([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    labels = np.array([1, 1, 0, 0])

    X_list = []
    y_list = []
    for c_idx in range(4):
        X_c = centers[c_idx] + np.random.randn(n_per_cluster, 2) * noise
        X_list.append(X_c)
        y_list.append(np.full((n_per_cluster,), labels[c_idx]))

    X = torch.from_numpy(np.vstack(X_list)).float()
    y = torch.from_numpy(np.concatenate(y_list)).long()
    return X, y


def run_experiment_a(
    num_seeds: int = 25,
    epochs: int = 80,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if smoke:
        num_seeds = 5
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_train, y_train = generate_z2_dataset(n_samples=400, seed=123)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    order_parameters = []
    final_accuracies = []

    for seed in range(num_seeds):
        torch.manual_seed(seed * 100 + 7)
        np.random.seed(seed * 100 + 7)

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
            features = model.forward_features(X_train)
            logits = model(X_train)
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y_train).float().mean().item()

            # Measure order parameter
            op = order_parameter_projection(features, y_train)
            order_parameters.append(op)
            final_accuracies.append(acc)

    results = {
        "experiment": "Exp A - Seed Dependence",
        "num_seeds": num_seeds,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "order_parameters": order_parameters,
        "mean_order_param": float(np.mean(order_parameters)),
        "std_order_param": float(np.std(order_parameters)),
        "mean_accuracy": float(np.mean(final_accuracies)),
    }

    # Save results JSON
    json_path = os.path.join(output_dir, "exp_a_seed_dependence.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot figure
    fig_path = os.path.join(assets_dir, "exp_a_order_parameter_hist.png")
    plot_order_parameter_distribution(
        order_parameters,
        title=f"Exp A: Order Parameter Distribution Across {num_seeds} Seeds",
        save_path=fig_path,
    )

    print(
        f"[Exp A] Finished: Mean m = {results['mean_order_param']:.3f}, Std = {results['std_order_param']:.3f}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment A: Seed Dependence")
    parser.add_argument("--seeds", type=int, default=25, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_a(num_seeds=args.seeds, epochs=args.epochs, smoke=args.smoke)
