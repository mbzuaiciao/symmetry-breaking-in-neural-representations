"""Experiment B: Symmetry Breaking as a Dynamical Process.

Core Question:
How does symmetry breaking evolve over the course of gradient descent?
Does the representation undergo a sharp symmetry-breaking transition or gradual drift?
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.exp_a_seed_dependence import generate_z2_dataset
from symmetry_reps.models import StandardMLP
from symmetry_reps.training import train_model
from symmetry_reps.visualization import plot_order_parameter_trajectories


def run_experiment_b(
    num_seeds: int = 8,
    epochs: int = 100,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if smoke:
        num_seeds = 3
        epochs = 20

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_train, y_train = generate_z2_dataset(n_samples=400, seed=42)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    trajectories_data = []

    for seed in range(num_seeds):
        torch.manual_seed(seed * 200 + 13)
        np.random.seed(seed * 200 + 13)

        model = StandardMLP(
            input_dim=2,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_hidden_layers=2,
            activation="tanh",
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        criterion = nn.CrossEntropyLoss()

        traj = train_model(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            eval_dataset=(X_train, y_train),
            eval_freq=1,
        )

        trajectories_data.append(
            {
                "seed": seed,
                "epochs": traj.epochs,
                "steps": traj.steps,
                "losses": traj.losses,
                "accuracies": traj.accuracies,
                "order_parameters": traj.order_parameters,
                "effective_ranks": traj.effective_ranks,
                "participation_ratios": traj.participation_ratios,
                "cka_to_init": traj.cka_to_init,
                "cka_to_final": traj.cka_to_final,
            }
        )

    # Save results JSON
    json_path = os.path.join(output_dir, "exp_b_training_trajectories.json")
    with open(json_path, "w") as f:
        json.dump(trajectories_data, f, indent=2)

    # Plot 1: Order Parameter Trajectories
    fig_path1 = os.path.join(assets_dir, "exp_b_order_param_trajectories.png")
    plot_order_parameter_trajectories(
        trajectories_data,
        title=f"Exp B: Symmetry Breaking Dynamics ({num_seeds} Seeds)",
        save_path=fig_path1,
    )

    # Plot 2: Multi-panel Dynamics (Loss, Rank, CKA to Init)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8))
    for traj in trajectories_data:
        ax1.plot(traj["epochs"], traj["losses"], alpha=0.5)
        ax2.plot(traj["epochs"], traj["effective_ranks"], alpha=0.5)
        ax3.plot(traj["epochs"], traj["cka_to_init"], alpha=0.5)

    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy")

    ax2.set_title("Representation Effective Rank")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Effective Rank")

    ax3.set_title("CKA Similarity to Init")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Linear CKA")

    plt.suptitle("Exp B: Representation Geometry Dynamics", y=1.03)
    plt.tight_layout()
    fig_path2 = os.path.join(assets_dir, "exp_b_geometry_dynamics.png")
    plt.savefig(fig_path2, dpi=300)

    print(f"[Exp B] Finished tracking {num_seeds} training trajectories.")
    return {"num_seeds": num_seeds, "epochs": epochs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment B: Training Trajectories")
    parser.add_argument("--seeds", type=int, default=8, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_b(num_seeds=args.seeds, epochs=args.epochs, smoke=args.smoke)
