"""Experiment F: Data Asymmetry and Representation Phase Transitions.

Core Question:
When the ground-truth data-generating distribution transitions from perfectly symmetric
to asymmetric, how does the learned internal representation transition?
Is the transition smooth or abrupt?
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

from symmetry_reps.metrics import order_parameter_projection
from symmetry_reps.models import StandardMLP


def generate_asymmetric_dataset(
    asymmetry_delta: float, n_samples: int = 400, noise: float = 0.1, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 4-cluster dataset where cluster positions are shifted by asymmetry_delta."""
    np.random.seed(seed)
    n_per_cluster = n_samples // 4
    # Asymmetry delta shifts cluster 0 away from the symmetric pair
    centers = np.array(
        [
            [1.0 + asymmetry_delta, 1.0 + asymmetry_delta],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )
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


def run_experiment_f(
    deltas: list[float] | None = None,
    seeds_per_delta: int = 15,
    epochs: int = 80,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if deltas is None:
        deltas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

    if smoke:
        deltas = [0.0, 0.5, 1.0]
        seeds_per_delta = 3
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    mean_ops = []
    std_ops = []
    abs_mean_ops = []

    for delta in deltas:
        ops = []
        for seed in range(seeds_per_delta):
            torch.manual_seed(seed * 400 + int(delta * 100))
            np.random.seed(seed * 400 + int(delta * 100))

            X_train, y_train = generate_asymmetric_dataset(
                asymmetry_delta=delta, n_samples=400, seed=seed
            )
            dataset = TensorDataset(X_train, y_train)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model = StandardMLP(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_hidden_layers=2,
                activation="tanh",
            )
            opt = torch.optim.Adam(model.parameters(), lr=0.02)
            crit = nn.CrossEntropyLoss()

            for _ in range(epochs):
                model.train()
                for bx, by in loader:
                    opt.zero_grad()
                    loss = crit(model(bx), by)
                    loss.backward()
                    opt.step()

            model.eval()
            with torch.no_grad():
                feat = model.forward_features(X_train)
                op = order_parameter_projection(feat, y_train)
                ops.append(op)

        mean_ops.append(float(np.mean(ops)))
        std_ops.append(float(np.std(ops)))
        abs_mean_ops.append(float(np.mean(np.abs(ops))))

    results = {
        "experiment": "Exp F - Data Asymmetry",
        "deltas": deltas,
        "seeds_per_delta": seeds_per_delta,
        "mean_ops": mean_ops,
        "std_ops": std_ops,
        "abs_mean_ops": abs_mean_ops,
    }

    json_path = os.path.join(output_dir, "exp_f_data_asymmetry.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(
        deltas,
        mean_ops,
        yerr=std_ops,
        fmt="o-",
        color="#2b5c8f",
        ecolor="lightblue",
        elinewidth=2,
        capsize=4,
        label="Ensemble Mean Order Parameter $\\langle m \\rangle$",
    )
    ax.plot(deltas, abs_mean_ops, "s--", color="#d95f02", label="Mean Absolute $|m|$")
    ax.set_title("Exp F: Learned Symmetry Breaking vs Data Asymmetry $\\delta$")
    ax.set_xlabel("Data Asymmetry $\\delta$")
    ax.set_ylabel("Order Parameter $m$")
    ax.legend(frameon=True)
    plt.tight_layout()

    fig_path = os.path.join(assets_dir, "exp_f_data_asymmetry.png")
    plt.savefig(fig_path, dpi=300)

    print(f"[Exp F] Finished: Data asymmetry sweep across {deltas}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment F: Data Asymmetry")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_f(epochs=args.epochs, smoke=args.smoke)
