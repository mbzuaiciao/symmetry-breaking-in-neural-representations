"""Experiment H: Regularization and Representation Geometry.

Core Question:
How does weight decay / L2 regularization influence symmetry breaking,
representation anisotropy, and dimensionality collapse?
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
from symmetry_reps.geometry import covariance_anisotropy, effective_rank
from symmetry_reps.metrics import order_parameter_projection
from symmetry_reps.models import StandardMLP


def run_experiment_h(
    weight_decays: list[float] | None = None,
    seeds_per_wd: int = 15,
    epochs: int = 80,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if weight_decays is None:
        weight_decays = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    if smoke:
        weight_decays = [0.0, 1e-3, 1e-1]
        seeds_per_wd = 3
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_train, y_train = generate_z2_dataset(n_samples=400, seed=42)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    mean_eranks = []
    mean_anisotropies = []
    mean_abs_ops = []

    for wd in weight_decays:
        eranks = []
        anisos = []
        ops = []

        for seed in range(seeds_per_wd):
            torch.manual_seed(seed * 700 + int(wd * 1000))
            np.random.seed(seed * 700 + int(wd * 1000))

            model = StandardMLP(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_hidden_layers=2,
                activation="tanh",
            )
            opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=wd)
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
                eranks.append(effective_rank(feat))
                anisos.append(covariance_anisotropy(feat))
                op = order_parameter_projection(feat, y_train)
                ops.append(op)

        mean_eranks.append(float(np.mean(eranks)))
        mean_anisotropies.append(float(np.mean(anisos)))
        mean_abs_ops.append(float(np.mean(np.abs(ops))))

    results = {
        "experiment": "Exp H - Regularization",
        "weight_decays": weight_decays,
        "seeds_per_wd": seeds_per_wd,
        "mean_eranks": mean_eranks,
        "mean_anisotropies": mean_anisotropies,
        "mean_abs_ops": mean_abs_ops,
    }

    json_path = os.path.join(output_dir, "exp_h_regularization.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(weight_decays, mean_eranks, "o-", color="#1f77b4", label="Effective Rank")
    ax1.plot(weight_decays, mean_anisotropies, "s-", color="#d62728", label="Covariance Anisotropy")
    ax1.set_title("Dimensionality & Anisotropy vs Weight Decay")
    ax1.set_xlabel("Weight Decay $\\lambda$")
    ax1.set_ylabel("Metric Value")
    ax1.set_xscale("symlog", linthresh=1e-4)
    ax1.legend(frameon=True)

    ax2.plot(
        weight_decays,
        mean_abs_ops,
        "^-",
        color="#2ca02c",
        label="Mean Polarization $\\langle |m| \\rangle$",
    )
    ax2.set_title("Order Parameter Magnitude vs Weight Decay")
    ax2.set_xlabel("Weight Decay $\\lambda$")
    ax2.set_ylabel("$\\langle |m| \\rangle$")
    ax2.set_xscale("symlog", linthresh=1e-4)
    ax2.legend(frameon=True)

    plt.suptitle("Exp H: Effect of Weight Decay on Geometry", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(assets_dir, "exp_h_regularization.png")
    plt.savefig(fig_path, dpi=300)

    print(f"[Exp H] Finished: Weight decay sweep across {weight_decays}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment H: Regularization")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_h(epochs=args.epochs, smoke=args.smoke)
