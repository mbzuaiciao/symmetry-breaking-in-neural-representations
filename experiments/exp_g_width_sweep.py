"""Experiment G: Network Width and Representation Geometry.

Core Question:
How does network width affect symmetry breaking, order-parameter variance,
and representation rank? Do wider networks break symmetry more consistently?
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
from symmetry_reps.geometry import effective_rank, participation_ratio
from symmetry_reps.metrics import order_parameter_projection
from symmetry_reps.models import StandardMLP


def run_experiment_g(
    widths: list[int] | None = None,
    seeds_per_width: int = 15,
    epochs: int = 80,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if widths is None:
        widths = [4, 8, 16, 32, 64, 128]

    if smoke:
        widths = [4, 16, 64]
        seeds_per_width = 3
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_train, y_train = generate_z2_dataset(n_samples=400, seed=42)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    mean_eranks = []
    mean_prs = []
    op_variances = []
    mean_abs_ops = []

    for w in widths:
        eranks = []
        prs = []
        ops = []

        for seed in range(seeds_per_width):
            torch.manual_seed(seed * 600 + w)
            np.random.seed(seed * 600 + w)

            model = StandardMLP(
                input_dim=2,
                hidden_dim=w,
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
                eranks.append(effective_rank(feat))
                prs.append(participation_ratio(feat))
                op = order_parameter_projection(feat, y_train)
                ops.append(op)

        mean_eranks.append(float(np.mean(eranks)))
        mean_prs.append(float(np.mean(prs)))
        op_variances.append(float(np.var(ops)))
        mean_abs_ops.append(float(np.mean(np.abs(ops))))

    results = {
        "experiment": "Exp G - Width Sweep",
        "widths": widths,
        "seeds_per_width": seeds_per_width,
        "mean_eranks": mean_eranks,
        "mean_prs": mean_prs,
        "op_variances": op_variances,
        "mean_abs_ops": mean_abs_ops,
    }

    json_path = os.path.join(output_dir, "exp_g_width_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(widths, mean_eranks, "o-", color="#1f77b4", label="Effective Rank")
    ax1.plot(widths, mean_prs, "s-", color="#2ca02c", label="Participation Ratio")
    ax1.set_title("Effective Dimensionality vs Width")
    ax1.set_xlabel("Network Width (Hidden Dimension)")
    ax1.set_ylabel("Dimension Metric")
    ax1.set_xscale("log", base=2)
    ax1.legend(frameon=True)

    ax2.plot(
        widths,
        op_variances,
        "^-",
        color="#d62728",
        label="Order Parameter Variance $\\mathrm{Var}(m)$",
    )
    ax2.plot(
        widths,
        mean_abs_ops,
        "d--",
        color="#9467bd",
        label="Mean Polarization $\\langle |m| \\rangle$",
    )
    ax2.set_title("Symmetry Breaking Polarization vs Width")
    ax2.set_xlabel("Network Width (Hidden Dimension)")
    ax2.set_ylabel("Order Parameter Statistic")
    ax2.set_xscale("log", base=2)
    ax2.legend(frameon=True)

    plt.suptitle("Exp G: Scaling of Representation Geometry with Width", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(assets_dir, "exp_g_width_sweep.png")
    plt.savefig(fig_path, dpi=300)

    print(f"[Exp G] Finished: Width sweep across {widths}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment G: Width Sweep")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_g(epochs=args.epochs, smoke=args.smoke)
