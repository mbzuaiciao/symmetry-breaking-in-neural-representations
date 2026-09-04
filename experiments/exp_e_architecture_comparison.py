"""Experiment E: Architecture Comparison (Hard-Coded Equivariance vs Learned Symmetry).

Core Question:
How does an architecture with hard-coded symmetry compare against an unconstrained
MLP in sample efficiency, generalization, symmetry error, and representation rank?
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
from symmetry_reps.geometry import effective_rank
from symmetry_reps.models import StandardMLP, Z2InvariantMLP
from symmetry_reps.symmetry import Z2Group, check_invariance


def run_experiment_e(
    sample_sizes: list[int] | None = None,
    seeds_per_config: int = 5,
    epochs: int = 100,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if sample_sizes is None:
        sample_sizes = [24, 48, 96, 192, 384]

    if smoke:
        sample_sizes = [24, 96]
        seeds_per_config = 2
        epochs = 20

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_test, y_test = generate_z2_dataset(n_samples=400, seed=999)
    z2 = Z2Group()

    results_unconstrained = {"acc": [], "sym_err": [], "erank": []}
    results_symmetric = {"acc": [], "sym_err": [], "erank": []}

    for n in sample_sizes:
        acc_u_list, sym_u_list, rk_u_list = [], [], []
        acc_s_list, sym_s_list, rk_s_list = [], [], []

        for seed in range(seeds_per_config):
            torch.manual_seed(seed * 111 + n)
            np.random.seed(seed * 111 + n)

            X_train, y_train = generate_z2_dataset(n_samples=n, seed=seed * 111 + n)
            loader = DataLoader(
                TensorDataset(X_train, y_train), batch_size=min(16, n), shuffle=True
            )

            # 1. Unconstrained MLP
            model_u = StandardMLP(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_hidden_layers=2,
                activation="tanh",
            )
            opt_u = torch.optim.Adam(model_u.parameters(), lr=0.02)
            crit = nn.CrossEntropyLoss()

            for _ in range(epochs):
                model_u.train()
                for bx, by in loader:
                    opt_u.zero_grad()
                    loss = crit(model_u(bx), by)
                    loss.backward()
                    opt_u.step()

            model_u.eval()
            with torch.no_grad():
                preds_u = torch.argmax(model_u(X_test), dim=-1)
                acc_u_list.append((preds_u == y_test).float().mean().item())
                feat_u = model_u.forward_features(X_test)
                rk_u_list.append(effective_rank(feat_u))
                err_u, _ = check_invariance(model_u, z2, X_test)
                sym_u_list.append(err_u)

            # 2. Symmetric MLP
            model_s = Z2InvariantMLP(
                input_dim=2,
                hidden_dim=hidden_dim,
                output_dim=2,
                num_hidden_layers=2,
                activation="tanh",
            )
            opt_s = torch.optim.Adam(model_s.parameters(), lr=0.02)

            for _ in range(epochs):
                model_s.train()
                for bx, by in loader:
                    opt_s.zero_grad()
                    loss = crit(model_s(bx), by)
                    loss.backward()
                    opt_s.step()

            model_s.eval()
            with torch.no_grad():
                preds_s = torch.argmax(model_s(X_test), dim=-1)
                acc_s_list.append((preds_s == y_test).float().mean().item())
                feat_s = model_s.forward_features(X_test)
                rk_s_list.append(effective_rank(feat_s))
                err_s, _ = check_invariance(model_s, z2, X_test)
                sym_s_list.append(err_s)

        results_unconstrained["acc"].append(float(np.mean(acc_u_list)))
        results_unconstrained["sym_err"].append(float(np.mean(sym_u_list)))
        results_unconstrained["erank"].append(float(np.mean(rk_u_list)))

        results_symmetric["acc"].append(float(np.mean(acc_s_list)))
        results_symmetric["sym_err"].append(float(np.mean(sym_s_list)))
        results_symmetric["erank"].append(float(np.mean(rk_s_list)))

    summary = {
        "experiment": "Exp E - Architecture Comparison",
        "sample_sizes": sample_sizes,
        "unconstrained": results_unconstrained,
        "symmetric": results_symmetric,
    }

    json_path = os.path.join(output_dir, "exp_e_architecture_comparison.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(sample_sizes, results_unconstrained["acc"], "o-", label="Standard MLP (Unconstrained)")
    ax1.plot(sample_sizes, results_symmetric["acc"], "s-", label="Symmetric Architecture")
    ax1.set_title("Test Accuracy vs Training Sample Size")
    ax1.set_xlabel("Number of Training Samples")
    ax1.set_ylabel("Test Accuracy")
    ax1.legend(frameon=True)

    ax2.plot(
        sample_sizes, results_unconstrained["sym_err"], "o-", label="Standard MLP (Unconstrained)"
    )
    ax2.plot(sample_sizes, results_symmetric["sym_err"], "s-", label="Symmetric Architecture")
    ax2.set_title("Symmetry Violation Error")
    ax2.set_xlabel("Number of Training Samples")
    ax2.set_ylabel("Invariance Error $E_{\\mathrm{inv}}$")
    ax2.set_yscale("log")
    ax2.legend(frameon=True)

    plt.suptitle("Exp E: Inductive Bias & Sample Efficiency", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(assets_dir, "exp_e_architecture_comparison.png")
    plt.savefig(fig_path, dpi=300)

    print(
        f"[Exp E] Finished. Final test acc: Unconstrained {results_unconstrained['acc'][-1]:.3f} vs Symmetric {results_symmetric['acc'][-1]:.3f}"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment E: Architecture Comparison")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_e(epochs=args.epochs, smoke=args.smoke)
