r"""Experiment C: External Symmetry-Breaking Field.

Core Question:
How does a small symmetry-breaking perturbation \epsilon bias the network's choice
of internal orientation, analogous to an external magnetic field in physical spin systems?
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
from symmetry_reps.metrics import order_parameter_projection
from symmetry_reps.models import StandardMLP
from symmetry_reps.visualization import plot_perturbation_response_curve


def run_experiment_c(
    epsilons: list[float] | None = None,
    seeds_per_eps: int = 20,
    epochs: int = 80,
    hidden_dim: int = 16,
    output_dir: str = "results",
    assets_dir: str = "assets",
    smoke: bool = False,
) -> dict:
    if epsilons is None:
        epsilons = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]

    if smoke:
        epsilons = [0.0, 1e-3, 1e-1]
        seeds_per_eps = 4
        epochs = 15

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    X_base, y_train = generate_z2_dataset(n_samples=400, seed=42)

    prob_pos_orientation = []
    mean_order_params = []
    raw_results = {}

    for eps in epsilons:
        # Add bias eps along the symmetry breaking axis (+1 for positive orientation)
        # Shift input features slightly along [eps, eps]
        X_perturbed = X_base.clone()
        X_perturbed = X_perturbed + eps * torch.tensor([[1.0, 0.0]])

        dataset = TensorDataset(X_perturbed, y_train)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        ops_for_eps = []
        for seed in range(seeds_per_eps):
            torch.manual_seed(seed * 300 + 42)
            np.random.seed(seed * 300 + 42)

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
                features = model.forward_features(X_perturbed)
                op = order_parameter_projection(features, y_train)
                ops_for_eps.append(op)

        frac_pos = float(np.mean([1.0 if val > 0 else 0.0 for val in ops_for_eps]))
        mean_op = float(np.mean(ops_for_eps))

        prob_pos_orientation.append(frac_pos)
        mean_order_params.append(mean_op)
        raw_results[str(eps)] = ops_for_eps

    results = {
        "experiment": "Exp C - External Symmetry Breaking Field",
        "epsilons": epsilons,
        "seeds_per_eps": seeds_per_eps,
        "prob_pos_orientation": prob_pos_orientation,
        "mean_order_params": mean_order_params,
        "raw_results": raw_results,
    }

    json_path = os.path.join(output_dir, "exp_c_external_field.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    fig_path = os.path.join(assets_dir, "exp_c_perturbation_response.png")
    plot_perturbation_response_curve(
        epsilons=epsilons,
        positive_fractions=prob_pos_orientation,
        title="Exp C: Orientation Selection vs External Bias $\\epsilon$",
        save_path=fig_path,
    )

    print(f"[Exp C] Finished. Perturbation response: {list(zip(epsilons, prob_pos_orientation))}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment C: External Field")
    parser.add_argument("--seeds", type=int, default=20, help="Seeds per epsilon")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    args = parser.parse_args()

    run_experiment_c(seeds_per_eps=args.seeds, epochs=args.epochs, smoke=args.smoke)
