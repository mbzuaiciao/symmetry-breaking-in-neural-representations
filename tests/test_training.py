"""Unit tests for training dynamics, trajectory tracking, and Hessian computation."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from symmetry_reps.models import StandardMLP
from symmetry_reps.training import (
    compute_hessian_eigenspectrum,
    estimate_top_hessian_eigenvalues,
    train_model,
)


def test_train_model_trajectory():
    torch.manual_seed(42)
    # Synthetic classification data
    X = torch.randn(40, 2)
    y = (X[:, 0] > 0).long()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=20)

    model = StandardMLP(input_dim=2, hidden_dim=4, output_dim=2, num_hidden_layers=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    trajectory = train_model(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=4,
        eval_freq=1,
    )

    assert len(trajectory.epochs) == 4
    assert len(trajectory.losses) == 4
    assert len(trajectory.accuracies) == 4
    assert len(trajectory.order_parameters) == 4
    assert len(trajectory.effective_ranks) == 4
    assert len(trajectory.cka_to_init) == 4
    assert len(trajectory.cka_to_final) == 4
    assert trajectory.cka_to_final[-1] == pytest.approx(1.0, abs=1e-4)


def test_hessian_eigenspectrum():
    torch.manual_seed(42)
    # Tiny linear model
    X = torch.randn(10, 2)
    y = torch.randn(10, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)

    model = nn.Linear(2, 1, bias=False)
    criterion = nn.MSELoss()

    eigenvalues, eigenvectors = compute_hessian_eigenspectrum(model, criterion, loader)
    assert len(eigenvalues) == 2
    # MSE loss Hessian for linear model is 2/N * X^T X, which is positive semi-definite
    assert torch.all(eigenvalues >= -1e-5)


def test_estimate_top_hessian_eigenvalues():
    torch.manual_seed(42)
    X = torch.randn(15, 2)
    y = torch.randn(15, 1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=15)

    model = nn.Linear(2, 1, bias=False)
    criterion = nn.MSELoss()

    exact_evals, _ = compute_hessian_eigenspectrum(model, criterion, loader)
    est_evals = estimate_top_hessian_eigenvalues(
        model, criterion, loader, num_eigenvalues=2, max_iter=30
    )

    assert len(est_evals) == 2
    assert est_evals[0] == pytest.approx(exact_evals[0].item(), rel=1e-2)
