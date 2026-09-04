"""Training dynamics, representation checkpointing, trajectory tracking,

and loss landscape Hessian analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from symmetry_reps.geometry import (
    compute_pca_spectrum,
    effective_rank,
    participation_ratio,
)
from symmetry_reps.metrics import linear_cka, order_parameter_projection


@dataclass
class TrainingTrajectory:
    """Stores representation geometry and training statistics across training steps/epochs."""

    steps: list[int] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    order_parameters: list[float] = field(default_factory=list)
    effective_ranks: list[float] = field(default_factory=list)
    participation_ratios: list[float] = field(default_factory=list)
    leading_eigenvalues: list[list[float]] = field(default_factory=list)
    cka_to_init: list[float] = field(default_factory=list)
    cka_to_final: list[float] = field(default_factory=list)
    saved_representations: dict[int, torch.Tensor] = field(default_factory=dict)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int = 100,
    eval_dataset: tuple[torch.Tensor, torch.Tensor] | None = None,
    symmetry_dir: torch.Tensor | None = None,
    eval_freq: int = 1,
    device: str = "cpu",
) -> TrainingTrajectory:
    """Train a neural network while recording representation geometry trajectories."""
    model.to(device)
    trajectory = TrainingTrajectory()

    # Determine evaluation tensors
    if eval_dataset is not None:
        eval_x, eval_y = eval_dataset[0].to(device), eval_dataset[1].to(device)
    else:
        # Extract from first batch of train_loader
        first_batch = next(iter(train_loader))
        eval_x, eval_y = first_batch[0].to(device), first_batch[1].to(device)

    # Initial representation checkpoint (step 0)
    model.eval()
    feat_fn = getattr(model, "forward_features", None)
    with torch.no_grad():
        if callable(feat_fn):
            raw_init = feat_fn(eval_x)
            h_init = raw_init.detach().cpu() if isinstance(raw_init, torch.Tensor) else model(eval_x).detach().cpu()
        else:
            h_init = model(eval_x).detach().cpu()

    trajectory.saved_representations[0] = h_init
    step_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            step_count += 1
            epoch_loss += loss.item() * len(y)

            if out.dim() > 1 and out.size(-1) > 1:
                preds = torch.argmax(out, dim=-1)
                correct += (preds == y).sum().item()
            else:
                preds = (torch.sigmoid(out.squeeze()) > 0.5).long()
                correct += (preds == y.squeeze()).sum().item()
            total += len(y)

        if epoch % eval_freq == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                if callable(feat_fn):
                    raw_curr = feat_fn(eval_x)
                    h_curr = raw_curr.detach().cpu() if isinstance(raw_curr, torch.Tensor) else model(eval_x).detach().cpu()
                else:
                    h_curr = model(eval_x).detach().cpu()

                eval_out = model(eval_x)
                if eval_out.dim() > 1 and eval_out.size(-1) > 1:
                    eval_acc = (torch.argmax(eval_out, dim=-1) == eval_y).float().mean().item()
                else:
                    eval_preds = (torch.sigmoid(eval_out.squeeze()) > 0.5).long()
                    eval_acc = (eval_preds == eval_y.squeeze()).float().mean().item()

                op = order_parameter_projection(h_curr, eval_y.cpu(), direction=symmetry_dir)
                erank = effective_rank(h_curr)
                pr = participation_ratio(h_curr)
                evals, _, _ = compute_pca_spectrum(h_curr)
                top_evals = evals[: min(5, len(evals))].tolist()

                cka_init = linear_cka(h_curr, h_init)

                trajectory.steps.append(step_count)
                trajectory.epochs.append(epoch)
                trajectory.losses.append(epoch_loss / max(total, 1))
                trajectory.accuracies.append(eval_acc)
                trajectory.order_parameters.append(op)
                trajectory.effective_ranks.append(erank)
                trajectory.participation_ratios.append(pr)
                trajectory.leading_eigenvalues.append(top_evals)
                trajectory.cka_to_init.append(cka_init)
                trajectory.saved_representations[epoch] = h_curr

    # Compute CKA to final representation retrospectively
    h_final = trajectory.saved_representations[epochs]
    for ep in trajectory.epochs:
        h_ep = trajectory.saved_representations[ep]
        trajectory.cka_to_final.append(linear_cka(h_ep, h_final))

    return trajectory


def compute_hessian_eigenspectrum(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute full exact Hessian eigenvalues and eigenvectors for small models.

    Uses torch.autograd.functional.hessian on flattened parameter vector.
    """
    model.to(device)
    model.eval()

    # Collect dataset
    all_x = []
    all_y = []
    for batch in dataloader:
        all_x.append(batch[0])
        all_y.append(batch[1])
    X = torch.cat(all_x, dim=0).to(device)
    Y = torch.cat(all_y, dim=0).to(device)

    # Flatten parameter vector
    param_list = [p for p in model.parameters() if p.requires_grad]
    param_shapes = [p.shape for p in param_list]
    sum(p.numel() for p in param_list)

    def loss_from_flat_params(flat_params: torch.Tensor) -> torch.Tensor:
        # Reconstruct state dict / assign params
        offset = 0
        new_params = []
        for shape in param_shapes:
            count = int(np.prod(shape))
            p_chunk = flat_params[offset : offset + count].view(shape)
            new_params.append(p_chunk)
            offset += count

        # Functional forward pass via torch.func.functional_call
        params_dict = {}
        for (name, _), p_val in zip(model.named_parameters(), new_params):
            params_dict[name] = p_val

        out = torch.func.functional_call(model, params_dict, (X,))
        return criterion(out, Y)

    flat_init = torch.cat([p.detach().flatten() for p in param_list])

    # Compute Hessian matrix (num_params, num_params)
    H = torch.autograd.functional.hessian(loss_from_flat_params, flat_init)

    # Symmetric eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(H)
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def estimate_top_hessian_eigenvalues(
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
    num_eigenvalues: int = 5,
    max_iter: int = 50,
    device: str = "cpu",
) -> list[float]:
    """Estimate top Hessian eigenvalues via Power Iteration with deflation."""
    model.to(device)
    model.eval()

    all_x = []
    all_y = []
    for batch in dataloader:
        all_x.append(batch[0])
        all_y.append(batch[1])
    X = torch.cat(all_x, dim=0).to(device)
    Y = torch.cat(all_y, dim=0).to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    def hessian_vector_product(v_list: list[torch.Tensor]) -> list[torch.Tensor]:
        out = model(X)
        loss = criterion(out, Y)
        grads = list(torch.autograd.grad(loss, params, create_graph=True, retain_graph=True))
        grad_v = torch.stack([(g * v).sum() for g, v in zip(grads, v_list)]).sum()
        hvp = list(torch.autograd.grad(grad_v, params, retain_graph=True))
        return [h.detach() for h in hvp]

    def flatten(t_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten() for t in t_list])

    def unflatten(flat_t: torch.Tensor) -> list[torch.Tensor]:
        offset = 0
        out = []
        for p in params:
            count = p.numel()
            out.append(flat_t[offset : offset + count].view_as(p))
            offset += count
        return out

    found_eigenvalues = []
    found_eigenvectors = []

    for _ in range(num_eigenvalues):
        # Random starting vector
        v_flat = torch.randn(sum(p.numel() for p in params), device=device)
        v_flat = v_flat / torch.norm(v_flat)

        for _ in range(max_iter):
            # Orthogonalize against previously found eigenvectors
            for u in found_eigenvectors:
                v_flat = v_flat - (u @ v_flat) * u
            v_flat = v_flat / torch.norm(v_flat).clamp(min=1e-8)

            v_list = unflatten(v_flat)
            hvp_list = hessian_vector_product(v_list)
            hvp_flat = flatten(hvp_list)

            # Deflate
            for lam, u in zip(found_eigenvalues, found_eigenvectors):
                hvp_flat = hvp_flat - lam * (u @ v_flat) * u

            rayleigh = (v_flat @ hvp_flat).item()
            norm = torch.norm(hvp_flat).item()
            if norm > 1e-12:
                v_flat = hvp_flat / norm

        found_eigenvalues.append(rayleigh)
        found_eigenvectors.append(v_flat)

    return found_eigenvalues
