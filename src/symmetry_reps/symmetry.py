"""Mathematical foundations of groups, representations, and symmetry actions."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class Group(ABC):
    """Abstract base class for mathematical groups acting on tensor representations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the group."""

    @property
    @abstractmethod
    def order(self) -> int | float:
        """Order (number of elements) of the group. float('inf') for continuous groups."""

    @abstractmethod
    def elements(self) -> Sequence[Any]:
        """Return a sequence of all group elements (for finite groups) or sample elements."""

    @abstractmethod
    def identity(self) -> Any:
        """Return the identity element e of the group."""

    @abstractmethod
    def compose(self, g1: Any, g2: Any) -> Any:
        """Group composition g1 * g2."""

    @abstractmethod
    def inverse(self, g: Any) -> Any:
        """Group inverse g^{-1}."""

    @abstractmethod
    def action_matrix(self, g: Any, dim: int) -> torch.Tensor:
        """Return the linear representation matrix rho(g) in dimension dim."""

    def act(self, g: Any, x: torch.Tensor) -> torch.Tensor:
        """Apply group element g to tensor x via linear representation rho(g) x.

        Supports batched input x of shape (..., D).
        """
        dim = x.shape[-1]
        rho = self.action_matrix(g, dim).to(dtype=x.dtype, device=x.device)
        return torch.matmul(x, rho.T)


class Z2Group(Group):
    r"""The cyclic group of order 2: Z_2 = {+1, -1} (or {0, 1} additively).

    Often represented as reflection across an axis, sign flip, or binary inversion.
    """

    @property
    def name(self) -> str:
        return "Z_2"

    @property
    def order(self) -> int:
        return 2

    def elements(self) -> list[int]:
        return [1, -1]

    def identity(self) -> int:
        return 1

    def compose(self, g1: int, g2: int) -> int:
        return g1 * g2

    def inverse(self, g: int) -> int:
        return g

    def action_matrix(self, g: int, dim: int) -> torch.Tensor:
        """Action matrix: For g = 1, identity matrix.

        For g = -1, sign flip on selected axes (default: flip the first coordinate or all).
        Here we default to sign reflection on all coordinates: rho(-1) = -I.
        """
        if g == 1:
            return torch.eye(dim)
        elif g == -1:
            return -torch.eye(dim)
        else:
            raise ValueError(f"Invalid element {g} for Z_2")

    def reflection_matrix(self, dim: int, axis: int = 0) -> torch.Tensor:
        """Action matrix reflecting along a specific coordinate axis."""
        mat = torch.eye(dim)
        mat[axis, axis] = -1.0
        return mat


class CyclicGroup(Group):
    r"""Finite cyclic group of order n: C_n or Z_n = {0, 1, ..., n-1}.

    Under addition modulo n. Often realized as discrete 2D rotations by 2*pi*k / n.
    """

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("Group order n must be >= 1")
        self.n = n

    @property
    def name(self) -> str:
        return f"C_{self.n}"

    @property
    def order(self) -> int:
        return self.n

    def elements(self) -> list[int]:
        return list(range(self.n))

    def identity(self) -> int:
        return 0

    def compose(self, g1: int, g2: int) -> int:
        return (g1 + g2) % self.n

    def inverse(self, g: int) -> int:
        return (-g) % self.n

    def action_matrix(self, g: int, dim: int) -> torch.Tensor:
        """Action matrix.

        For dim=2: 2D rotation matrix by angle theta = 2*pi*g / n.
        For dim > 2: block diagonal 2D rotations on coordinate pairs.
        """
        if g not in range(self.n):
            raise ValueError(f"Invalid element {g} for C_{self.n}")

        theta = 2.0 * np.pi * g / self.n
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))

        mat = torch.eye(dim)
        for i in range(0, dim - 1, 2):
            mat[i, i] = cos_t
            mat[i, i + 1] = -sin_t
            mat[i + 1, i] = sin_t
            mat[i + 1, i + 1] = cos_t
        return mat


class PermutationGroup(Group):
    r"""Symmetric group S_n of permutations on n elements."""

    def __init__(self, n: int):
        if n < 1:
            raise ValueError("Permutation size n must be >= 1")
        self.n = n

    @property
    def name(self) -> str:
        return f"S_{self.n}"

    @property
    def order(self) -> int:
        return int(np.prod(range(1, self.n + 1)))

    def elements(self) -> list[tuple[int, ...]]:
        return list(itertools.permutations(range(self.n)))

    def identity(self) -> tuple[int, ...]:
        return tuple(range(self.n))

    def compose(self, g1: tuple[int, ...], g2: tuple[int, ...]) -> tuple[int, ...]:
        # (g1 * g2)(i) = g1(g2(i))
        return tuple(g1[g2[i]] for i in range(self.n))

    def inverse(self, g: tuple[int, ...]) -> tuple[int, ...]:
        inv = [0] * self.n
        for idx, val in enumerate(g):
            inv[val] = idx
        return tuple(inv)

    def action_matrix(self, g: tuple[int, ...], dim: int) -> torch.Tensor:
        if dim != self.n:
            raise ValueError(f"Permutation group S_{self.n} requires dimension {self.n}, got {dim}")
        mat = torch.zeros(dim, dim)
        for i, val in enumerate(g):
            mat[val, i] = 1.0
        return mat


class ContinuousRotation2D(Group):
    r"""Continuous 2D rotation group SO(2).

    Elements are represented by rotation angles theta in [0, 2*pi).
    """

    @property
    def name(self) -> str:
        return "SO(2)"

    @property
    def order(self) -> float:
        return float("inf")

    def elements(self) -> list[float]:
        """Sample discrete elements for evaluation."""
        return [k * 2.0 * np.pi / 8 for k in range(8)]

    def sample_uniform(self, num_samples: int = 1) -> list[float]:
        return list(np.random.uniform(0, 2 * np.pi, size=num_samples))

    def identity(self) -> float:
        return 0.0

    def compose(self, g1: float, g2: float) -> float:
        return float((g1 + g2) % (2.0 * np.pi))

    def inverse(self, g: float) -> float:
        return float((-g) % (2.0 * np.pi))

    def action_matrix(self, g: float, dim: int) -> torch.Tensor:
        cos_t = float(np.cos(g))
        sin_t = float(np.sin(g))
        mat = torch.eye(dim)
        for i in range(0, dim - 1, 2):
            mat[i, i] = cos_t
            mat[i, i + 1] = -sin_t
            mat[i + 1, i] = sin_t
            mat[i + 1, i + 1] = cos_t
        return mat


def compute_orbit(group: Group, x: torch.Tensor) -> list[torch.Tensor]:
    """Compute the orbit G * x = { rho(g) x : g in G } for a finite group or sampled elements."""
    elems = group.elements()
    return [group.act(g, x) for g in elems]


def check_invariance(
    fn: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
    group: Group,
    x: torch.Tensor,
    elements: Sequence[Any] | None = None,
) -> tuple[float, bool]:
    r"""Compute the invariance error:

        E_inv = \frac{1}{|G|} \sum_{g \in G} \| fn(rho(g) x) - fn(x) \|_2

    Returns:
        (mean_error, is_invariant_bool)
    """
    if elements is None:
        elements = group.elements()

    fx = fn(x)
    errors = []
    for g in elements:
        gx = group.act(g, x)
        fgx = fn(gx)
        diff = torch.norm(fgx - fx, dim=-1).mean().item()
        errors.append(diff)

    mean_err = float(np.mean(errors))
    return mean_err, mean_err < 1e-5


def check_equivariance(
    fn: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
    group: Group,
    x: torch.Tensor,
    output_group: Group | None = None,
    elements: Sequence[Any] | None = None,
) -> tuple[float, bool]:
    r"""Compute the equivariance error:

        E_equiv = \frac{1}{|G|} \sum_{g \in G} \| fn(rho_in(g) x) - rho_out(g) fn(x) \|_2

    Returns:
        (mean_error, is_equivariant_bool)
    """
    if output_group is None:
        output_group = group
    if elements is None:
        elements = group.elements()

    fx = fn(x)
    errors = []
    for g in elements:
        gx = group.act(g, x)
        fgx = fn(gx)
        g_fx = output_group.act(g, fx)
        diff = torch.norm(fgx - g_fx, dim=-1).mean().item()
        errors.append(diff)

    mean_err = float(np.mean(errors))
    return mean_err, mean_err < 1e-5


def symmetry_error(
    fn: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
    group: Group,
    dataloader: Any,
    is_equivariant: bool = False,
    device: str = "cpu",
) -> float:
    """Compute average symmetry violation error over an entire dataloader."""
    total_error = 0.0
    total_samples = 0

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)

        batch_size = x.size(0)
        if is_equivariant:
            err, _ = check_equivariance(fn, group, x)
        else:
            err, _ = check_invariance(fn, group, x)

        total_error += err * batch_size
        total_samples += batch_size

    return float(total_error / max(1, total_samples))
