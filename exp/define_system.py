from __future__ import annotations

"""Utilities for constructing random linear dynamical systems."""

import numpy as np


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the spectral radius of a square matrix."""

    eigenvalues = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigenvalues)))


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Stack the observability blocks [C; C A; C A^2; ...; C A^{n-1}]."""

    n = A.shape[0]
    blocks = []
    A_power = np.eye(n, dtype=A.dtype)

    for _ in range(n):
        blocks.append(C @ A_power)
        A_power = A_power @ A

    return np.vstack(blocks)


def is_observable(
    A: np.ndarray,
    C: np.ndarray,
    tol: float = 1e-10,
) -> tuple[bool, int, np.ndarray]:
    """Check whether (A, C) is observable."""

    observability = observability_matrix(A, C)
    rank = np.linalg.matrix_rank(observability, tol=tol)
    return rank == A.shape[0], rank, observability


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Stack the controllability blocks [B, AB, A^2B, ..., A^{n-1}B]."""

    n = A.shape[0]
    blocks = []
    A_power = np.eye(n, dtype=A.dtype)

    for _ in range(n):
        blocks.append(A_power @ B)
        A_power = A_power @ A

    return np.hstack(blocks)


def is_controllable(
    A: np.ndarray,
    B: np.ndarray,
    tol: float = 1e-10,
) -> tuple[bool, int, np.ndarray]:
    """Check whether (A, B) is controllable."""

    controllability = controllability_matrix(A, B)
    rank = np.linalg.matrix_rank(controllability, tol=tol)
    return rank == A.shape[0], rank, controllability


def define_system(
    n: int,
    p: int,
    m: int,
    rho_target: float,
    seed: int,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a stable linear system with random input/output matrices."""

    rng_sys = np.random.default_rng(seed)

    A_rand = rng_sys.standard_normal((n, n))
    A = (rho_target / spectral_radius(A_rand)) * A_rand
    B = rng_sys.standard_normal((n, p)) / np.sqrt(p)
    C = rng_sys.standard_normal((m, n)) / np.sqrt(m)

    if verbose:
        print("Spectral radius of A: ", spectral_radius(A))

        is_obs, observability_rank, _ = is_observable(A, C)
        print(f"Observable? {is_obs} (rank {observability_rank} of {n})")

        is_ctrl, controllability_rank, _ = is_controllable(A, B)
        print(f"Controllable? {is_ctrl} (rank {controllability_rank} of {n})")

    return A, B, C
