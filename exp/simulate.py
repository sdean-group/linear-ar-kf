from __future__ import annotations

"""Simulation utilities for the linear dynamical systems experiments."""

import numpy as np

try:
    from .steady_state_KF import steady_state_kalman_gain
except ImportError:
    from steady_state_KF import steady_state_kalman_gain


def _cov_sqrt(S: np.ndarray) -> np.ndarray:
    """Return a symmetric square root R such that R @ R.T ~= S."""

    S = np.asarray(S, dtype=float)
    if S.ndim == 0:
        return np.array([[np.sqrt(S)]])

    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _draw_unit_variance_noise(
    rng: np.random.Generator,
    size: int,
    dist_type: str,
) -> np.ndarray:
    if dist_type == "gaussian":
        return rng.standard_normal(size)
    if dist_type == "uniform":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size=size)
    if dist_type == "rademacher":
        return rng.choice([-1.0, 1.0], size=size)
    raise ValueError(f"Unsupported dist_type: {dist_type}")


def _sample_correlated_noise(
    rng: np.random.Generator,
    covariance_sqrt: np.ndarray,
    covariance: np.ndarray,
    dist_type: str,
) -> np.ndarray:
    """Sample noise with the same scaling convention used in the original notebooks."""

    size = covariance.shape[0]
    base_noise = _draw_unit_variance_noise(rng, size=size, dist_type=dist_type)

    if dist_type == "gaussian":
        return covariance_sqrt @ base_noise

    covariance_diag = np.sqrt(np.diag(covariance))
    covariance_sqrt_diag = np.sqrt(np.clip(np.diag(covariance_sqrt), 1e-12, None))
    scaled_noise = base_noise * covariance_diag / covariance_sqrt_diag
    return covariance_sqrt @ scaled_noise


def _sample_control(
    rng: np.random.Generator,
    p: int,
    t: int,
    T: int,
    u_mode: str | None,
    u_scale: float,
    period: int,
) -> np.ndarray:
    if u_mode is None:
        return rng.standard_normal(p) / np.sqrt(p)

    amplitude = u_scale * (t + 1) / T

    if u_mode == "ramp_bias":
        return amplitude * np.ones(p)
    if u_mode == "ramp_sine":
        return amplitude * np.sin(2 * np.pi * t / period) * np.ones(p)

    raise ValueError(f"Unsupported u_mode: {u_mode}")


def simulate(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    T: int,
    seed: int | None = None,
    u_mode: str | None = None,
    u_scale: float = 1.0,
    period: int = 50,
    dist_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate states, observations, and controls for a linear dynamical system."""

    rng = np.random.default_rng(seed)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    W = np.asarray(W, dtype=float)
    V = np.asarray(V, dtype=float)

    n, _ = A.shape
    nB, p = B.shape
    m, nC = C.shape
    if nB != n or nC != n:
        raise ValueError("Dimension mismatch")

    _, state_covariance, _ = steady_state_kalman_gain(A, C, W, V)
    state_covariance_sqrt = _cov_sqrt(state_covariance)
    process_covariance_sqrt = _cov_sqrt(W)
    measurement_covariance_sqrt = _cov_sqrt(V)

    x = state_covariance_sqrt @ rng.standard_normal(n)

    X = np.zeros((T + 1, n))
    Y = np.zeros((T + 1, m))
    U = np.zeros((T, p))
    X[0] = x

    initial_noise = _sample_correlated_noise(rng, measurement_covariance_sqrt, V, dist_type)
    Y[0] = C @ x + initial_noise

    for t in range(T):
        process_noise = _sample_correlated_noise(rng, process_covariance_sqrt, W, dist_type)
        measurement_noise = _sample_correlated_noise(
            rng,
            measurement_covariance_sqrt,
            V,
            dist_type,
        )
        control = _sample_control(rng, p, t, T, u_mode, u_scale, period)

        x = A @ x + B @ control + process_noise
        y = C @ x + measurement_noise

        U[t] = control
        X[t + 1] = x
        Y[t + 1] = y

    return X, Y, U
