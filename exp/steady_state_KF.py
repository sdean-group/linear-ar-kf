from __future__ import annotations

"""Steady-state Kalman filter utilities."""

import numpy as np
from scipy.linalg import solve_discrete_are


def steady_state_kalman_gain(
    A: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the steady-state estimation Riccati equation and return (K, P, S)."""

    A = np.asarray(A, float)
    C = np.asarray(C, float)
    W = np.asarray(W, float)
    V = np.asarray(V, float)

    P = solve_discrete_are(A.T, C.T, W, V)
    S = C @ P @ C.T + V
    K = P @ C.T @ np.linalg.inv(S)
    return K, P, S


def run_steady_state_kalman_filter(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    initial_state_mean: np.ndarray,
    control_inputs: np.ndarray,
    measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted state means aligned with the trajectory and the steady-state gain."""

    K, _, _ = steady_state_kalman_gain(A, C, W, V)

    x = initial_state_mean.copy()
    predicted_means = [x]

    for u_t, y_t in zip(control_inputs, measurements):
        innovation = y_t - C @ x
        x_update = x + K @ innovation
        x = A @ x_update + B @ u_t
        predicted_means.append(x)

    return np.asarray(predicted_means)[:-1], K
