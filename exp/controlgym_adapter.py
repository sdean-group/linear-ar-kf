from __future__ import annotations

"""Utilities for importing linear systems from controlgym."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from .define_system import is_controllable, is_observable, spectral_radius
except ImportError:
    from define_system import is_controllable, is_observable, spectral_radius


CONTROLGYM_LINEAR_SYSTEM_IDS = (
    "toy",
    "ac1",
    "ac2",
    "ac3",
    "ac4",
    "ac5",
    "ac6",
    "ac7",
    "ac8",
    "ac9",
    "ac10",
    "bdt1",
    "bdt2",
    "cbm",
    "cdp",
    "cm1",
    "cm2",
    "cm3",
    "cm4",
    "cm5",
    "dis1",
    "dis2",
    "dlr",
    "he1",
    "he2",
    "he3",
    "he4",
    "he5",
    "he6",
    "iss",
    "je1",
    "je2",
    "lah",
    "pas",
    "psm",
    "rea",
    "umv",
)


@dataclass(frozen=True)
class ControlGymLinearSystem:
    """Discrete-time linear system extracted from a controlgym linear environment."""

    env_id: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    W: np.ndarray
    V: np.ndarray
    N: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    D21: np.ndarray
    noise_cov: float
    init_state: np.ndarray
    random_init_state_cov: float


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


def load_controlgym_linear_system(
    env_id: str,
    *,
    sample_time: float = 0.1,
    noise_cov: float = 0.1,
    random_init_state_cov: float = 0.1,
    seed: int | None = None,
    **kwargs,
) -> ControlGymLinearSystem:
    """Load a controlgym linear system and convert it to this repo's matrix format.

    The underlying controlgym linear environments evolve according to

        x_{t+1} = A x_t + B1 d_t + B2 u_t
        y_t     = C x_t + D21 d_t

    where d_t ~ N(0, noise_cov * I). This helper exposes:

    - B = B2
    - W = noise_cov * B1 B1^T
    - V = noise_cov * D21 D21^T
    - N = noise_cov * B1 D21^T

    Here N is the process/measurement cross-covariance, which matters because
    controlgym uses the same disturbance in both equations.
    """

    if env_id not in CONTROLGYM_LINEAR_SYSTEM_IDS:
        valid_ids = ", ".join(CONTROLGYM_LINEAR_SYSTEM_IDS)
        raise ValueError(
            f"Unsupported controlgym linear env_id '{env_id}'. "
            f"Valid linear ids: {valid_ids}"
        )

    try:
        from controlgym.envs import make as make_controlgym
    except ImportError as exc:
        raise ImportError(
            "controlgym is not installed. Install it first, then retry loading "
            "a controlgym-backed system."
        ) from exc

    env = make_controlgym(
        id=env_id,
        sample_time=sample_time,
        noise_cov=noise_cov,
        random_init_state_cov=random_init_state_cov,
        seed=seed,
        **kwargs,
    )

    if getattr(env, "category", None) != "linear":
        raise ValueError(
            f"controlgym env '{env_id}' is not a linear-control environment and "
            "does not match this repo's finite-dimensional LDS assumptions."
        )

    A = np.asarray(env.A, dtype=float)
    B1 = np.asarray(env.B1, dtype=float)
    B2 = np.asarray(env.B2, dtype=float)
    C = np.asarray(env.C, dtype=float)
    D21 = np.asarray(env.D21, dtype=float)

    disturbance_covariance = noise_cov * np.eye(B1.shape[1], dtype=float)
    W = B1 @ disturbance_covariance @ B1.T
    V = D21 @ disturbance_covariance @ D21.T
    N = B1 @ disturbance_covariance @ D21.T

    return ControlGymLinearSystem(
        env_id=env_id,
        A=A,
        B=B2,
        C=C,
        W=W,
        V=V,
        N=N,
        B1=B1,
        B2=B2,
        D21=D21,
        noise_cov=float(noise_cov),
        init_state=np.asarray(env.init_state, dtype=float).reshape(A.shape[0]),
        random_init_state_cov=float(random_init_state_cov),
    )


def simulate_controlgym_linear_system(
    system: ControlGymLinearSystem,
    T: int,
    *,
    seed: int | None = None,
    u_mode: str | None = None,
    u_scale: float = 1.0,
    period: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate trajectories with the same disturbance coupling used by controlgym."""

    rng = np.random.default_rng(seed)

    A = np.asarray(system.A, dtype=float)
    B1 = np.asarray(system.B1, dtype=float)
    B2 = np.asarray(system.B2, dtype=float)
    C = np.asarray(system.C, dtype=float)
    D21 = np.asarray(system.D21, dtype=float)

    n = A.shape[0]
    p = B2.shape[1]
    m = C.shape[0]
    q = B1.shape[1]

    disturbance_covariance = system.noise_cov * np.eye(q, dtype=float)
    init_covariance = system.random_init_state_cov * np.eye(n, dtype=float)

    x = rng.multivariate_normal(system.init_state, init_covariance)

    X = np.zeros((T + 1, n))
    Y = np.zeros((T + 1, m))
    U = np.zeros((T, p))
    X[0] = x

    for t in range(T):
        disturbance = rng.multivariate_normal(np.zeros(q), disturbance_covariance)
        control = _sample_control(rng, p, t, T, u_mode, u_scale, period)

        Y[t] = C @ x + D21 @ disturbance
        U[t] = control

        x = A @ x + B1 @ disturbance + B2 @ control
        X[t + 1] = x

    terminal_disturbance = rng.multivariate_normal(np.zeros(q), disturbance_covariance)
    Y[T] = C @ x + D21 @ terminal_disturbance

    return X, Y, U


def screen_controlgym_linear_systems(
    env_ids: tuple[str, ...] | list[str] | None = None,
    **load_kwargs,
) -> pd.DataFrame:
    """Summarize controlgym linear systems to help choose candidate environments."""

    if env_ids is None:
        env_ids = list(CONTROLGYM_LINEAR_SYSTEM_IDS)

    rows = []
    for env_id in env_ids:
        system = load_controlgym_linear_system(env_id, **load_kwargs)
        A, B, C = system.A, system.B, system.C

        is_obs, observability_rank, _ = is_observable(A, C)
        is_ctrl, controllability_rank, _ = is_controllable(A, B)

        rows.append(
            {
                "env_id": env_id,
                "n": int(A.shape[0]),
                "p": int(B.shape[1]),
                "m": int(C.shape[0]),
                "disturbance_dim": int(system.B1.shape[1]),
                "spectral_radius": float(spectral_radius(A)),
                "is_observable": bool(is_obs),
                "observability_rank": int(observability_rank),
                "is_controllable": bool(is_ctrl),
                "controllability_rank": int(controllability_rank),
                "rank_W": int(np.linalg.matrix_rank(system.W)),
                "rank_V": int(np.linalg.matrix_rank(system.V)),
                "cross_cov_norm": float(np.linalg.norm(system.N, ord="fro")),
            }
        )

    return pd.DataFrame(rows).sort_values(["n", "m", "p", "env_id"]).reset_index(drop=True)
