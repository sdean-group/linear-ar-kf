from __future__ import annotations

"""Training and evaluation utilities for the linear AR experiments."""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

try:
    from .ARmodel import TwoLayerLinearAR
    from .dataloader import make_dataset
    from .simulate import simulate
    from .steady_state_KF import run_steady_state_kalman_filter
except ImportError:
    from ARmodel import TwoLayerLinearAR
    from dataloader import make_dataset
    from simulate import simulate
    from steady_state_KF import run_steady_state_kalman_filter


def _system_dimensions(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[int, int, int]:
    return A.shape[0], B.shape[1], C.shape[0]


def _simulate_kalman_trajectory(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    T: int,
    *,
    seed: int,
    u_mode: str | None = None,
    u_scale: float = 1.0,
    period: int = 50,
    dist_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a trajectory and align it with steady-state Kalman predictions."""

    n, _, _ = _system_dimensions(A, B, C)
    _, Y, U = simulate(
        A,
        B,
        C,
        W,
        V,
        T,
        seed=seed,
        u_mode=u_mode,
        u_scale=u_scale,
        period=period,
        dist_type=dist_type,
    )
    kf_pred_x, _ = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), U, Y)
    return Y, U, kf_pred_x


def _build_model(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    torch_seed: int,
) -> TwoLayerLinearAR:
    torch.manual_seed(torch_seed)
    return TwoLayerLinearAR(input_dim, hidden_dim, output_dim)


def _train_model(
    model: TwoLayerLinearAR,
    train_loader,
    *,
    epochs: int,
    lr: float,
    step_size: int,
    gamma: float,
    weight_decay: float,
    loss_reduction: str = "mean",
    verbose: bool = False,
) -> list[float]:
    criterion = nn.MSELoss(reduction=loss_reduction)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_inputs, batch_outputs in train_loader:
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss_reduction == "sum":
                total_loss += loss.item()
            else:
                total_loss += loss.item() * len(batch_inputs)

        average_loss = total_loss / len(train_loader.dataset)
        losses.append(float(average_loss))

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch + 1}/{epochs}], Training Loss: {average_loss:.4f}, "
                f"Learning Rate: {current_lr:.6f}"
            )

        scheduler.step()

    return losses


def _predict_states_from_features(
    model: TwoLayerLinearAR,
    inputs_np: np.ndarray,
    target_states_np: np.ndarray,
) -> tuple[np.ndarray, int]:
    feature_matrix = inputs_np @ model.linear1.weight.detach().cpu().numpy().T
    coeff, _, _, _ = np.linalg.lstsq(feature_matrix, target_states_np, rcond=None)
    predicted_states = feature_matrix @ coeff
    coeff_rank = int(np.linalg.matrix_rank(coeff))
    return predicted_states, coeff_rank


def train_test(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    L: int = 10,
    H: int = 10,
    T: int = 2000,
    testT: int = 1000,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.01,
    step_size: int = 1,
    gamma: float = 0.9,
    weight_decay: float = 1e-3,
    train_seed: int = 10,
    torch_seed: int = 1000,
    test_seed: int = 20,
    test_umode: str = "ramp_sine",
    dist_type: str = "gaussian",
) -> tuple[TwoLayerLinearAR, np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Train the AR model, regress learned features to KF states, and return both."""

    n, p, m = _system_dimensions(A, B, C)

    train_Y, train_U, train_kf_x = _simulate_kalman_trajectory(
        A,
        B,
        C,
        W,
        V,
        T + H,
        seed=train_seed,
        dist_type=dist_type,
    )
    train_loader, _, _, _ = make_dataset(
        L,
        H,
        train_Y,
        train_U,
        train_kf_x,
        batch_size=batch_size,
    )

    model = _build_model((m + p) * L, n, m * H, torch_seed)
    train_losses = _train_model(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        weight_decay=weight_decay,
        loss_reduction="sum",
        verbose=True,
    )

    model.eval()
    test_Y, test_U, test_kf_x = _simulate_kalman_trajectory(
        A,
        B,
        C,
        W,
        V,
        testT + H,
        seed=test_seed,
        u_mode=test_umode,
        u_scale=2.0,
        period=40,
        dist_type=dist_type,
    )
    _, test_inputs_np, _, test_kf_states_np = make_dataset(
        L,
        H,
        test_Y,
        test_U,
        test_kf_x,
        batch_size=batch_size,
    )

    predicted_states_from_regression, coeff_rank = _predict_states_from_features(
        model,
        test_inputs_np,
        test_kf_states_np,
    )
    print("coefficient rank: ", coeff_rank)

    return model, test_kf_states_np, predicted_states_from_regression, {"train": train_losses}


def train_and_eval_error(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    T: int,
    L: int,
    H: int,
    *,
    testT: int = 600,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.01,
    step_size: int = 10,
    gamma: float = 0.9,
    weight_decay: float = 1e-3,
    train_seed: int = 10,
    torch_seed: int = 1000,
    test_seed: int = 20,
    test_umode: str = "ramp_bias",
    dist_type: str = "gaussian",
) -> dict[str, Any]:
    """Train once and return the mean L2 distance to the Kalman states."""

    n, p, m = _system_dimensions(A, B, C)

    train_Y, train_U, train_kf_x = _simulate_kalman_trajectory(
        A,
        B,
        C,
        W,
        V,
        T + H,
        seed=train_seed,
        dist_type=dist_type,
    )
    train_loader, _, _, _ = make_dataset(
        L,
        H,
        train_Y,
        train_U,
        train_kf_x,
        batch_size=batch_size,
    )

    model = _build_model((m + p) * L, n, m * H, torch_seed)
    _train_model(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        weight_decay=weight_decay,
        loss_reduction="mean",
    )

    model.eval()
    test_Y, test_U, test_kf_x = _simulate_kalman_trajectory(
        A,
        B,
        C,
        W,
        V,
        testT + H,
        seed=test_seed,
        u_mode=test_umode,
        u_scale=2.0,
        dist_type=dist_type,
    )
    _, test_inputs_np, _, test_kf_states_np = make_dataset(
        L,
        H,
        test_Y,
        test_U,
        test_kf_x,
        batch_size=batch_size,
    )

    learned_states, _ = _predict_states_from_features(model, test_inputs_np, test_kf_states_np)
    diff = learned_states - test_kf_states_np
    mean_l2 = float(np.mean(np.linalg.norm(diff, axis=1)))

    return {
        "T": int(T),
        "mean_l2": mean_l2,
    }


def run_sweep(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    T_list,
    L: int,
    H: int,
    *,
    repeats: int = 5,
    base_train_seed: int = 10,
    base_torch_seed: int = 1000,
    **kwargs,
) -> pd.DataFrame:
    """Run repeated training sweeps over different trajectory lengths."""

    rows = []
    for T in T_list:
        mean_l2s = []
        for repeat_idx in range(repeats):
            result = train_and_eval_error(
                A,
                B,
                C,
                W,
                V,
                T,
                L,
                H,
                train_seed=base_train_seed + repeat_idx,
                torch_seed=base_torch_seed + repeat_idx,
                **kwargs,
            )
            mean_l2s.append(result["mean_l2"])

        mean_l2s = np.asarray(mean_l2s)
        rows.append(
            {
                "T": int(T),
                "mean_l2_mean": float(mean_l2s.mean()),
                "mean_l2_std": float(mean_l2s.std(ddof=1)) if repeats > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def train_only_AR(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    hidden_dim: int,
    L: int,
    H: int,
    T: int = 2000,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.01,
    step_size: int = 1,
    gamma: float = 0.9,
    weight_decay: float = 1e-3,
    train_seed: int = 10,
    torch_seed: int = 1000,
    dist_type: str = "gaussian",
) -> tuple[TwoLayerLinearAR, list[float]]:
    """Train only the AR model and return per-epoch losses."""

    _, p, m = _system_dimensions(A, B, C)

    train_Y, train_U, train_kf_x = _simulate_kalman_trajectory(
        A,
        B,
        C,
        W,
        V,
        T + H,
        seed=train_seed,
        dist_type=dist_type,
    )
    train_loader, _, _, _ = make_dataset(
        L,
        H,
        train_Y,
        train_U,
        train_kf_x,
        batch_size=batch_size,
    )

    model = _build_model((m + p) * L, hidden_dim, m * H, torch_seed)
    losses = _train_model(
        model,
        train_loader,
        epochs=epochs,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        weight_decay=weight_decay,
        loss_reduction="mean",
    )

    print(f"Training Loss: {min(losses):.4f}")
    return model, losses
