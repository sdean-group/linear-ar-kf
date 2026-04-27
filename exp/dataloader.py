from __future__ import annotations

"""Dataset helpers for autoregressive training windows."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_autoregressive_arrays(
    L: int,
    H: int,
    Y: np.ndarray,
    U: np.ndarray,
    kf_pred_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build numpy arrays for AR inputs, output targets, and Kalman states."""

    ar_data_inputs = []
    ar_data_outputs = []
    ar_data_kf_states = []

    T = Y.shape[0] - 1

    for t in range(L, T - max(L, H)):
        input_features = np.concatenate(
            [
                Y[t - L : t][::-1].flatten(order="C"),
                U[t - L : t][::-1].flatten(order="C"),
            ]
        )
        ar_data_inputs.append(input_features)
        ar_data_outputs.append(Y[t : t + H, :].flatten(order="C"))
        ar_data_kf_states.append(kf_pred_x[t, :].flatten(order="C"))

    return (
        np.asarray(ar_data_inputs),
        np.asarray(ar_data_outputs),
        np.asarray(ar_data_kf_states),
    )


def make_dataset(
    L: int,
    H: int,
    Y: np.ndarray,
    U: np.ndarray,
    KFpredX: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tuple[DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """Create a PyTorch dataloader plus the underlying numpy arrays."""

    ar_data_inputs_np, ar_data_outputs_np, ar_data_kf_states_np = build_autoregressive_arrays(
        L,
        H,
        Y,
        U,
        KFpredX,
    )

    ar_inputs_tensor = torch.tensor(ar_data_inputs_np, dtype=torch.float32)
    ar_outputs_tensor = torch.tensor(ar_data_outputs_np, dtype=torch.float32)
    dataset = TensorDataset(ar_inputs_tensor, ar_outputs_tensor)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, ar_data_inputs_np, ar_data_outputs_np, ar_data_kf_states_np
