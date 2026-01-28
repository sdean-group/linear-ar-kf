import pandas as pd
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from simulate import simulate
from steady_state_KF import run_steady_state_kalman_filter
from dataloader import make_dataset
from ARmodel import TwoLayerLinearAR

def train_test(
        A,
        B,
        C, 
        W,
        V,
        L=10, 
        H=10, 
        T=2000, 
        testT=1000, 
        epochs=50,
        batch_size=64,
        lr=0.01, 
        step_size=1, 
        gamma=0.9, 
        weight_decay=1e-3, 
        train_seed=10, 
        torch_seed=1000, 
        test_seed=20, 
        test_umode='ramp_sine', 
        dist_type='gaussian'
    ):

    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]

    # -------- simulation for training data -------
    X, Y, U = simulate(A, B, C, W, V, T+H, seed=train_seed, dist_type=dist_type)

    # Run the steady-state Kalman filter for the trajectory
    true_kf_x, K = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), U, Y)

    # Make training dataset
    train_loader, _, _, _ = make_dataset(L, H, Y, U, true_kf_x, batch_size=batch_size)

    # Define two layer linear model

    torch.manual_seed(torch_seed)
    model = TwoLayerLinearAR((m+p)*L, n, m*H)

    # train model
    results = {"train":[]}

    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average = total_loss / len(train_loader.dataset)
        results["train"].append(average)

        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {average:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        scheduler.step()

    # test model
    model.eval()

    testX, testY, testU = simulate(A, B, C, W, V, testT+H, seed=test_seed, u_mode=test_umode, u_scale=2.0, period=40, dist_type=dist_type)
    test_kf_x, testK = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), testU, testY)
    _, test_ar_data_inputs_np, _, test_ar_data_kf_states_np = make_dataset(L, H, testY, testU, test_kf_x, batch_size=64)

    G1 = model.linear1.weight.detach().numpy()
    Z = test_ar_data_inputs_np @ G1.T

    coeff, _, rank, _ = np.linalg.lstsq(Z, test_ar_data_kf_states_np, rcond=None)
    predicted_states_from_regression = Z @ coeff

    print("regression rank: ", rank)

    return model, test_ar_data_kf_states_np, predicted_states_from_regression, results


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
):
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]

    # --- training trajectory ---
    _, Y, U = simulate(
        A, B, C, W, V, T+H, seed=train_seed,
        dist_type=dist_type
    )
    true_kf_x, _ = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), U, Y)

    train_loader, _, _, _ = make_dataset(L, H, Y, U, true_kf_x, batch_size=batch_size)

    # --- model training (predict future outputs) ---
    torch.manual_seed(torch_seed)
    model = TwoLayerLinearAR((m + p) * L, n, m * H)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.train()
    for _ in range(epochs):
        for inputs, targets in train_loader:
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    # --- test trajectory ---
    model.eval()
    _, testY, testU = simulate(
        A, B, C, W, V, testT+H, seed=test_seed, u_mode=test_umode, u_scale=2.0,
        dist_type=dist_type
    )
    test_kf_pred_x, _ = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), testU, testY)
    _, test_inputs_np, _, test_kf_states_np = make_dataset(L, H, testY, testU, test_kf_pred_x, batch_size=batch_size)

    # --- learned state: regress KF states onto learned features (linear1) ---
    G = model.linear1.weight.detach().numpy()   # (n, (m+p)*L)
    Z = test_inputs_np @ G.T                    # (N_test, n)

    coeff, _, rank, _ = np.linalg.lstsq(Z, test_kf_states_np, rcond=None)
    learned_states = Z @ coeff                  # (N_test, n)

    diff = learned_states - test_kf_states_np
    mean_l2 = float(np.mean(np.linalg.norm(diff, axis=1)))   # mean Euclidean distance
    # rmse = float(np.sqrt(np.mean(diff**2)))                  # coordinate-wise RMSE

    return {
        "T": int(T),
        "mean_l2": mean_l2,
        # "rmse": rmse,
        # "rank": int(rank),
    }

def run_sweep(
        A,
        B,
        C,
        W,
        V,
        T_list,
        L,
        H,
        *,
        repeats: int = 5,
        base_train_seed: int = 10,
        base_torch_seed: int = 1000,
        **kwargs
):
    rows = []
    for T in T_list:
        mean_l2s = []
        for r in range(repeats):
            result = train_and_eval_error(
                A, B, C, W, V, T, L, H,
                train_seed=base_train_seed + r,
                torch_seed=base_torch_seed + r,
                **kwargs
            )
            mean_l2s.append(result["mean_l2"])
        
        mean_l2s = np.asarray(mean_l2s)
        rows.append({
            "T": int(T),
            "mean_l2_mean": float(mean_l2s.mean()),
            "mean_l2_std": float(mean_l2s.std(ddof=1)) if repeats > 1 else 0.0
        })
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

def train_only_AR(
        A,
        B,
        C,
        W,
        V,
        hidden_dim,
        L,
        H,
        T=2000,
        epochs=50,
        batch_size=64,
        lr=0.01,
        step_size=1,
        gamma=0.9,
        weight_decay=1e-3,
        train_seed=10,
        torch_seed=1000,
        dist_type='gaussian'
    ):
    
    n = A.shape[0]
    p = B.shape[1]
    m = C.shape[0]

    # -------- simulation for training data -------
    X, Y, U = simulate(A, B, C, W, V, T+H, seed=train_seed, dist_type=dist_type)

    # Run the steady-state Kalman filter for the trajectory
    true_kf_x, K = run_steady_state_kalman_filter(A, B, C, W, V, np.zeros(n), U, Y)

    # Make training dataset
    train_loader, _, _, _ = make_dataset(L, H, Y, U, true_kf_x, batch_size=batch_size)

    # Define two layer linear model
    torch.manual_seed(torch_seed)
    model = TwoLayerLinearAR((m+p)*L, hidden_dim, m*H)

    # train model
    # results = {"train":[], "val":[]}
    losses = []

    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            outputs, _ = model(batch_inputs)
            loss = criterion(outputs, batch_outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)

        average = total_loss / len(train_loader.dataset)
        losses.append(average)

        scheduler.step()
    
    print(f'Training Loss: {min(losses):.4f}')

    return model, losses