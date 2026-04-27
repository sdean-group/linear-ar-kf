from __future__ import annotations

"""Utilities for hidden-dimension search."""

try:
    from .train_test import train_only_AR
except ImportError:
    from train_test import train_only_AR


def parameter_search(
    A,
    B,
    C,
    W,
    V,
    L,
    H,
    T,
    epochs,
    batch_size,
    lr,
    step_size,
    gamma,
    weight_decay,
    train_seed,
    torch_seed,
    search_range,
):
    """Search over hidden dimensions and keep the configuration with the best loss."""

    best_loss = float("inf")
    best_n = None
    losses_dict = {}

    for hidden_dim in search_range:
        print(f"doing n={hidden_dim}")
        _, losses = train_only_AR(
            A,
            B,
            C,
            W,
            V,
            hidden_dim=hidden_dim,
            L=L,
            H=H,
            T=T,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            weight_decay=weight_decay,
            train_seed=train_seed,
            torch_seed=torch_seed,
        )
        losses_dict[hidden_dim] = losses

        current_best = min(losses)
        if current_best < best_loss:
            best_loss = current_best
            best_n = hidden_dim

    return best_n, losses_dict
