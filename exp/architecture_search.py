from train_test import train_only_AR

# architecture search for n
def parameter_search(A, B, C, W, V, L, H, T, 
                     epochs, batch_size, lr, step_size, gamma, lambda_reg,
                     train_seed, torch_seed, search_range):
    """
    Parameter search over n.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The n with the least validation loss.
    """
    best_loss = float("inf")
    best_n = None

    ns = search_range
    # ns = np.arange(start=max(i-r,1), stop=i+r+1)

    # ns = np.arange(start=1, stop=10 + r +1)
    losses_dict = {}

    for N in ns:
        print(f"doing n={N}")
        model, losses = train_only_AR(
            A,
            B,
            C,
            W,
            V,
            hidden_dim=N,
            L=L,
            H=H,
            T=T,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            step_size=step_size,
            gamma=gamma,
            lambda_reg=lambda_reg,
            train_seed=train_seed,
            torch_seed=torch_seed,
            dist_type='gaussian',
        )
        losses_dict[N] = losses

        if min(losses) < best_loss:
            best_loss = min(losses)
            best_n = N

    return best_n, losses_dict