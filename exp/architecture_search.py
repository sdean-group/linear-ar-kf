from train_test import train_only_AR

# architecture search for n
def parameter_search(A, B, C, W, V, L, H, T, 
                     epochs, batch_size, lr, step_size, gamma, weight_decay,
                     train_seed, torch_seed, search_range):
    best_loss = float("inf")
    best_n = None

    ns = search_range
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
            weight_decay=weight_decay,
            train_seed=train_seed,
            torch_seed=torch_seed
        )
        losses_dict[N] = losses

        if min(losses) < best_loss:
            best_loss = min(losses)
            best_n = N

    return best_n, losses_dict