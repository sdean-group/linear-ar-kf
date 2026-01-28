import numpy as np
from steady_state_KF import steady_state_kalman_gain

def _cov_sqrt(S):
    """Return symmetric square-root R such that R @ R.T ≈ S."""
    S = np.asarray(S, dtype=float)
    if S.ndim == 0:
        return np.array([[np.sqrt(S)]])
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def simulate(A, B, C, W, V, T, seed=None, 
             u_mode=None, u_scale=1.0, period=50,
             dist_type='gaussian'):

    rng = np.random.default_rng(seed)

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    n, _ = A.shape
    nB, p = B.shape
    m, nC = C.shape
    assert nB == n and nC == n, "Dimension mismatch"

    # Initialize x0
    _, P, _ = steady_state_kalman_gain(A, C, W, V)
    RP = _cov_sqrt(P)
    x = RP @ rng.standard_normal(n)

    # Square roots of covariances
    RW = _cov_sqrt(W)
    RV = _cov_sqrt(V)

    # Allocate
    X = np.zeros((T+1, n))
    Y = np.zeros((T+1, m))
    U = np.zeros((T, p))

    X[0] = x

    # Initial measurement noise
    if dist_type == 'uniform':
        v0 = RV @ (rng.uniform(-np.sqrt(3), np.sqrt(3), size=m) * np.sqrt(V.diagonal()) / np.sqrt(RV.diagonal())) # Scale for unit variance uniform noise
    elif dist_type == 'rademacher':
        v0 = RV @ (rng.choice([-1, 1], size=m) * np.sqrt(V.diagonal()) / np.sqrt(RV.diagonal())) # Scale for unit variance rademacher noise
    else: # Default to gaussian
        v0 = RV @ rng.standard_normal(m)
    Y[0] = C @ x + v0

    # Simulation loop
    for t in range(T):
        if dist_type == 'uniform':
            w = RW @ (rng.uniform(-np.sqrt(3), np.sqrt(3), size=n) * np.sqrt(W.diagonal()) / np.sqrt(RW.diagonal())) # Scale for unit variance uniform noise
            v = RV @ (rng.uniform(-np.sqrt(3), np.sqrt(3), size=m) * np.sqrt(V.diagonal()) / np.sqrt(RV.diagonal())) # Scale for unit variance uniform noise
        elif dist_type == 'rademacher':
            w = RW @ (rng.choice([-1, 1], size=n) * np.sqrt(W.diagonal()) / np.sqrt(RW.diagonal())) # Scale for unit variance rademacher noise
            v = RV @ (rng.choice([-1, 1], size=m) * np.sqrt(V.diagonal()) / np.sqrt(RV.diagonal())) # Scale for unit variance rademacher noise
        else: # Default to gaussian
            w = RW @ rng.standard_normal(n)
            v = RV @ rng.standard_normal(m)

        if u_mode == None:
            # Random input
            u = rng.standard_normal(p) / np.sqrt(p)

        elif u_mode == "ramp_bias":
            amp = u_scale * (t+1) / T
            u = amp * np.ones(p)

        elif u_mode == "ramp_sine":
            amp = u_scale * (t+1) / T
            u = amp * np.sin(2*np.pi * t / period) * np.ones(p)

        x = A @ x + B @ u + w
        y = C @ x + v

        U[t] = u
        X[t+1] = x
        Y[t+1] = y

    return X, Y, U