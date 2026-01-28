import numpy as np
from scipy.linalg import solve_discrete_are

# ---------- steady-state Kalman gain ----------
def steady_state_kalman_gain(A, C, W, V):
    """
    Solve the *estimation* DARE for P_inf:

        P = A P A^T + W - A P C^T (C P C^T + V)^{-1} C P A^T
    Using scipy's solve_discrete_are on the transposed (A^T, C^T, W, V).
    """
    A = np.asarray(A, float); C = np.asarray(C, float)
    W = np.asarray(W, float); V = np.asarray(V, float)

    # Solve estimation DARE via control DARE on (A^T, C^T, W, V)
    P = solve_discrete_are(A.T, C.T, W, V)
    S = C @ P @ C.T + V
    K = P @ C.T @ np.linalg.inv(S)
    return K, P, S

def run_steady_state_kalman_filter(
    A, B, C, W, V, initial_state_mean, control_inputs, measurements
):
    """
    Runs the steady-state Kalman filter.
    initial_state_mean: hat{x}_{0|-1}
    Returns:
        estimated_state_means, predicted_state_means,
    """
    K, P, S = steady_state_kalman_gain(A, C, W, V)

    x = initial_state_mean.copy()
    pred_means = []
    pred_means.append(x)

    for u_t, y_t in zip(control_inputs, measurements):
        # Innovation
        innov = y_t - C @ x

        # Update with steady-state Kalman gain
        x_update = x + K @ innov

        # Prediction
        x = A @ x_update + B @ u_t

        pred_means.append(x)

    return np.array(pred_means)[:-1], K