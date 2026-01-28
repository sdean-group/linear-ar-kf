import numpy as np

def observability_matrix(A, C):
    """Stack [C; C A; C A^2; ...; C A^{n-1}]"""
    n = A.shape[0]
    blocks = []
    Ak = np.eye(n)
    for k in range(n):
        blocks.append(C @ Ak)
        Ak = Ak @ A
    return np.vstack(blocks)

def is_observable(A, C, tol=1e-10):
    Om = observability_matrix(A, C)
    rOm = np.linalg.matrix_rank(Om, tol=tol)
    return rOm == A.shape[0], rOm, Om

def controllability_matrix(A, B):
    """Stack [B; AB; A^2B; ...; A^{n-1}B]"""
    n = A.shape[0]
    blocks = []
    Ak = np.eye(n, dtype=A.dtype)  # A^0
    for _ in range(n):
        blocks.append(Ak @ B)
        Ak = Ak @ A

    return np.hstack(blocks)

def is_controllable(A, B, tol=1e-10):
    Cm = controllability_matrix(A, B)
    rCm = np.linalg.matrix_rank(Cm, tol=tol)
    return rCm == A.shape[0], rCm, Cm

def define_system(n,p,m,rho_target,seed):

    rng_sys = np.random.default_rng(seed)

    A_rand = rng_sys.standard_normal((n, n))
    rhoA = float(np.max(np.abs(np.linalg.eigvals(A_rand))))
    A = (rho_target / rhoA) * A_rand
    print("Spectral radius of A: ", float(np.max(np.abs(np.linalg.eigvals(A)))))

    B = rng_sys.standard_normal((n, p)) / np.sqrt(p)
    C = rng_sys.standard_normal((m, n)) / np.sqrt(m)

    # ---------- observability check ----------
    ok, rankOm, Om = is_observable(A, C)
    print(f"Observable? {ok} (rank {rankOm} of {n})")

    # --------- controllability check --------
    ok, rankCm, Cm = is_controllable(A, B)
    print(f"Controllable? {ok} (rank {rankCm} of {n})")

    return A, B, C