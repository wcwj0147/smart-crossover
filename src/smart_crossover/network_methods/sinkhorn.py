import numpy as np

from smart_crossover.formats import OptTransport


def sinkhorn(ot_instance: OptTransport, reg: float, num_iter: int = 1000, tol: float = 1e-9) -> np.ndarray:
    s, d, M = ot_instance.s, ot_instance.d, ot_instance.M
    n, m = M.shape

    # Exponential of the negative cost matrix divided by the regularization term
    K = np.exp(-M / reg)

    # Initialize the scaling factors
    u = np.ones(n)
    v = np.ones(m)

    # Sinkhorn iterations
    for _ in range(num_iter):
        u_old = u.copy()
        u = s / (K @ v)
        v = d / (K.T @ u)

        # Check for convergence
        if np.linalg.norm(u - u_old) < tol:
            break

    # Compute the optimal transport plan
    P = np.diag(u) @ K @ np.diag(v)

    return P
