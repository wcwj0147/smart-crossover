import numpy as np

from smart_crossover.formats import OptTransport
from smart_crossover.output import Output
from smart_crossover.timer import Timer


def sinkhorn(ot_instance: OptTransport, reg: float = 0.2, num_iter: int = 1000, tol: float = 1e-9) -> Output:
    """
    Compute the optimal transport plan using the Sinkhorn algorithm.

    Args:
        ot_instance: the optimal transport problem.
        reg: the regularization term.
        num_iter: the maximum number of iterations.
        tol: the tolerance for convergence.

    Returns:
        the output, including the optimal transport plan and the runtime.

    """
    timer = Timer()
    timer.start_timer()

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

    timer.end_timer()
    return Output(x=P.ravel(), runtime=timer.total_duration)
