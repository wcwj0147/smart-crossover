from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from smart_crossover.formats import StandardLP, GeneralLP
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Output
from smart_crossover.solver_caller.solving import solve_lp


def perturb_c(lp_ori: GeneralLP,
              x: np.ndarray) -> np.ndarray:
    """
    Perturb the input array `c` based on the interior-point solution `x`.

    Args:
        lp_ori: The original LP.
        x: An interior-point solution used to generate the perturbation.

    Returns:
        Perturbed cost vector.

    """
    SCALE_FACTOR_FOR_PERTURBATION = 1e-2
    PERTURB_THRESHOLD = 1e-5
    n = len(x)

    def compute_projector(A: sp.csr_matrix, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute the projector P_{X^{-1} L}(c) = (I − X A^⊤ (A X^2 A)^† A X) c"""
        X = sp.diags(x)
        X2 = sp.diags(x ** 2)

        AT = A.transpose()
        A_X = A @ X
        A_X2_AT = A @ X2 @ AT

        # Compute temp1 = A X c
        temp1 = A_X @ c

        # Solve the linear least-squares problem using lsqr
        solution = lsqr(A=A_X2_AT, b=temp1, atol=1e-6, btol=1e-6, iter_lim=1000, show=False)
        y = solution[0]

        # Compute the final result
        temp2 = AT @ y
        result = c - X @ temp2

        return result

    perturbation_vector = np.zeros_like(x, dtype=np.float64)
    x_min = np.minimum(x - lp_ori.l, lp_ori.u - x)
    # Transfer to standard form temporarily to calculate the projector.
    projector_extend = compute_projector(lp_ori.get_standard_A(), lp_ori.get_standard_var_vector(x_min), lp_ori.get_standard_c())
    projector = lp_ori.recover_standard_var_vector(projector_extend)

    # Compute the perturbation vector = 1 / (x_min * |projector| * SCALE_FACTOR_FOR_PERTURBATION * n)
    denominator = x_min * np.abs(projector) * SCALE_FACTOR_FOR_PERTURBATION * n
    perturbation_vector[denominator > PERTURB_THRESHOLD] = 1 / denominator[denominator > PERTURB_THRESHOLD]

    c_pt = lp_ori.c + perturbation_vector
    return c_pt


def get_perturb_problem(lp: GeneralLP,
                        perturb_method: str,
                        x: Optional[np.ndarray],
                        s: Optional[np.ndarray]) -> LPManager:
    """
    Find an approximated optimal face using the given interior-point solution. And get the subproblem
    restricted on that optimal face. Use the perturbed objective c_pt.

    Args:
        lp: the original LP problem in StandardLP format.
        perturb_method: the perturbation method (choose from 'random', 'primal', 'dual').
        x: the primal interior-point solution.
        s: the slack of the constraints in the dual LP.

    Returns:
        A LP manager of a sub-problem restricted on the approximated optimal face.

    """
    BETA = 1e-2
    c_perturbed = perturb_c(lp, x)
    subLP_manager = LPManager(GeneralLP(A=lp.A, b=lp.b, c=c_perturbed, l=lp.l, u=lp.u, sense=lp.sense))
    subLP_manager.fix_variables(ind_fix_to_low=np.where(x - lp.l < BETA * s)[0],
                                ind_fix_to_up=np.where(lp.u - x < BETA * -s)[0])
    subLP_manager.update_subproblem()
    return subLP_manager


def run_perturb_algorithm(lp: GeneralLP,
                          perturb_method: str,
                          solver: str = "GRB",
                          barrierTol: float = 1e-8,
                          optimalityTol: float = 1e-6) -> Output:
    """

    Args:
        lp:
        perturb_method:
        solver:
        barrierTol:
        optimalityTol:

    Returns:

    """
    barrier_output = solve_lp(lp, solver, method='barrier', barrierTol=barrierTol, presolve='off', crossover='off')

    def get_dual_slack(A: sp.csr_matrix, c: np.ndarray, y: np.ndarray) -> np.ndarray:
        return c - A.transpose() @ y

    perturbLP_manager = get_perturb_problem(lp, perturb_method, barrier_output.x, get_dual_slack(lp.A, lp.c, barrier_output.y))
    perturb_barrier_output = solve_lp(perturbLP_manager.lp_sub, method='barrier', solver=solver, barrierTol=barrierTol,
                                      presolve="on")
    final_output = solve_lp(lp, solver, presolve="off", method='simplex',
                            optimalityTol=optimalityTol,
                            warm_start_solution=(perturbLP_manager.recover_x_from_sub_x(perturb_barrier_output.x),
                                                 perturb_barrier_output.y))
    return Output(x=final_output.x,
                  y=final_output.y,
                  obj_val=final_output.obj_val,
                  runtime=barrier_output.runtime + perturb_barrier_output.runtime + final_output.runtime,
                  iter_count=perturb_barrier_output.iter_count + final_output.iter_count)
