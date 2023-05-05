from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from smart_crossover.formats import StandardLP
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Output
from smart_crossover.solver_caller.solving import solve_lp


def perturb_c(lp_ori: StandardLP,
              x: np.ndarray) -> np.ndarray:
    """
    Perturb the input array `c` based on the interior-point solution `x`.

    Args:
        lp_ori: The original LP.
        x: An interior-point solution used to generate the perturbation.

    Returns:
        Perturbed cost vector.

    """
    SCALE_FACTOR_FOR_PERTURBATION = 1e-3
    PERTURB_THRESHOLD = 1e-8
    n = len(x)

    def get_perturbation_vector() -> np.ndarray:
        perturbation_vector = np.zeros_like(x, dtype=np.float64)
        x_min = np.minimum(x, lp_ori.u - x)

        # Compute the projector P_{X_k^{-1} L}(c) = (I − X_k A^⊤ ^† A X_k) c
        I = sp.eye(n)
        X_k = sp.diags(x_min)

        A_X_k = lp_ori.A @ X_k
        A_X_k2_A_T = A_X_k @ A_X_k.T

        # Compute the Moore-Penrose inverse (A X_k^2 A^⊤)^† using an iterative method for sparse matrices
        # Compute the SVD using scipy.sparse.svds
        NUM_SINGULAR_VALUES = 10  # Number of singular values and vectors to compute
        U, s, Vt = svds(A_X_k2_A_T, k=NUM_SINGULAR_VALUES)

        # Compute the approximate Moore-Penrose inverse using the SVD
        S_inv = np.diag(1 / s)
        A_X_k2_A_T_inv_approx = Vt.T @ S_inv @ U.T

        projector = (I - X_k @ lp_ori.A.T @ A_X_k2_A_T_inv_approx @ A_X_k) @ lp_ori.c

        # Compute the perturbation vector = 1 / (x_min * |projector| * SCALE_FACTOR_FOR_PERTURBATION * n)
        perturbation_vector[x_min >= PERTURB_THRESHOLD] = 1 / (x_min[x_min >= PERTURB_THRESHOLD]
                                                               * np.abs(projector[x_min >= PERTURB_THRESHOLD])
                                                               * SCALE_FACTOR_FOR_PERTURBATION * n)

        return perturbation_vector

    perturbation = get_perturbation_vector()
    c_pt = lp_ori.c + perturbation
    return c_pt


def get_perturb_problem(lp: StandardLP,
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
    subLP_manager = LPManager(StandardLP(A=lp.A, b=lp.b, c=c_perturbed, l=lp.l, u=lp.u))
    subLP_manager.fix_variables(ind_fix_to_low=np.where(x < BETA * s)[0],
                                ind_fix_to_up=np.where(lp.u - x < BETA * -s)[0])
    subLP_manager.update_subproblem()
    return subLP_manager


def run_perturb_algorithm(lp: StandardLP,
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

    def get_dual_slack() -> np.ndarray:
        return lp.c - lp.A.transpose() @ barrier_output.y

    perturbLP_manager = get_perturb_problem(lp, perturb_method, barrier_output.x, get_dual_slack())
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
