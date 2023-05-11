from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from smart_crossover.formats import GeneralLP
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Output, Basis
from smart_crossover.solver_caller.caller import SolverSettings
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
    PERTURB_THRESHOLD = 1e-3
    n = len(x)

    def apply_projector(Y, c, tol=1e-8, max_iter=1000):
        """ Apply the projection matrix Y^T (Y Y^T)† Y to a vector c using conjugate gradient method. """
        Yc = Y @ c
        z, _ = splinalg.cg(Y @ Y.T, Yc, tol=tol, maxiter=max_iter)
        return Y.T @ z

    perturbation_vector = np.zeros_like(x, dtype=np.float64)
    x_min = np.minimum(x - lp_ori.l, lp_ori.u - x)
    # Transfer to standard form temporarily to calculate the projector.
    projector = apply_projector(lp_ori.A @ sp.diags(x_min), lp_ori.c)

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
                          optimalityTol: float = 1e-6,
                          log_file_head: str = '') -> Output:
    """

    Args:
        lp:
        perturb_method:
        solver:
        barrierTol:
        optimalityTol:
        log_file:

    Returns:

    """
    barrier_output = solve_lp(lp, solver,
                              method='barrier',
                              settings=SolverSettings(barrierTol=barrierTol, presolve='off', crossover='off', log_file=log_file_head+'_ori_bar.log'))

    def get_dual_slack(A: sp.csr_matrix, c: np.ndarray, y: np.ndarray) -> np.ndarray:
        return c - A.transpose() @ y

    perturbLP_manager = get_perturb_problem(lp, perturb_method, barrier_output.x, get_dual_slack(lp.A, lp.c, barrier_output.y))
    perturb_barrier_output = solve_lp(perturbLP_manager.lp_sub, solver=solver,
                                      method='barrier',
                                      settings=SolverSettings(barrierTol=barrierTol, presolve="on", log_file=log_file_head+'_ptb_bar.log'))
    final_output = solve_lp(lp, solver=solver,
                            method='primal_simplex',
                            settings=SolverSettings(presolve="off", optimalityTol=optimalityTol, log_file=log_file_head+'_final.log'),
                            warm_start_solution=(perturbLP_manager.recover_x_from_sub_x(perturb_barrier_output.x),
                                                 perturb_barrier_output.y),
                            warm_start_basis=perturbLP_manager.recover_basis_from_sub_basis(perturb_barrier_output.basis))
    return Output(x=final_output.x,
                  y=final_output.y,
                  obj_val=final_output.obj_val,
                  runtime=barrier_output.runtime + perturb_barrier_output.runtime + final_output.runtime,
                  iter_count=perturb_barrier_output.iter_count + final_output.iter_count)
