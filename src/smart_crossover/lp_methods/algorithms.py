import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from smart_crossover.formats import GeneralLP
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Output
from smart_crossover.parameters import OPTIMAL_FACE_ESTIMATOR, OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO, PERTURB_THRESHOLD, \
    CONSTANT_SCALE_FACTOR, PRIMAL_DUAL_GAP_THRESHOLD
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.solving import solve_lp


def run_perturb_algorithm(lp: GeneralLP,
                          solver: str = "GRB",
                          barrierTol: float = 1e-8,
                          optimalityTol: float = 1e-6,
                          log_file: str = '') -> Output:
    """

    Args:
        lp: the original LP problem in GeneralLP format.
        solver: the solver used to solve the LP.
        barrierTol: the barrier tolerance.
        optimalityTol: the optimality tolerance.
        log_file: the log file name.

    Returns:
         the output of the perturbed algorithm.

    """
    barrier_output = solve_lp(lp, solver,
                              method='barrier',
                              settings=SolverSettings(barrierTol=barrierTol, presolve='on', crossover='off', log_file=log_file))

    gamma = OPTIMAL_FACE_ESTIMATOR
    while True:

        perturbLP_manager = get_perturb_problem(lp, barrier_output.x, barrier_output.y, gamma)

        perturb_barrier_output = solve_lp(perturbLP_manager.lp_sub, solver=solver,
                                          method='barrier',
                                          settings=SolverSettings(presolve="on", log_file=log_file))

        if perturb_barrier_output.status == 'INFEASIBLE' or perturb_barrier_output.status =='UNBOUNDED':
            gamma = gamma * OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO
        else:
            break

    check_perturb_output_precision(perturbLP_manager, perturb_barrier_output.x, lp.c, barrier_output.obj_val)

    final_output = solve_lp(lp, solver=solver,
                            method='simplex' if solver == 'MSK' else 'primal_simplex',
                            settings=SolverSettings(presolve="on", optimalityTol=optimalityTol, log_file=log_file),
                            warm_start_solution=(perturbLP_manager.recover_x_from_sub_x(perturb_barrier_output.x),
                                                 perturb_barrier_output.y),
                            warm_start_basis=perturbLP_manager.recover_basis_from_sub_basis(perturb_barrier_output.basis))

    return Output(x=final_output.x,
                  y=final_output.y,
                  obj_val=final_output.obj_val)


def get_perturb_problem(lp: GeneralLP,
                        x: Optional[np.ndarray],
                        y: Optional[np.ndarray],
                        gamma: Optional[float]) -> LPManager:
    """
    Find an approximated optimal face using the given interior-point solution. And get the subproblem
    restricted on that optimal face. Use the perturbed objective c_pt.

    Args:
        lp: the original LP problem in StandardLP format.
        x: the primal interior-point solution.
        y: the dual interior-point solution.
        gamma: the parameter to approximate optimal face: x < gamma * s

    Returns:
        A LP manager of a sub-problem restricted on the approximated optimal face.

    """

    def get_dual_slack(A: sp.csr_matrix, c: np.ndarray, y: np.ndarray) -> np.ndarray:
        return c - A.transpose() @ y

    def get_primal_slack(A: sp.csr_matrix, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        return b - A @ x

    s_d = get_dual_slack(lp.A, lp.c, y)
    s_p = get_primal_slack(lp.A, lp.b, x)

    subLP_manager = LPManager(lp.copy())
    subLP_manager.fix_variables(ind_fix_to_low=np.where(x - lp.l < gamma * s_d)[0],
                                ind_fix_to_up=np.where(lp.u - x < gamma * -s_d)[0])
    subLP_manager.fix_constraints(ind_fix_to_up=np.where(s_p < gamma * -y)[0])
    logging.critical("  The number of fixed variables is %d." % subLP_manager.get_num_fixed_variables())
    logging.critical("  The number of fixed constraints is %d." % subLP_manager.get_num_fixed_constraints())
    subLP_manager.update_subproblem()

    subLP_manager.update_c(perturb_c(subLP_manager.lp_sub, subLP_manager.get_subx(x)))

    return subLP_manager


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
    n = len(x)

    x_real = get_x_perturb_val(lp_ori, x)
    # Compute the perturbation vector = scale_factor / x_real * np.random / ||np.random||.
    np.random.seed(42)
    p = np.random.uniform(0.9, 1, x_real.size)
    p = p / np.linalg.norm(p)
    projector = get_projector_Xc(lp_ori, x_real)
    scale_factor = get_scale_factor(projector, n + np.count_nonzero(lp_ori.sense == '<'))
    p = p / x_real * scale_factor

    logging.critical("  Projector: %(pj)s, \n  Scale factor: %(sf)s", {'pj': np.linalg.norm(projector), 'sf': scale_factor})

    c_pt = lp_ori.c + p
    return c_pt


def get_projector_Xc(lp_ori: GeneralLP,
                     x: np.ndarray) -> np.ndarray:
    """Get the projector of c."""

    # calculate the projector: [I - (A X)^T (A X X A^T)† (A X)] X c
    return apply_projector(lp_ori.A @ sp.diags(x), sp.diags(x) @ lp_ori.c)


def apply_projector(Y, v, tol=1e-8, max_iter=1000):
    """ Apply the projection matrix (I - Y^T (Y Y^T)† Y) to a vector v using conjugate gradient method. """
    Yv = Y @ v
    z, _ = splinalg.cg(Y @ Y.T, Yv, tol=tol, maxiter=max_iter)
    return v - Y.T @ z


def get_scale_factor(projector: np.ndarray, n: int) -> float:
    """Get the scale factor for perturbation."""
    # scale_factor = ||projector|| / (CONSTANT_SCALE_FACTOR * n)
    return np.linalg.norm(projector) / (CONSTANT_SCALE_FACTOR * n)


def get_x_perturb_val(lp: GeneralLP,
                      x: np.ndarray) -> np.ndarray:
    """Get the x values used for perturbation estimation."""
    x_min = np.minimum(x - lp.l, lp.u - x)
    x_min[x_min < PERTURB_THRESHOLD] = PERTURB_THRESHOLD
    x_min[lp.get_free_variables()] = 0  # free variables are not perturbed
    return x_min


def check_perturb_output_precision(sublp_manager: LPManager,
                                   x_ptb: np.ndarray,
                                   c_ori: np.ndarray,
                                   barrier_obj: float) -> bool:
    """Check if the perturbation is precise enough."""

    x = sublp_manager.get_orix(x_ptb)
    my_primal_obj = c_ori @ x
    primal_dual_gap = abs(my_primal_obj - barrier_obj)
    relative_primal_dual_gap = primal_dual_gap / (abs(my_primal_obj) + abs(barrier_obj) + 1)
    logging.critical("  Primal-dual gap: %(gap).2e", {'gap': relative_primal_dual_gap})
    if relative_primal_dual_gap < PRIMAL_DUAL_GAP_THRESHOLD:
        return True
