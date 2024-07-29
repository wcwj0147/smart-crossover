import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import gurobipy as gp

from smart_crossover.formats import GeneralLP
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Output
from smart_crossover.parameters import OPTIMAL_FACE_ESTIMATOR, OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO, PERTURB_THRESHOLD, \
    CONSTANT_SCALE_FACTOR, PRIMAL_DUAL_GAP_THRESHOLD, PROJECTOR_THRESHOLD, PERTURB_UPPER_BOUND
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

    print(f"*** Running the perturbation crossover algorithm... ***")
    barrier_output = solve_lp(lp, solver,
                              method='barrier',
                              settings=SolverSettings(barrierTol=barrierTol, presolve='on', crossover='off', log_file=log_file))

    is_feas_problem = check_feasibility_problem(lp)

    gamma, gamma_dual = OPTIMAL_FACE_ESTIMATOR, OPTIMAL_FACE_ESTIMATOR
    while True:

        print(f"*** Getting and solving a perturb subproblem... ***")
        perturbLP_manager = get_perturb_problem(lp, barrier_output.x, barrier_output.y, gamma, gamma_dual, is_feas=is_feas_problem)

        perturb_barrier_output = solve_lp(perturbLP_manager.lp_sub, solver=solver,
                                          method='barrier',
                                          settings=SolverSettings(presolve="on", log_file=log_file),
                                          warm_start_solution=(perturbLP_manager.get_subx(barrier_output.x),
                                                               barrier_output.y))

        if perturb_barrier_output.status == 'INFEASIBLE' or perturb_barrier_output.status =='UNBOUNDED':
            gamma = gamma * OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO
            gamma_dual = gamma_dual * OPTIMAL_FACE_ESTIMATOR_UPDATE_RATIO ** 2
            print(f"*** The perturbation is infeasible or unbounded. Increasing the optimal face and try again... ***")
        else:
            break

    is_optimal = check_perturb_output_precision(perturbLP_manager, perturb_barrier_output.x, lp.c, barrier_output.obj_val)

    if is_optimal:
        print(f"*** A primal optimal BFS is found. ***")
        return perturb_barrier_output

    final_output = solve_lp(lp, solver=solver,
                            method='simplex' if solver == 'MSK' else 'primal_simplex',
                            settings=SolverSettings(presolve="on", optimalityTol=optimalityTol, log_file=log_file),
                            warm_start_solution=(perturbLP_manager.recover_x_from_sub_x(perturb_barrier_output.x),
                                                 perturb_barrier_output.y),
                            warm_start_basis=perturbLP_manager.recover_basis_from_sub_basis(perturb_barrier_output.basis))

    return final_output


def get_perturb_problem(lp: GeneralLP,
                        x: np.ndarray,
                        y: np.ndarray,
                        gamma: float,
                        gamma_dual: float,
                        is_feas: bool) -> LPManager:
    """
    Find an approximated optimal face using the given interior-point solution. And get the subproblem
    restricted on that optimal face. Use the perturbed objective c_pt.

    Args:
        lp: the original LP problem in StandardLP format.
        x: the primal interior-point solution.
        y: the dual interior-point solution.
        gamma: the parameter to approximate optimal face: x < gamma * s
        c_new: the perturbed objective function.

    Returns:
        A LP manager of a sub-problem restricted on the approximated optimal face.

    """
    s_d = lp.get_dual_slack(y)
    s_p = lp.get_primal_slack(x)

    subLP_manager = LPManager(lp.copy())
    subLP_manager.lp.c = perturb_c(lp, x, is_feas)
    subLP_manager.fix_variables(ind_fix_to_low=np.where(x - lp.l < gamma * s_d)[0],
                                ind_fix_to_up=np.where(lp.u - x < gamma * -s_d)[0])
    subLP_manager.fix_constraints(ind_fix_to_up=np.where(s_p < gamma_dual * -y)[0])
    print(f"  The number of fixed variables is %d." % subLP_manager.get_num_fixed_variables())
    print(f"  The number of fixed constraints is %d." % subLP_manager.get_num_fixed_constraints())
    subLP_manager.update_subproblem()

    return subLP_manager


def perturb_c(lp: GeneralLP,
              x: np.ndarray,
              is_feas: bool) -> np.ndarray:
    """
    Perturb the input array `c` based on the interior-point solution `x`.

    Args:
        lp: The subLP of the original one.
        x: An interior-point solution used to generate the perturbation.
        is_feas: Whether the problem is a feasibility problem.

    Returns:
        Perturbed cost vector.

    """
    n = len(x)

    x_real = get_x_perturb_val(lp, x)
    x_real[x_real < PERTURB_THRESHOLD] = 1e-6
    x_real[lp.get_free_ind()] = 1   # Just to avoid division by 0.

    # Compute the standard perturbation vector = scale_factor / x_real * np.random / ||np.random||.
    np.random.seed(42)
    p = np.random.uniform(0.9, 1, x_real.size)
    p = p / np.linalg.norm(p)

    if is_feas:
        c_pt = lp.c + p
        return c_pt

    projector = get_projector_Xc(lp, x_real)
    scale_factor = get_scale_factor(projector, n + np.count_nonzero(lp.sense == '<'))
    # logging.critical("  Projector: %(pj)s, \n  Scale factor: %(sf)s",
    #                  {'pj': np.linalg.norm(projector), 'sf': scale_factor})

    p = np.minimum(p / x_real * scale_factor / CONSTANT_SCALE_FACTOR, PERTURB_UPPER_BOUND)
    p[lp.get_free_ind()] = 0
    c_pt = lp.c + p
    return c_pt


def get_projector_c(lp: GeneralLP) -> np.ndarray:
    """Get the projector of c onto plane {Ax = 0}."""

    # calculate the projector: [I - (A)^T (A A^T)† (A)] c
    # note that to calculate projector, it is only meaningful when the LP is in standard form
    return apply_projector_qp(lp.get_standard_A(), lp.get_standard_c())


def get_projector_Xc(lp: GeneralLP,
                     x: np.ndarray) -> np.ndarray:
    """Get the projector of Xc."""

    # calculate the projector: [I - (A X)^T (A X X A^T)† (A X)] X c
    # note that to calculate projector, it is only meaningful when the LP is in standard form
    xx = lp.get_standard_x(x)
    xx_free, xx_nonfree = xx[lp.get_free_ind()], xx[lp.get_nonfree_ind()]
    if lp.get_free_ind().size == 0:
        return apply_projector(lp.get_standard_A() @ sp.diags(xx),
                               sp.diags(xx) @ lp.get_standard_c())
    else:
        A_1 = lp.get_nonfree_var_matrix()
        A_2 = lp.get_free_var_matrix()
        trans, _ = splinalg.cg(A_2.T @ A_2, lp.get_standard_c()[lp.get_free_ind()], tol=1e-8, maxiter=1000)
        c_nonfree = lp.get_standard_c()[lp.get_nonfree_ind()] - A_1.T @ A_2 @ trans
        return apply_projector_qp(lp.get_nonfree_var_matrix() @ sp.diags(xx_nonfree),
                                  sp.diags(xx_nonfree) @ c_nonfree,
                                  lp.get_free_var_matrix())


def apply_projector(Y, v, tol=1e-8, max_iter=1000):
    """ Apply the projection matrix (I - Y^T (Y Y^T)† Y) to a vector v using conjugate gradient method. """
    Yv = Y @ v
    z, _ = splinalg.cg(Y @ Y.T, Yv, tol=tol, maxiter=max_iter)
    return v - Y.T @ z


def get_scale_factor(projector: np.ndarray, n: int) -> float:
    """Get the scale factor for perturbation."""
    # scale_factor = ||projector|| / n
    return np.linalg.norm(projector) / n


def get_x_perturb_val(lp: GeneralLP,
                      x: np.ndarray) -> np.ndarray:
    """Get the x values used for perturbation estimation."""
    x_free = x[lp.get_free_ind()]
    x_min = np.minimum(x - lp.l, lp.u - x)
    x_min[lp.get_free_ind()] = x_free
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
    print()
    print(f"*** Primal-dual gap: %(gap).2e ***" % {'gap': relative_primal_dual_gap})
    if relative_primal_dual_gap < PRIMAL_DUAL_GAP_THRESHOLD:
        return True


def check_feasibility_problem(lp: GeneralLP) -> bool:
    """ Check if the problem is a feasibility problem. """
    proj_c = get_projector_c(lp)
    if np.linalg.norm(proj_c) / np.linalg.norm(lp.c) < PROJECTOR_THRESHOLD:
        print("*** The problem is a feasibility problem. ***")
        return True
    return False


def apply_projector_qp(A: sp.csr_matrix, v: np.ndarray, A_f: sp.csr_matrix = None) -> np.ndarray:
    """ Solve the projector of v onto plane {Ax = 0}

    We formulate the following least square problem:
    minimize sum_j(z_j^2)
        s.t. z_j = x_j - v_j, j = 1, ..., n
             A x = 0

    """

    ls = gp.Model()
    m, n = A.shape
    x = ls.addMVar(shape=n, lb=float('-inf'))
    z = ls.addMVar(shape=n, lb=float('-inf'))
    ls.setObjective(z @ z, gp.GRB.MINIMIZE)
    ls.addConstr(z - x == - v)
    if A_f is None:
        ls.addConstr(A @ x == 0)
    else:
        f = ls.addMVar(shape=A_f.shape[1], lb=float('-inf'))
        ls.addConstr(A @ x + A_f @ f == 0)

    ls.setParam('BarQCPConvTol', 1e-1)
    ls.optimize()

    return x.X
