import numpy as np

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
    SCALE_FACTOR = 1e-3
    PERTURB_THRESHOLD = 1e-8
    PI_OVER_TWO = np.pi / 2
    n = len(x)

    def generate_noise() -> np.ndarray:
        noise = 1 / 6 * np.random.randn(n, 1) + 1 / 2
        noise = np.maximum(noise, 0.01)
        noise = np.minimum(noise, 0.99)
        return np.squeeze(noise)

    def get_indicators() -> np.ndarray:
        indicators = np.zeros_like(x, dtype=np.float64)
        x_min = np.minimum(x, lp_ori.u - x)
        indicators[x_min >= PERTURB_THRESHOLD] = np.arctan(1 / x_min[x_min >= PERTURB_THRESHOLD])
        return indicators

    perturbation = SCALE_FACTOR * generate_noise() * (1 + 99 * get_indicators() / PI_OVER_TWO)
    c_pt = lp_ori.c + np.abs(perturbation)
    return c_pt


def get_perturb_problem(lp: StandardLP,
                        x: np.ndarray,
                        s: np.ndarray) -> LPManager:
    """
    Find an approximated optimal face using the given interior-point solution. And get the subproblem
    restricted on that optimal face. Use the perturbed objective c_pt.

    Args:
        lp: the original LP problem in StandardLP format.
        x: the primal interior-point solution.
        s: the slack of the constraints in the dual LP.

    Returns:
        A subproblem restricted on the approximated optimal face.

    """
    BETA = 1e-2
    c_perturbed = perturb_c(lp, x)
    subLP_manager = LPManager(StandardLP(A=lp.A, b=lp.b, c=c_perturbed, l=lp.l, u=lp.u))
    subLP_manager.fix_variables(ind_fix_to_low=np.where(x < BETA * s)[0],
                                ind_fix_to_up=np.where(lp.u - x < BETA * -s)[0])
    subLP_manager.update_subproblem()
    return subLP_manager


def run_perturb_algorithm(lp: StandardLP,
                          solver: str = "GRB",
                          barrierTol: float = 1e-8,
                          optimalityTol: float = 1e-6) -> Output:
    """

    Args:
        lp:
        solver:
        barrierTol:
        optimalityTol:

    Returns:

    """
    barrier_output = solve_lp(lp, solver, method='barrier', barrierTol=barrierTol, presolve='off', crossover='off')

    def get_dual_slack() -> np.ndarray:
        return lp.c - lp.A.transpose() @ barrier_output.y

    perturbLP_manager = get_perturb_problem(lp, barrier_output.x, get_dual_slack())
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
