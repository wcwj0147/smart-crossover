import numpy as np

from smart_crossover.formats import StandardLP, SubLPManager
from smart_crossover.output import Output
from smart_crossover.solver_caller.utils import solve_lp_barrier, solve_lp


def perturb_c(c: np.ndarray[float],
              x: np.ndarray[float]) -> np.ndarray[np.float64]:
    """
    Perturb the input array `c` based on the interior-point solution `x`.

    Args:
        c: Original cost vector to be perturbed.
        x: An interior-point solution used to generate the perturbation.

    Returns:
        Perturbed cost vector.

    """
    SCALE_FACTOR = 1e-3
    PI_OVER_TWO = np.pi / 2
    n = len(x)

    def generate_noise() -> np.ndarray[np.float64]:
        noise = 1 / 6 * np.random.randn(n, 1) + 1 / 2
        noise = np.maximum(noise, 0.01)
        noise = np.minimum(noise, 0.99)
        return noise

    perturbation = SCALE_FACTOR * generate_noise() * (1 + 99 * np.arctan(1 / (x + 1e-16)) / PI_OVER_TWO)
    c_pt = c + np.abs(perturbation)
    return c_pt


def get_perturb_problem(lp: StandardLP,
                        x: np.ndarray[float],
                        s: np.ndarray[float]) -> SubLPManager:
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
    BETA = 1e-1
    lp.c = perturb_c(lp.c, x)
    subLP_manager = SubLPManager(ind_fix_to_low=np.where(x < BETA * s)[0],
                                 ind_fix_to_up=np.where(lp.u - x < BETA * -s)[0],
                                 lp=lp)
    return subLP_manager


def run_perturb_algorithm(lp: StandardLP,
                          solver: str = "GRB") -> Output:
    """

    Args:
        lp:
        solver:

    Returns:

    """
    barrier_output = solve_lp_barrier(lp, solver)

    def get_dual_slack() -> np.ndarray[float]:
        return lp.c - lp.A.transpose() @ barrier_output.y

    perturbLP_manager = get_perturb_problem(lp, barrier_output.x, get_dual_slack())
    perturb_barrier_output = solve_lp_barrier(perturbLP_manager.lp_sub, solver)
    final_output = solve_lp(lp, solver,
                            warm_start_solution=(perturbLP_manager.recover_x(perturb_barrier_output.x),
                                                 perturb_barrier_output.y))
    return Output(x=final_output.x,
                  y=final_output.y,
                  obj_val=final_output.obj_val,
                  runtime=barrier_output.runtime + perturb_barrier_output.runtime + final_output.runtime,
                  iter_count=perturb_barrier_output.iter_count + final_output.iter_count,
                  bar_iter_count=barrier_output.bar_iter_count + perturb_barrier_output.bar_iter_count)
