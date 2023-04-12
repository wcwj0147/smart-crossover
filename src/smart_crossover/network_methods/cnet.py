import datetime

import gurobipy
import numpy as np
import scipy.sparse as sp

from smart_crossover import get_project_root
from smart_crossover.formats import MinCostFlow
from smart_crossover.lp_manager import LPManager, MCFManager
from smart_crossover.output import Output, Basis
from smart_crossover.parameters import COLUMN_GENERATION_RATIO
from smart_crossover.solver_caller.gurobi import GrbCaller
from smart_crossover.solver_caller.utils import generate_solver_caller
from smart_crossover.timer import Timer


def cnet_mcf(mcf: MinCostFlow,
             x: np.ndarray,
             solver: str = "GRB") -> Output:
    """Solve the MCF problems using the Column generation based network crossover (CNET).

    Attributes:
        mcf: the MCF problem.
        x: an interior-point / inaccurate solution of the MCF problem to warm-start the basis identification.
        solver: the solver to use.

    """

    # Set the timer.
    timer = Timer()
    timer.start_timer()

    # Set the MCF manager.
    mcf_manager = MCFManager(mcf)
    # Get sorted flows from the interior-point solution x.
    queue = mcf_manager.get_sorted_flows(x)

    # Cost rescaling to avoid numerical issues.
    mcf_manager.rescale_cost(np.max(np.abs(mcf.c)))

    # Set subproblem.
    mcf_manager.fix_variables(ind_fix_to_up=np.where(x >= mcf.u / 2)[0], ind_fix_to_low=np.where(x < mcf.u / 2)[0])
    # Set extended problem.
    mcf_manager.extend_mcf_bigM(mcf_manager.n * np.max(mcf.u))
    mcf_manager.update_subproblem()

    # Set initial basis.
    vbasis_1 = np.concatenate((-np.ones(mcf_manager.n), np.zeros(mcf_manager.m)))
    vbasis_1[mcf_manager.var_info['fix_up']] = -2
    cbasis_1 = np.concatenate([-np.ones(mcf_manager.m), np.zeros(1)])
    mcf_manager.add_basis(Basis(vbasis_1, cbasis_1))

    # Initialize the column generation.
    left_pointer = 0
    num_vars_in_next_subproblem = int(1.2 * mcf_manager.m)
    is_not_optimal = True
    obj_val = None
    iter_count = 0

    while is_not_optimal:

        if left_pointer >= len(queue):
            print(' ##### Column generation fails! #####')
            break
        right_pointer = min(num_vars_in_next_subproblem, len(queue))
        mcf_manager.add_free_variables(queue[left_pointer:right_pointer])
        mcf_manager.update_subproblem()

        # Solve the sub MCF problem.
        timer.end_timer()
        sub_output = mcf_manager.solve_subproblem(solver)
        obj_val = mcf_manager.recover_obj_val(sub_output.obj_val)
        timer.accumulate_time(sub_output.runtime)
        timer.start_timer()

        # Update the basis.
        mcf_manager.basis = mcf_manager.recover_basis_from_sub_basis(sub_output.basis)

        # Recover the solution.
        x = mcf_manager.recover_x_from_sub_x(sub_output.x)

        # Check stop criterion, and update num_vars_in_next_subproblem.
        if mcf_manager.check_optimality_condition(x, sub_output.y):
            is_not_optimal = False

        num_vars_in_next_subproblem = int(COLUMN_GENERATION_RATIO * num_vars_in_next_subproblem)
        left_pointer = right_pointer
        iter_count += sub_output.iter_count

    timer.end_timer()
    return Output(x=x, obj_val=obj_val, runtime=timer.total_duration, iter_count=iter_count, basis=mcf_manager.basis)


# Debug
goto_mps_path = get_project_root() / "data/goto"
model = gurobipy.read("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/netgen_8_14a.mps")
gur_runner = GrbCaller()
gur_runner.read_model(model)
x = np.load("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/x_netgen.npy")
mcf = gur_runner.return_MCF()
cnet_mcf(mcf, x, "GRB")
