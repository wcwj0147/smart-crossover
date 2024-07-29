from typing import Optional

import numpy as np

from smart_crossover.formats import OptTransport, MinCostFlow
from smart_crossover.network_methods.net_manager import OTManager, MCFManagerStd, NetworkManager
from smart_crossover.network_methods.tree_BI import tree_basis_identify
from smart_crossover.output import Output
from smart_crossover.parameters import COLUMN_GENERATION_RATIO
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.timer import Timer


def network_crossover(
        x: np.ndarray,
        ot: Optional[OptTransport] = None,
        mcf: Optional[MinCostFlow] = None,
        method: str = "tnet",
        solver: str = "GRB",
        solver_settings: SolverSettings = SolverSettings(log_console=0),
    ) -> Output:
    """
    Solve the network problem (MCF/OT) using TNET, CNET_OT, or CNET_MCF algorithms.

    Args:
        ot: the optimal transport problem (for TNET and CNET_OT).
        mcf: the minimum cost flow problem (for CNET_MCF).
        x: an interior-point / inaccurate solution of the network problem to warm-start the basis identification.
        method: the algorithm to use ('tnet', 'cnet_ot', or 'cnet_mcf').
        solver: the solver to use.
        solver_settings: the settings for the solver.

    Returns:
        the output of the selected algorithm.

    """

    print(f"*** Running {method} algorithm. ***")

    # Set the timer.
    timer = Timer()
    timer.start_timer()
    push_iter = 0   # counts the number of push iterations in TNET.

    if method == "tnet" or method == "cnet_ot":
        manager = OTManager(ot)
    elif method == "cnet_mcf":
        manager = MCFManagerStd(mcf)
    else:
        raise ValueError("Invalid method specified. Choose from 'tnet', 'cnet_ot', or 'cnet_mcf'.")

    # Get sorted flows from the interior-point solution x.
    queue, flow_indicators = manager.get_sorted_flows(x)

    if method == "tnet":
        manager.get_mcf()
        tree_basis, push_iter = tree_basis_identify(manager, flow_indicators)
        manager.set_basis(tree_basis)
        manager.add_free_variables(tree_basis.vbasis == 0)
    else:  # method in ["cnet_ot", "cnet_mcf"]
        if method == "cnet_ot":
            manager.extend_by_bigM(manager.m * np.max(ot.M))
            manager.get_mcf()
        elif method == "cnet_mcf":
            manager.rescale_cost(np.max(np.abs(mcf.c)))
            manager.fix_variables(ind_fix_to_up=np.where(x >= mcf.u / 2)[0], ind_fix_to_low=np.where(x < mcf.u / 2)[0])
            manager.extend_by_bigM(manager.m * np.max(mcf.c))
        manager.update_subproblem()
        manager.set_initial_basis()

    timer.end_timer()
    cg_output = column_generation(manager, queue, solver, solver_settings)

    print(f"*** Optimal solution found with {cg_output.iter_count + push_iter} simplex iterations in {timer.total_duration + cg_output.runtime} seconds. ***")

    return Output(x=cg_output.x, obj_val=cg_output.obj_val,
                  runtime=timer.total_duration + cg_output.runtime,
                  iter_count=cg_output.iter_count + push_iter,
                  basis=cg_output.basis)


def column_generation(net_manager: NetworkManager,
                      queue: np.ndarray,
                      solver: str,
                      solver_settings: SolverSettings) -> Output:

    # Initialize the column generation.
    timer = Timer()
    timer.start_timer()
    left_pointer = 0
    num_vars_in_next_subproblem = int(10 * net_manager.m) if net_manager.n / net_manager.m > 1000 else int(1.2 * net_manager.m)
    is_not_optimal = True
    x = None
    obj_val = None
    iter_count = 0
    cg_iter_count = 1

    while is_not_optimal:

        if left_pointer >= len(queue):
            print(' ##### Column generation fails! #####')
            break
        right_pointer = min(num_vars_in_next_subproblem, len(queue))
        net_manager.add_free_variables(queue[left_pointer:right_pointer])
        net_manager.update_subproblem()

        # Solve the sub problem.
        timer.end_timer()
        sub_output = net_manager.solve_subproblem(solver, solver_settings)
        obj_val = net_manager.recover_obj_val(sub_output.obj_val)
        timer.accumulate_time(sub_output.runtime)
        timer.start_timer()

        # Update the basis.
        net_manager.set_basis(net_manager.recover_basis_from_sub_basis(sub_output.basis))

        # Recover the solution.
        x = net_manager.recover_x_from_sub_x(sub_output.x)

        # Check stop criterion, and update num_vars_in_next_subproblem.
        if net_manager.check_optimality_condition(x, sub_output.y):
            is_not_optimal = False

        num_vars_in_next_subproblem = int(COLUMN_GENERATION_RATIO * num_vars_in_next_subproblem)
        left_pointer = right_pointer
        iter_count += sub_output.iter_count

        print(f"***  CG iteration {cg_iter_count} completed. ***")
        cg_iter_count += 1


    timer.end_timer()
    return Output(x=x, obj_val=obj_val, runtime=timer.total_duration, iter_count=iter_count, basis=net_manager.basis)
