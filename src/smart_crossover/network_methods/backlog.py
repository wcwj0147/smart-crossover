import gurobipy
import numpy as np

from smart_crossover import get_project_root
from smart_crossover.formats import MinCostFlow, OptTransport
from smart_crossover.network_methods.net_manager import MCFManager, OTManager, NetworkManager
from smart_crossover.network_methods.tree_BI import max_weight_spanning_tree, push_tree_to_bfs
from smart_crossover.output import Output, Basis
from smart_crossover.parameters import COLUMN_GENERATION_RATIO
from smart_crossover.solver_caller.gurobi import GrbCaller
from smart_crossover.timer import Timer


def tnet_ot(ot: OptTransport,
            x: np.ndarray[np.float_],
            solver: str = "GRB") -> Output:
    """Solve the OT problems using the Tree based network crossover (TNET).

    Args:
        ot: the network problem (MCF / OT).
        x: an interior-point / inaccurate solution of the OT problem to warm-start the basis identification.
        solver: the solver to use.

    Returns:
        the output of the TNET algorithm.

    """

    # Set the timer.
    timer = Timer()
    timer.start_timer()

    # Set the ot manager.
    ot_manager = OTManager(ot)

    # Get sorted flows from the interior-point solution x.
    queue, flow_indicators = ot_manager.get_sorted_flows(x)
    # Find a max-weight spanning tree.
    tree = max_weight_spanning_tree(ot, flow_indicators)
    # Push the tree to a basic feasible solution of the OT problem.
    tree_vbasis = push_tree_to_bfs(ot_manager, tree)
    tree_basis = Basis(tree_vbasis, np.concatenate([-np.ones(ot_manager.m), np.array([0])]))
    ot_manager.set_basis(tree_basis)

    timer.end_timer()
    cg_output = column_generation(ot_manager, queue, solver)
    return Output(x=cg_output.x, obj_val=cg_output.obj_val, runtime=timer.total_duration + cg_output.runtime, basis=cg_output.basis)


def cnet_ot(ot: OptTransport,
            x: np.ndarray,
            solver: str = "GRB") -> Output:
    """Solve the OT problems using the Column generation based network crossover (CNET).

    Args:
        ot: the network problem (MCF / OT).
        x: an interior-point / inaccurate solution of the OT problem to warm-start the basis identification.
        solver: the solver to use.

    Returns:
        the output of the CNET algorithm.

    """
    # Set the timer.
    timer = Timer()
    timer.start_timer()

    # Set the ot manager.
    ot_manager = OTManager(ot)

    # Get sorted flows from the interior-point solution x.
    queue, _ = ot_manager.get_sorted_flows(x)

    # Set extended problem.
    ot_manager.extend_by_bigM(ot_manager.n * np.max(ot.M))
    ot_manager.update_subproblem()
    ot_manager.set_initial_basis()

    timer.end_timer()
    cg_output = column_generation(ot_manager, queue, solver)

    return Output(x=cg_output.x, obj_val=cg_output.obj_val, runtime=timer.total_duration + cg_output.runtime, basis=cg_output.basis)


def cnet_mcf(mcf: MinCostFlow,
             x: np.ndarray,
             solver: str = "GRB") -> Output:
    """Solve the MCF problems using the Column generation based network crossover (CNET).

    Args:
        mcf: the network problem (MCF / OT).
        x: an interior-point / inaccurate solution of the MCF problem to warm-start the basis identification.
        solver: the solver to use.

    Returns:
        the output of the CNET algorithm.

    """

    # Set the timer.
    timer = Timer()
    timer.start_timer()

    # Set the mcf manager.
    mcf_manager = MCFManager(mcf)

    # Get sorted flows from the interior-point solution x.
    queue, _ = mcf_manager.get_sorted_flows(x)

    # Cost rescaling to avoid numerical issues.
    mcf_manager.rescale_cost(np.max(np.abs(mcf.c)))
    # Set extended problem.
    mcf_manager.fix_variables(ind_fix_to_up=np.where(x >= mcf.u / 2)[0], ind_fix_to_low=np.where(x < mcf.u / 2)[0])
    mcf_manager.extend_by_bigM(mcf_manager.n * np.max(mcf.u))
    mcf_manager.update_subproblem()
    mcf_manager.set_initial_basis()

    timer.end_timer()
    cg_output = column_generation(mcf_manager, queue, solver)

    return Output(x=cg_output.x, obj_val=cg_output.obj_val, runtime=timer.total_duration + cg_output.runtime, basis=cg_output.basis)


def column_generation(net_manager: NetworkManager,
                      queue: np.ndarray[np.int64],
                      solver: str) -> Output:

    # Initialize the column generation.
    timer = Timer()
    timer.start_timer()
    left_pointer = 0
    num_vars_in_next_subproblem = int(1.2 * net_manager.m)
    is_not_optimal = True
    x = None
    obj_val = None
    iter_count = 0

    while is_not_optimal:

        if left_pointer >= len(queue):
            print(' ##### Column generation fails! #####')
            break
        right_pointer = min(num_vars_in_next_subproblem, len(queue))
        net_manager.add_free_variables(queue[left_pointer:right_pointer])
        net_manager.update_subproblem()

        # Solve the sub problem.
        timer.end_timer()
        sub_output = net_manager.solve_subproblem(solver)
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

    timer.end_timer()
    return Output(x=x, obj_val=obj_val, runtime=timer.total_duration, iter_count=iter_count, basis=net_manager.basis)

