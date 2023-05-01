import numpy as np
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pywrap_pdlp
import scipy.sparse as sp

from smart_crossover.formats import MinCostFlow
from smart_crossover.output import Output
from smart_crossover.timer import Timer


def mcf_to_pdlp(mcf: MinCostFlow) -> pywrap_pdlp.QuadraticProgram:
    m, n = mcf.A.shape

    lp = pywrap_pdlp.QuadraticProgram()

    # Objective function
    lp.objective_vector = mcf.c

    # Constraint matrix
    lp.constraint_matrix = sp.csc_matrix(mcf.A)

    # Constraint bounds
    lp.constraint_lower_bounds = mcf.b
    lp.constraint_upper_bounds = mcf.b

    # Variable bounds
    lp.variable_lower_bounds = np.zeros(n)
    lp.variable_upper_bounds = mcf.u

    return lp


def pdlp_for_mcf(mcf: MinCostFlow) -> Output:
    timer = Timer()
    timer.start_timer()

    params = solvers_pb2.PrimalDualHybridGradientParams()
    optimality_criteria = params.termination_criteria.simple_optimality_criteria
    optimality_criteria.eps_optimal_relative = 1.0e-6
    optimality_criteria.eps_optimal_absolute = 1.0e-6
    params.termination_criteria.time_sec_limit = np.inf
    params.num_threads = 1
    params.verbosity_level = 0
    params.presolve_options.use_glop = False

    # Call the main solve function.
    lp = mcf_to_pdlp(mcf)
    result = pywrap_pdlp.primal_dual_hybrid_gradient(lp, params.SerializeToString())
    solve_log = solve_log_pb2.SolveLog.FromString(result.solve_log_str)

    if solve_log.termination_reason == solve_log_pb2.TERMINATION_REASON_OPTIMAL:
        print('Solve successful')
    else:
        print(
            'Solve not successful. Status:',
            solve_log_pb2.TerminationReason.Name(solve_log.termination_reason))

    print('Primal solution:', result.primal_solution)
    print('Dual solution:', result.dual_solution)
    print('Reduced costs:', result.reduced_costs)

    solution_type = solve_log.solution_type
    print('Solution type:', solve_log_pb2.PointType.Name(solution_type))
    for ci in solve_log.solution_stats.convergence_information:
        if ci.candidate_type == solution_type:
            print('Primal objective:', ci.primal_objective)
            print('Dual objective:', ci.dual_objective)

    print('Iterations:', solve_log.iteration_count)
    print('Solve time (sec):', solve_log.solve_time_sec)

    return Output(x=result.primal_solution, y=result.dual_solution, rcost=result.reduced_costs, runtime=solve_log.solve_time_sec, iter_count=solve_log.iteration_count)
