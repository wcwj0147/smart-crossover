from typing import Optional, Tuple, Union

import numpy as np

from smart_crossover.formats import StandardLP, MinCostFlow, OptTransport, GeneralLP
from smart_crossover.output import Output, Basis
from smart_crossover.solver_caller.caller import SolverSettings, SolverCaller
from smart_crossover.solver_caller.cplex import CplCaller
from smart_crossover.solver_caller.gurobi import GrbCaller
from smart_crossover.solver_caller.mosek import MskCaller


def generate_solver_caller(solver: str = "GRB",
                           solver_settings: SolverSettings = SolverSettings()) -> SolverCaller:
    if solver == "GRB":
        caller = GrbCaller(solver_settings)
    elif solver == "CPL":
        caller = CplCaller(solver_settings)
    elif solver == "MSK":
        caller = MskCaller(solver_settings)
    else:
        raise ValueError("Invalid solver specified. Choose from 'GRB', 'CPL' and 'MSK'.")
    return caller


def solve_problem(solver_caller: SolverCaller,
                  method: str,
                  settings: SolverSettings,
                  warm_start_basis: Optional[Basis] = None,
                  warm_start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Output:
    if method in ["default", "simplex", "network_simplex", "primal_simplex", "dual_simplex"]:
        if warm_start_solution is not None:
            solver_caller.add_warm_start_solution(warm_start_solution)
        if warm_start_basis is not None:
            solver_caller.add_warm_start_basis(warm_start_basis)
        if method == "simplex":
            solver_caller.run_simplex()
        elif method == "network_simplex":
            solver_caller.run_network_simplex()
        elif method == "primal_simplex":
            solver_caller.run_primal_simplex()
        elif method == "dual_simplex":
            solver_caller.run_dual_simplex()
        elif method == "default":
            solver_caller.run_default()
    elif method == "barrier":
        if settings.crossover == 'on':
            solver_caller.run_barrier()
        else:
            solver_caller.run_barrier_no_crossover()
    else:
        raise ValueError("Invalid method specified. Choose from 'default' or 'barrier'.")
    return solver_caller.return_output()


def solve_lp(lp: Union[GeneralLP, StandardLP],
             solver: str = "GRB",
             method: str = "default",
             settings: SolverSettings = SolverSettings(),
             warm_start_basis: Optional[Basis] = None,
             warm_start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Output:
    solver_caller = generate_solver_caller(solver, settings)
    if isinstance(lp, StandardLP):
        solver_caller.read_stdlp(lp)
    elif isinstance(lp, GeneralLP):
        solver_caller.read_genlp(lp)
    else:
        raise ValueError("Invalid LP format.")
    return solve_problem(solver_caller, method, settings, warm_start_basis, warm_start_solution)


def solve_mcf(mcf: MinCostFlow,
              solver: str = "GRB",
              method: str = "default",
              settings: SolverSettings = SolverSettings(),
              warm_start_basis: Optional[Basis] = None) -> Output:
    solver_caller = generate_solver_caller(solver, settings)
    solver_caller.read_mcf(mcf)
    return solve_problem(solver_caller, method, settings, warm_start_basis)


def solve_ot(ot: OptTransport,
             solver: str = "GRB",
             method: str = "default",
             settings: SolverSettings = SolverSettings(),
             warm_start_basis: Optional[Basis] = None) -> Output:
    solver_caller = generate_solver_caller(solver, settings)
    solver_caller.read_ot(ot)
    return solve_problem(solver_caller, method, settings, warm_start_basis)
