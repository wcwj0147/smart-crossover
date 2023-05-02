from typing import Optional, Tuple

import numpy as np

from smart_crossover.formats import StandardLP, MinCostFlow, OptTransport
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
        raise ValueError("Invalid solver specified. Choose from 'GRB' or 'CPL'.")
    return caller


def solve_lp(lp: StandardLP,
             solver: str = "GRB",
             method: str = "default",
             optimalityTol: float = 1e-6,
             barrierTol: float = 1e-8,
             presolve: int = -1,
             crossover: str = 'on',
             warm_start_basis: Optional[Basis] = None,
             warm_start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(presolve=presolve, optimalityTol=optimalityTol, barrierTol=barrierTol))
    solver_caller.read_lp(lp)
    if method == "default":
        if warm_start_solution is not None:
            solver_caller.add_warm_start_solution(warm_start_solution)
        if warm_start_basis is not None:
            solver_caller.add_warm_start_basis(warm_start_basis)
        solver_caller.run_default()
    elif method == "barrier":
        if crossover == 'on':
            solver_caller.run_barrier()
        else:
            solver_caller.run_barrier_no_crossover()
    else:
        raise ValueError("Invalid method specified. Choose from 'default' or 'barrier'.")
    return solver_caller.return_output()


def solve_mcf(mcf: MinCostFlow,
              solver: str = "GRB",
              method: str = "default",
              presolve: str = "on",
              crossover: str = 'on',
              optimalityTol: float = 1e-6,
              barrierTol: float = 1e-8,
              warm_start_basis: Optional[Basis] = None) -> Output:
    solver_settings = SolverSettings(presolve=presolve, optimalityTol=optimalityTol, barrierTol=barrierTol)
    solver_caller = generate_solver_caller(solver, solver_settings)
    solver_caller.read_mcf(mcf)
    if method == "barrier":
        if crossover == 'on':
            solver_caller.run_barrier()
        else:
            solver_caller.run_barrier_no_crossover()
    else:
        if warm_start_basis is not None:
            solver_caller.add_warm_start_basis(warm_start_basis)
        if method == "default":
            solver_caller.run_default()
        elif method == "network_simplex":
            solver_caller.run_network_simplex()
        else:
            raise ValueError("Invalid method specified. Choose from 'default', 'barrier', or 'network_simplex'.")

    return solver_caller.return_output()


def solve_ot(ot: OptTransport,
             solver: str = "GRB",
             method: str = "default",
             presolve: str = "on",
             crossover: str = 'on',
             optimalityTol: float = 1e-6,
             barrierTol: float = 1e-8,
             warm_start_basis: Optional[Basis] = None) -> Output:
    solver_settings = SolverSettings(presolve=presolve, optimalityTol=optimalityTol, barrierTol=barrierTol)
    solver_caller = generate_solver_caller(solver, solver_settings)
    solver_caller.read_ot(ot)

    if method == "barrier":
        if crossover == 'on':
            solver_caller.run_barrier()
        else:
            solver_caller.run_barrier_no_crossover()
    else:
        if warm_start_basis is not None:
            solver_caller.add_warm_start_basis(warm_start_basis)
        if method == "default":
            solver_caller.run_default()
        elif method == "network_simplex":
            solver_caller.run_network_simplex()
        else:
            raise ValueError("Invalid method specified. Choose from 'default', 'barrier', or 'network_simplex'.")

    return solver_caller.return_output()
