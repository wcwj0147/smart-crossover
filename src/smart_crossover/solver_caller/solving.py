from typing import Optional, Tuple

import numpy as np

from smart_crossover.formats import StandardLP, MinCostFlow, OptTransport
from smart_crossover.output import Output, Basis
from smart_crossover.solver_caller.caller import SolverSettings, SolverCaller
from smart_crossover.solver_caller.gurobi import GrbCaller


def generate_solver_caller(solver: str = "GRB",
                           solver_settings: SolverSettings = SolverSettings()) -> SolverCaller:
    if solver == "GRB":
        runner = GrbCaller(solver_settings)
    else:
        runner = GrbCaller(solver_settings)
    # Todo: add other solvers.
    return runner


def solve_lp_barrier(lp: StandardLP,
                     solver: str = "GRB",
                     barrierTol: float = 1e-8,
                     optimalityTol: float = 1e-6) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(barrierTol=barrierTol, optimalityTol=optimalityTol))
    solver_caller.read_lp(lp)
    solver_caller.run_barrier()
    return solver_caller.return_output()


def solve_lp(lp: StandardLP,
             solver: str = "GRB",
             optimalityTol: float = 1e-6,
             warm_start_basis: Optional[Basis] = None,
             warm_start_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(optimalityTol=optimalityTol))
    solver_caller.read_lp(lp)
    if warm_start_solution is not None:
        solver_caller.add_warm_start_solution(warm_start_solution)
    if warm_start_basis is not None:
        solver_caller.add_warm_start_basis(warm_start_basis)
    solver_caller.run_simplex()
    return solver_caller.return_output()


def solve_mcf(mcf: MinCostFlow,
              solver: str = "GRB",
              optimalityTol: float = 1e-6,
              warm_start_basis: Optional[Basis] = None) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(optimalityTol=optimalityTol))
    solver_caller.read_mcf(mcf)
    if warm_start_basis is not None:
        solver_caller.add_warm_start_basis(warm_start_basis)
    solver_caller.run_simplex()
    return solver_caller.return_output()


def solve_mcf_barrier(mcf: MinCostFlow,
                      solver: str = "GRB",
                      barrierTol: float = 1e-8,
                      optimalityTol: float = 1e-6) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(barrierTol=barrierTol, optimalityTol=optimalityTol))
    solver_caller.read_mcf(mcf)
    solver_caller.run_barrier()
    return solver_caller.return_output()


def solve_ot(ot: OptTransport,
             solver: str = "GRB",
             optimalityTol: float = 1e-6,
             warm_start_basis: Optional[Basis] = None) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(optimalityTol=optimalityTol))
    solver_caller.read_ot(ot)
    if warm_start_basis is not None:
        solver_caller.add_warm_start_basis(warm_start_basis)
    solver_caller.run_simplex()
    return solver_caller.return_output()


def solve_ot_barrier(ot: OptTransport,
                     solver: str = "GRB",
                     barrierTol: float = 1e-8,
                     optimalityTol: float = 1e-6) -> Output:
    solver_caller = generate_solver_caller(solver, SolverSettings(barrierTol=barrierTol, optimalityTol=optimalityTol))
    solver_caller.read_ot(ot)
    solver_caller.run_barrier()
    return solver_caller.return_output()
