"""Module to call solvers to run barrier/simplex algorithms on LP problems."""
from smart_crossover.solver_runner import SolverRunner
from smart_crossover.solver_runner.gurobi_runner import GrbRunner


def generate_runner(solver: str = "GRB") -> SolverRunner:
    if solver == "GRB":
        runner = GrbRunner()
    else:
        runner = GrbRunner()
    # only test with Gurobi for now.
    return runner
