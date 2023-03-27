"""Module to call solvers to run barrier/simplex algorithms on LP problems."""
import datetime
from dataclasses import dataclass
from typing import Union

import gurobipy
import numpy as np

from smart_crossover.input import MCFInput
from smart_crossover.output import Basis, Output
from smart_crossover.solver_caller.gurobi import GrbCaller


class SolverCaller:
    """Class to call optimization solver on LP problems."""
    solver_name: str
    model: Union[gurobipy.Model]

    def read_model_from_path(self, path: str) -> None:
        """Read an LP model from .mps file in the given path."""
        ...

    def read_model(self, model: Union[gurobipy.Model]) -> None:
        ...

    def read_mcf_input(self, mcf_input: MCFInput) -> None:
        ...

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        ...

    def return_basis(self) -> Basis:
        ...

    def return_MCF_model(self) -> MCFInput:
        ...

    def return_x(self) -> np.ndarray:
        ...

    def return_y(self) -> np.ndarray:
        ...

    def return_obj_val(self) -> float:
        ...

    def return_runtime(self) -> datetime.timedelta:
        ...

    def return_iter_count(self) -> float:
        ...

    def return_reduced_cost(self) -> np.ndarray:
        ...

    def return_output(self) -> Output:
        return Output(x=self.return_x(),
                      y=self.return_y(),
                      obj_val=self.return_obj_val(),
                      runtime=self.return_runtime(),
                      iter_count=self.return_iter_count(),
                      rcost=self.return_reduced_cost(),
                      basis=self.return_basis())

    def turn_off_presolve(self) -> None:
        ...

    def run_barrier(self) -> None:
        """Run barrier algorithm on the current model, crossover on."""
        ...

    def run_simplex(self) -> None:
        """Run simplex/network simplex algorithm on the current model."""
        ...

    def run_network_simplex(self) -> None:
        ...

    def run_barrier_no_crossover(self) -> None:
        """Run barrier algorithm on the current model, crossover off."""
        ...

    def reset_model(self) -> None:
        ...

    def _run(self) -> None:
        """Run the solver with the current settings."""
        ...


@dataclass
class SolverSettings:
    presolve: int = -1,
    tolerance: float = 1e-8,
    time_limit: int = 3600,
    log_file: str = "",
    log_console: int = 1


def generate_caller(solver: str = "GRB",
                    solver_settings: SolverSettings = SolverSettings()) -> SolverCaller:
    if solver == "GRB":
        runner = GrbCaller(solver_settings)
    else:
        runner = GrbCaller(solver_settings)
    # only test with Gurobi for now.
    return runner
