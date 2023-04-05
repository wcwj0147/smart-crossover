"""Module to call solvers to run barrier/simplex algorithms on LP problems."""
import datetime
from dataclasses import dataclass
from typing import Union, Tuple

import gurobipy
import numpy as np

from smart_crossover.formats import MinCostFlow, OptTrans, StandardLP
from smart_crossover.output import Basis, Output


@dataclass
class SolverSettings:
    presolve: int = 0
    barrierTol: float = 1e-8
    optimalityTol: float = 1e-6
    time_limit: int = 600
    log_file: str = ""
    log_console: int = 1


class SolverCaller:
    """Class to call optimization solver on LP problems."""
    solver_name: str
    model: Union[gurobipy.Model]

    def read_model_from_file(self, path: str) -> None:
        """Read a model from .mps/.lp file in the given file path."""
        ...

    def read_model(self, model: Union[gurobipy.Model]) -> None:
        ...

    def read_mcf(self, mcf: MinCostFlow) -> None:
        ...

    def read_ot(self, ot: OptTrans) -> None:
        ...

    def read_lp(self, lp: StandardLP) -> None:
        ...

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        ...

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray[np.float64], np.ndarray[np.float64]]):
        ...

    def return_basis(self) -> Basis:
        ...

    def return_MCF_model(self) -> MinCostFlow:
        ...

    def return_x(self) -> np.ndarray:
        ...

    def return_y(self) -> np.ndarray:
        ...

    def return_barx(self) -> np.ndarray:
        ...

    def return_obj_val(self) -> float:
        ...

    def return_runtime(self) -> datetime.timedelta:
        ...

    def return_iter_count(self) -> float:
        ...

    def return_bar_iter_count(self) -> int:
        ...

    def return_reduced_cost(self) -> np.ndarray:
        ...

    def return_output(self) -> Output:
        return Output(x=self.return_x(),
                      y=self.return_y(),
                      x_bar=self.return_barx(),
                      obj_val=self.return_obj_val(),
                      runtime=self.return_runtime(),
                      iter_count=self.return_iter_count(),
                      basis=self.return_basis()
                      )

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

    def get_model_report(self) -> None:
        ...
