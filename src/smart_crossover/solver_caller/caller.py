"""Module to call solvers to run barrier/simplex algorithms on LP problems."""
import datetime
from abc import ABC
from dataclasses import dataclass
from typing import Union, Tuple

import gurobipy
import numpy as np
import scipy.sparse as sp
import cplex
import mosek.fusion as mf

from smart_crossover.formats import MinCostFlow, OptTransport, StandardLP
from smart_crossover.output import Basis, Output


@dataclass
class SolverSettings:
    presolve: str = "on"
    barrierTol: float = 1e-8
    optimalityTol: float = 1e-6
    timeLimit: int = 600
    log_file: str = ""
    log_console: int = 1


class SolverCaller(ABC):
    """Class to call optimization solver on LP problems."""
    solver_name: str
    model: Union[gurobipy.Model, cplex.Cplex, mf.Model]

    def read_model_from_file(self, path: str) -> None:
        """Read a model from .mps/.lp file in the given file path."""
        ...

    def read_model(self, model: Union[gurobipy.Model]) -> None:
        ...

    def read_mcf(self, mcf: MinCostFlow) -> None:
        ...

    def read_ot(self, ot: OptTransport) -> None:
        self.read_mcf(ot.to_MCF())

    def read_lp(self, lp: StandardLP) -> None:
        ...

    def get_A(self) -> sp.csr_matrix:
        ...

    def get_b(self) -> np.ndarray:
        ...

    def get_sense(self) -> np.ndarray:
        ...

    def get_c(self) -> np.ndarray:
        ...

    def get_u(self) -> np.ndarray:
        ...

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        ...

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        ...

    def return_basis(self) -> Basis:
        ...

    def return_MCF(self) -> MinCostFlow:
        return MinCostFlow(self.get_A(), self.get_b(), self.get_c(), self.get_u())

    def return_StdLP(self) -> StandardLP:
        return StandardLP(self.get_A(), self.get_b(), self.get_c(), self.get_u())

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

    def return_iter_count(self) -> int:
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

    def run_default(self) -> None:
        """Run the default algorithm on the current model."""
        ...

    def run_barrier(self) -> None:
        """Run barrier algorithm on the current model, crossover on."""
        ...

    def run_barrier_no_crossover(self) -> None:
        """Run barrier algorithm on the current model, crossover off."""
        ...

    def run_simplex(self) -> None:
        ...

    def run_dual_simplex(self) -> None:
        ...

    def run_network_simplex(self) -> None:
        ...

    def reset_model(self) -> None:
        ...

    def _run(self) -> None:
        ...

    def _set_presolve(self) -> None:
        ...

    def _set_log(self) -> None:
        ...

    def _set_time_limit(self) -> None:
        ...

    def _set_tol(self) -> None:
        ...
