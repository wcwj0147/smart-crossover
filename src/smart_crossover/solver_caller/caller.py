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

from smart_crossover.formats import MinCostFlow, OptTransport, StandardLP, GeneralLP
from smart_crossover.output import Basis, Output


@dataclass
class SolverSettings:
    """Class to store the settings of the solver.

    Attributes:
        presolve: Whether to use presolve.
        crossover: Whether to use crossover.
        barrierTol: Barrier tolerance.
        optimalityTol: Optimality tolerance.
        timeLimit: Time limit for the solver.
        log_file: Path to the log file.
        log_console: Whether to log on console.
        iterLimit: Maximum number of iterations.
        simplexPricing: Pricing strategy for simplex method.

    """
    presolve: str = "on"
    crossover: str = "on"
    barrierTol: float = 1e-8
    optimalityTol: float = 1e-6
    timeLimit: int = 3600
    log_file: str = ""
    log_console: int = 1
    iterLimit: int = 1000
    simplexPricing: str = ''


class SolverCaller(ABC):
    """Class to call optimization solver on LP problems.

    Attributes:
        solver_name: Name of the solver.
        model: The model solved or to be solved.

    """
    solver_name: str
    model: Union[gurobipy.Model, cplex.Cplex, mf.Model]

    def read_model_from_file(self, path: str) -> None:
        """Read a model from .mps/.lp file in the given file path."""
        ...

    def read_model(self, model: Union[gurobipy.Model]) -> None:
        self.model = model

    def read_mcf(self, mcf: MinCostFlow) -> None:
        """Read a MinCostFlow problem. """
        ...

    def read_ot(self, ot: OptTransport) -> None:
        """Read an Optimal Transport problem."""
        self.read_mcf(ot.to_MCF())

    def read_stdlp(self, stdlp: StandardLP) -> None:
        """Read a StandardLP problem."""
        ...

    def read_genlp(self, genlp: GeneralLP) -> None:
        """Read a GeneralLP problem."""
        ...

    def get_A(self) -> sp.csr_matrix:
        """Get A matrix of the current model."""
        ...

    def get_b(self) -> np.ndarray:
        """Get b vector of the current model."""
        ...

    def get_sense(self) -> np.ndarray:
        """Get sense vector of the current model."""
        ...

    def get_c(self) -> np.ndarray:
        """Get c vector of the current model."""
        ...

    def get_l(self) -> np.ndarray:
        """Get l vector of the current model."""
        ...

    def get_u(self) -> np.ndarray:
        """Get u vector of the current model."""
        ...

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        """Add a warm start basis to the current model."""
        ...

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        """Add a warm start solution to the current model."""
        ...

    def return_basis(self) -> Basis:
        """Return the basis of the current model."""
        ...

    def return_mcf(self) -> MinCostFlow:
        """Return the MinCostFlow problem."""
        return MinCostFlow(self.get_A(), self.get_b(), self.get_c(), self.get_u())

    def return_stdlp(self) -> StandardLP:
        """Return the StandardLP problem."""
        return StandardLP(A=self.get_A(), b=self.get_b(), c=self.get_c(), u=self.get_u())

    def return_genlp(self) -> GeneralLP:
        """Return the GeneralLP problem."""
        return GeneralLP(A=self.get_A(), b=self.get_b(), c=self.get_c(), l=self.get_l(), u=self.get_u(), sense=self.get_sense())

    def return_x(self) -> np.ndarray:
        """Return the primal solution."""
        ...

    def return_y(self) -> np.ndarray:
        """Return the dual solution."""
        ...

    def return_barx(self) -> np.ndarray:
        """Return the barrier solution."""
        ...

    def return_obj_val(self) -> float:
        """Return the objective value."""
        ...

    def return_runtime(self) -> datetime.timedelta:
        """Return the runtime of the solver."""
        ...

    def return_iter_count(self) -> int:
        """Return the number of iterations."""
        ...

    def return_bar_iter_count(self) -> int:
        """Return the number of barrier iterations."""
        ...

    def return_reduced_cost(self) -> np.ndarray:
        """Return the reduced cost."""
        ...

    def return_status(self) -> str:
        """Return the status of the solver."""
        ...

    def return_output(self) -> Output:
        """Return the output."""
        if self.return_status() != "OPTIMAL":
            return Output(
                runtime=self.return_runtime(),
                status=self.return_status())
        return Output(x=self.return_x(),
                      y=self.return_y(),
                      x_bar=self.return_barx(),
                      obj_val=self.return_obj_val(),
                      runtime=self.return_runtime(),
                      iter_count=self.return_iter_count(),
                      bar_iter_count=self.return_bar_iter_count(),
                      basis=self.return_basis(),
                      status=self.return_status()
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
        """Run simplex algorithm."""
        ...

    def run_primal_simplex(self) -> None:
        """Run primal simplex algorithm."""
        ...

    def run_dual_simplex(self) -> None:
        """Run dual simplex algorithm."""
        ...

    def run_network_simplex(self) -> None:
        """Run network simplex algorithm."""
        ...

    def reset_model(self) -> None:
        """Clear the current model."""
        ...

    def _run(self) -> None:
        """Run the solver."""
        ...

    def _set_presolve(self) -> None:
        """Set the presolve option."""
        ...

    def _set_log(self) -> None:
        """Set the log file."""
        ...

    def _set_time_limit(self) -> None:
        """Set the time limit."""
        ...

    def _set_tol(self) -> None:
        """Set the tolerance."""
        ...

    def _set_pricing(self) -> None:
        """Set the pricing strategy."""
        ...
