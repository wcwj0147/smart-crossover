from typing import Protocol, Union, Tuple
import gurobipy
import numpy as np

from smart_crossover.input import MCFInput
from smart_crossover.output import Basis


# Add mosek and cplex later...
class SolverRunner(Protocol):
    """Class to call optimization solver on LP problems."""
    solver_name: str
    model: Union[gurobipy.Model]

    def read_model_from_path(self, path: str) -> None:
        """Read an LP model from .mps file in the given path."""
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

    def return_runtime(self) -> float:
        ...

    def return_iter_count(self) -> float:
        ...

    def return_reduced_cost(self) -> np.ndarray:
        ...

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
