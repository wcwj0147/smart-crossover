from typing import Tuple

import gurobipy
from gurobipy import GRB
import numpy as np
import scipy

from smart_crossover.input import MCFInput
from smart_crossover.output import Output
from smart_crossover.solver_runner import SolverRunner


class GrbRunner(SolverRunner):
    model: gurobipy.Model
    """Gurobi runner."""

    def __init__(self,
                 tolerance: float = 1e-8,
                 time_limit: int = 3600,
                 log_file: str = "",
                 log_console: int = 1) -> None:
        self.solver_name = "gurobi"
        self.tolerance = tolerance
        self.time_limit = time_limit
        self.log_file = log_file
        self.log_console = log_console

    def read_model_from_path(self, path: str) -> None:
        model = gurobipy.read(path)
        self.read_model(model)

    def read_model(self, model: gurobipy.Model) -> None:
        self.model = model

    def read_mcf_input(self, mcf_input: MCFInput) -> None:
        model = gurobipy.Model()
        x = model.addMVar(shape=mcf_input.c.size, lb=mcf_input.l, ub=mcf_input.u)
        model.setObjective(mcf_input.c @ x, GRB.MINIMIZE)
        model.addMConstr(mcf_input.A, x, '=', mcf_input.b)
        self.model = model

    def get_A(self) -> scipy.sparse.csr_matrix:
        return self.model.getA()

    def get_b(self) -> np.ndarray:
        return np.array(self.model.getAttr("RHS", self.model.getConstrs()))

    def get_c(self) -> np.ndarray:
        return np.array(self.model.getAttr("obj", self.model.getVars()))

    def get_l(self) -> np.ndarray:
        return np.array(self.model.getAttr("LB", self.model.getVars()))

    def get_u(self) -> np.ndarray:
        return np.array(self.model.getAttr("UB", self.model.getVars()))

    def add_warm_start_basis(self,
                             vbasis: np.ndarray,
                             cbasis: np.ndarray) -> None:
        for i, var in enumerate(self.model.getVars()):
            var.VBasis = vbasis[i]
        for j, constr in enumerate(self.model.getConstrs()):
            constr.CBasis = cbasis[j]

    def return_basis(self) -> Tuple[np.ndarray, np.ndarray]:
        vbasis = np.array([var.VBasis for var in self.model.getVars()])
        cbasis = np.array([constr.CBasis for constr in self.model.getConstrs()])
        return vbasis, cbasis

    def turn_off_presolve(self) -> None:
        self.model.setParam("Presolve", 0)

    def run_barrier(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.tolerance)
        self.model.setParam("Crossover", -1)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_barrier_no_crossover(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.tolerance)
        self.model.setParam("Crossover", 0)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_simplex(self) -> None:
        self.model.setParam("Method", 1)
        self._set_log()
        self._set_time_limit()
        self._run()

    def run_network_simplex(self) -> None:
        self.model.setParam("NetworkAlg", 1)
        self._set_log()
        self._set_time_limit()
        self._run()

    def reset_model(self) -> None:
        self.model.reset()

    def return_MCF_model(self) -> MCFInput:
        return MCFInput(self.get_A(), self.get_b(), self.get_c(), self.get_l(), self.get_u())

    def return_x(self) -> np.ndarray:
        assert self.model.Status == GRB.OPTIMAL, "The model is not solved to optimal!"
        return np.array(self.model.getAttr("X", self.model.getVars()))

    def return_obj_val(self) -> float:
        return self.model.ObjVal

    def return_runtime(self) -> float:
        return self.model.Runtime

    def return_iter_count(self) -> float:
        return self.model.IterCount

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.model.getAttr("RC", self.model.getVars()))

    def _run(self) -> None:
        self.model.optimize()

    def _set_log(self) -> None:
        self.model.setParam("LogFile", self.log_file)
        self.model.setParam("LogToConsole", self.log_console)

    def _set_time_limit(self) -> None:
        self.model.setParam("TimeLimit", self.time_limit)
