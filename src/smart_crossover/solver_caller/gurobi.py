import datetime
from typing import Tuple

import gurobipy
from gurobipy import GRB
import numpy as np
import scipy

from smart_crossover.input import MCFInput
from smart_crossover.output import Basis, Output
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class GrbCaller(SolverCaller):
    model: gurobipy.Model
    """Gurobi runner."""

    def __init__(self,
                 solver_settings: SolverSettings) -> None:
        self.solver_name = "gurobi"
        self.settings = solver_settings

    def read_model_from_path(self, path: str) -> None:
        model = gurobipy.read(path)
        self.read_model(model)

    def read_model(self, model: gurobipy.Model) -> None:
        self.model = model

    def read_mcf_input(self, mcf_input: MCFInput) -> None:
        model = gurobipy.Model()
        x = model.addMVar(shape=mcf_input.c.size, lb=mcf_input.l, ub=mcf_input.u, name='x')
        model.setObjective(mcf_input.c @ x, GRB.MINIMIZE)
        model.addMConstr(mcf_input.A, x, '=', mcf_input.b, name='Ax')
        self.model = model
        self.model.update()  # Let MVar, MConstr info to appear in the model

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
                             basis: Basis) -> None:
        self.model.setAttr("VBasis", self.model.getVars(), basis.vbasis.tolist())
        self.model.setAttr("CBasis", self.model.getConstrs(), basis.cbasis.tolist())
        self.model.setParam("LPWarmStart", 2)  # Make warm-start basis work when conduct presolve.

    def return_basis(self) -> Basis:
        vbasis = np.array(self.model.getAttr("VBasis", self.model.getVars()))
        cbasis = np.array(self.model.getAttr("CBasis", self.model.getConstrs()))
        return Basis(vbasis, cbasis)

    def run_barrier(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.settings.tolerance)
        self.model.setParam("Crossover", -1)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_barrier_no_crossover(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.settings.tolerance)
        self.model.setParam("Crossover", 0)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_simplex(self) -> None:
        # self.model.setParam("Method", 0)
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

    def return_y(self) -> np.ndarray:
        return np.array(self.model.getAttr("Pi", self.model.getConstrs()))

    def return_obj_val(self) -> float:
        return self.model.ObjVal

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.model.Runtime)

    def return_iter_count(self) -> float:
        return self.model.IterCount

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.model.getAttr("RC", self.model.getVars()))

    def _run(self) -> None:
        self.model.optimize()

    def _set_presolve(self) -> None:
        self.model.setParam("Presolve", self.settings.presolve)

    def _set_log(self) -> None:
        self.model.setParam("LogFile", self.settings.log_file)
        self.model.setParam("LogToConsole", self.settings.log_console)

    def _set_time_limit(self) -> None:
        self.model.setParam("TimeLimit", self.settings.time_limit)
