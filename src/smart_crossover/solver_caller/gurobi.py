import datetime
from math import inf
from typing import Optional, Any, Tuple

import gurobipy
from gurobipy import GRB
import numpy as np
import scipy

from smart_crossover.formats import MinCostFlow, StandardLP, GeneralLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class GrbCaller(SolverCaller):
    model: gurobipy.Model
    """Gurobi runner."""

    def __init__(self,
                 solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "gurobi"
        self.settings = solver_settings

    def read_model_from_file(self, path: str) -> None:
        model = gurobipy.read(path)
        self.read_model(model)

    def read_model(self, model: gurobipy.Model) -> None:
        self.model = model

    def read_mcf(self, mcf: MinCostFlow) -> None:
        model = gurobipy.Model()
        x = model.addMVar(shape=mcf.c.size, ub=mcf.u, name='x')
        model.setObjective(mcf.c @ x, GRB.MINIMIZE)
        model.addMConstr(mcf.A, x, '=', mcf.b, name='Ax')
        self.model = model
        self.model.update()  # Let MVar, MConstr info to appear in the model

    def read_lp(self, lp: StandardLP) -> None:
        model = gurobipy.Model()
        x = model.addMVar(shape=lp.c.size, ub=lp.u, name='x')
        model.setObjective(lp.c @ x, GRB.MINIMIZE)
        model.addMConstr(lp.A, x, '=', lp.b, name='Ax')
        self.model = model
        self.model.update()

    def get_A(self) -> scipy.sparse.csr_matrix:
        return self.model.getA()

    def get_b(self) -> np.ndarray:
        return np.array(self.model.getAttr("RHS", self.model.getConstrs()))

    def get_sense(self) -> np.ndarray:
        return np.array(self.model.getAttr("Sense", self.model.getConstrs()))

    def get_c(self) -> np.ndarray:
        return np.array(self.model.getAttr("obj", self.model.getVars()))

    def get_l(self) -> np.ndarray:
        return np.array(self.model.getAttr("LB", self.model.getVars()))

    def get_u(self) -> np.ndarray:
        return np.array(self.model.getAttr("UB", self.model.getVars()))

    def get_model_report(self) -> str:
        report_str = f"\n {self.model.ModelName}: \n"
        report_str += f"Model has {self.model.NumVars} variables and {self.model.NumConstrs} constraints.\n"
        eq_constr_count = sum(1 for constr in self.model.getConstrs() if constr.Sense == '=')
        le_constr_count = sum(1 for constr in self.model.getConstrs() if constr.Sense == '<')
        ge_constr_count = sum(1 for constr in self.model.getConstrs() if constr.Sense == '>')
        report_str += f"where {eq_constr_count} constraints are '=', {le_constr_count} constraints are '<', and {ge_constr_count} constraints are '>'.\n"
        ub_var_count = sum(1 for var in self.model.getVars() if var.UB != inf)
        lb_var_count = sum(1 for var in self.model.getVars() if var.LB != 0)
        free_var_count = sum(1 for var in self.model.getVars() if var.UB == inf and var.LB == -inf)
        report_str += f"{ub_var_count} variables have upper bounds, and {lb_var_count} variables have non-zero lower bounds.\n"
        report_str += f"{free_var_count} variables are free.\n"
        return report_str

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        self.model.setAttr("VBasis", self.model.getVars(), basis.vbasis.tolist())
        self.model.setAttr("CBasis", self.model.getConstrs(), basis.cbasis.tolist())
        self.model.setParam("LPWarmStart", 2)  # Make warm-start basis work when conduct presolve.

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray[np.float64], np.ndarray[np.float64]]):
        self.model.setAttr("PStart", self.model.getVars(), start_solution[0].tolist())
        self.model.setAttr("DStart", self.model.getConstrs(), start_solution[1].tolist())
        self.model.setParam("LPWarmStart", 2)

    def return_basis(self) -> Optional[Basis]:
        if self.model.Params.Method == 2:
            return None
        vbasis = np.array(self.model.getAttr("VBasis", self.model.getVars()))
        cbasis = np.array(self.model.getAttr("CBasis", self.model.getConstrs()))
        return Basis(vbasis, cbasis)

    def run_barrier(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.settings.barrierTol)
        self.model.setParam("Crossover", -1)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_barrier_no_crossover(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("BarConvTol", self.settings.barrierTol)
        self.model.setParam("Crossover", 0)
        self._set_log()
        self._set_time_limit()
        self.turn_off_presolve()
        self._run()

    def run_simplex(self) -> None:
        self.model.setParam("Method", -1)
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

    def return_MCF(self) -> MinCostFlow:
        return MinCostFlow(self.get_A(), self.get_b(), self.get_c(), self.get_u())

    def return_StdLP(self) -> StandardLP:
        return StandardLP(self.get_A(), self.get_b(), self.get_c(), self.get_u())

    def return_GenLP(self) -> GeneralLP:
        return GeneralLP(self.get_A(), self.get_b(), self.get_sense(), self.get_c(), self.get_l(), self.get_u())

    def return_x(self) -> np.ndarray:
        assert self.model.Status == GRB.OPTIMAL, "The model is not solved to optimal!"
        return np.array(self.model.getAttr("X", self.model.getVars()))

    def return_y(self) -> np.ndarray:
        return np.array(self.model.getAttr("Pi", self.model.getConstrs()))

    def return_barx(self) -> Optional[np.ndarray]:
        if self.model.Params.Method != 2:
            return None
        return np.array(self.model.getAttr("BarX", self.model.getVars()))

    def return_obj_val(self) -> float:
        return self.model.ObjVal

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.model.Runtime)

    def return_iter_count(self) -> float:
        return self.model.IterCount

    def return_bar_iter_count(self) -> int:
        return self.model.BarIterCount

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
