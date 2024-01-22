import datetime
from math import inf
from typing import Optional, Tuple

import gurobipy
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

from smart_crossover.formats import MinCostFlow, StandardLP, GeneralLP
from smart_crossover.output import Basis, Output
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class GrbCaller(SolverCaller):
    model: gurobipy.Model
    """Gurobi runner."""

    def __init__(self,
                 solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "GRB"
        self.settings = solver_settings

    def read_model_from_file(self, path: str) -> None:
        self.model = gurobipy.read(path)
        self.model = self.model.relax()
        self.model = self.model.presolve()

    def read_stdlp(self, stdlp: StandardLP) -> None:
        model = gurobipy.Model()
        x = model.addMVar(shape=stdlp.c.size, ub=stdlp.u, name='x')
        model.setObjective(stdlp.c @ x, GRB.MINIMIZE)
        model.addMConstrs(stdlp.A, x, '=', stdlp.b, name='Ax')
        self.model = model
        self.model.update()

    def read_mcf(self, mcf: MinCostFlow) -> None:
        self.read_stdlp(mcf)

    def read_genlp(self, genlp: GeneralLP) -> None:
        """Read a GeneralLP instance."""
        model = gurobipy.Model()
        x = model.addMVar(shape=genlp.c.size, lb=genlp.l, ub=genlp.u, name='x')
        model.setObjective(genlp.c @ x, GRB.MINIMIZE)
        model.addMConstrs(genlp.A, x, genlp.sense, genlp.b, name='Ax')
        self.model = model
        self.model.update()

    def get_A(self) -> sp.csr_matrix:
        return self.model.getA().tocsr()

    def get_b(self) -> np.ndarray:
        return np.array(self.model.getAttr("RHS", self.model.getConstrs()))

    def get_sense(self) -> np.ndarray:
        return np.array(self.model.getAttr("Sense", self.model.getConstrs()))

    def get_c(self) -> np.ndarray:
        return np.array(self.model.getAttr("obj", self.model.getVars()))

    def get_objcon(self) -> float:
        return self.model.getAttr("objCon")

    def get_l(self) -> np.ndarray:
        return np.array(self.model.getAttr("LB", self.model.getVars()))

    def get_u(self) -> np.ndarray:
        return np.array(self.model.getAttr("UB", self.model.getVars()))

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        self.model.setAttr("VBasis", self.model.getVars(), basis.vbasis.tolist())
        self.model.setAttr("CBasis", self.model.getConstrs(), basis.cbasis.tolist())
        # self.model.setParam("LPWarmStart", 2)  # Make warm-start basis work when conduct presolve.

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        self.model.setAttr("PStart", self.model.getVars(), start_solution[0].tolist())
        self.model.setAttr("DStart", self.model.getConstrs(), start_solution[1].tolist())
        # self.model.setParam("LPWarmStart", 2)

    def return_basis(self) -> Optional[Basis]:
        if self.model.Params.Crossover == 0:
            return None
        vbasis = np.array(self.model.getAttr("VBasis", self.model.getVars()))
        cbasis = np.array(self.model.getAttr("CBasis", self.model.getConstrs()))
        return Basis(vbasis, cbasis)

    def run_barrier(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("Crossover", -1)
        self._run()

    def run_barrier_no_crossover(self) -> None:
        self.model.setParam("Method", 2)
        self.model.setParam("Crossover", 0)
        self._run()

    def run_default(self) -> None:
        self.model.setParam("Method", -1)
        self._run()

    def run_simplex(self) -> None:
        self.model.setParam("Method", -1)
        self._run()

    def run_primal_simplex(self) -> None:
        self.model.setParam("Method", 0)
        self._run()

    def run_dual_simplex(self) -> None:
        self.model.setParam("Method", 1)
        self._run()

    def run_network_simplex(self) -> None:
        self.model.setParam("NetworkAlg", 1)
        self._run()

    def reset_model(self) -> None:
        self.model.reset()

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

    def return_iter_count(self) -> int:
        return round(self.model.IterCount)

    def return_bar_iter_count(self) -> int:
        return self.model.BarIterCount

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.model.getAttr("RC", self.model.getVars()))

    def return_status(self) -> str:
        if self.model.getAttr('Status') == GRB.OPTIMAL:
            return "OPTIMAL"
        elif self.model.getAttr('Status') == GRB.INFEASIBLE:
            return "INFEASIBLE"
        elif self.model.getAttr('Status') == GRB.UNBOUNDED:
            return "UNBOUNDED"
        else:
            return "UNKNOWN"

    def _run(self) -> None:
        self._set_log()
        self._set_tol()
        self._set_time_limit()
        self._set_iter_limit()
        self._set_presolve()
        self._set_pricing()
        self.model.optimize()

    def _set_presolve(self) -> None:
        if self.settings.presolve == "off":
            self.model.setParam("Presolve", 0)
        else:
            self.model.setParam("Presolve", -1)

    def _set_tol(self) -> None:
        self.model.setParam("BarConvTol", self.settings.barrierTol)
        self.model.setParam("OptimalityTol", self.settings.optimalityTol)

    def _set_log(self) -> None:
        self.model.setParam("LogFile", self.settings.log_file)
        self.model.setParam("LogToConsole", self.settings.log_console)

    def _set_time_limit(self) -> None:
        self.model.setParam("TimeLimit", self.settings.timeLimit)

    def _set_iter_limit(self) -> None:
        self.model.setParam("BarIterLimit", self.settings.iterLimit)

    def _set_pricing(self) -> None:
        if self.settings.simplexPricing == 'SE':
            self.model.setParam("SimplexPricing", 1)
        if self.settings.simplexPricing == 'PP':
            self.model.setParam("SimplexPricing", 0)

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
