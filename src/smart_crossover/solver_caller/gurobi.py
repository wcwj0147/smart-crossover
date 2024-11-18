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
    """Gurobi caller."""

    def __init__(self,
                 solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        """Initialize the Gurobi caller. Add the solver name and settings."""
        self.solver_name = "GRB"
        self.settings = solver_settings

    def read_model_from_file(self, path: str) -> None:
        """Read a model from .mps file in the given file path. Relax the problem (for MIP model) and presolve it."""
        self.model = gurobipy.read(path)
        self.model = self.model.relax()
        self.model = self.model.presolve()

    def read_stdlp(self, stdlp: StandardLP) -> None:
        """Read a StandardLP instance."""
        model = gurobipy.Model()
        x = model.addMVar(shape=stdlp.c.size, ub=stdlp.u, name='x')
        model.setObjective(stdlp.c @ x, GRB.MINIMIZE)
        model.addMConstrs(stdlp.A, x, '=', stdlp.b, name='Ax')
        self.model = model
        self.model.update()

    def read_mcf(self, mcf: MinCostFlow) -> None:
        """Read a MinCostFlow instance."""
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
        """Get A matrix (model constraints: Ax (sense) b) of the current model."""
        return self.model.getA().tocsr()

    def get_b(self) -> np.ndarray:
        """Get b vector (model constraints: Ax (sense) b) of the current model."""
        return np.array(self.model.getAttr("RHS", self.model.getConstrs()))

    def get_sense(self) -> np.ndarray:
        """Get sense vector (model constraints: Ax (sense) b) of the current model."""
        return np.array(self.model.getAttr("Sense", self.model.getConstrs()))

    def get_c(self) -> np.ndarray:
        """Get c vector (cost) of the current model."""
        return np.array(self.model.getAttr("obj", self.model.getVars()))

    def get_objcon(self) -> float:
        """Get the constant term in the objective function."""
        return self.model.getAttr("objCon")

    def get_l(self) -> np.ndarray:
        """Get l vector (lower bound on variables) of the current model."""
        return np.array(self.model.getAttr("LB", self.model.getVars()))

    def get_u(self) -> np.ndarray:
        """Get u vector (upper bound on variables) of the current model."""
        return np.array(self.model.getAttr("UB", self.model.getVars()))

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        """Add a warm start basis to the current model.

        Args:
            basis: The basis, include vbasis and cbasis, to be added.
        """
        self.model.setAttr("VBasis", self.model.getVars(), basis.vbasis.tolist())
        self.model.setAttr("CBasis", self.model.getConstrs(), basis.cbasis.tolist())
        # self.model.setParam("LPWarmStart", 2)  # Make warm-start basis work when conduct presolve.

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        """Add a warm start solution to the current model.

        Args:
            start_solution: The warm start solution, include primal and dual solution, to be added.
        """
        self.model.setAttr("PStart", self.model.getVars(), start_solution[0].tolist())
        self.model.setAttr("DStart", self.model.getConstrs(), start_solution[1].tolist())
        # self.model.setParam("LPWarmStart", 2)

    def return_basis(self) -> Optional[Basis]:
        """ Get the basis of the current model if solved by simplex method or barrier method when crossover is on."""
        if self.model.Params.Crossover == 0:
            return None
        vbasis = np.array(self.model.getAttr("VBasis", self.model.getVars()))
        cbasis = np.array(self.model.getAttr("CBasis", self.model.getConstrs()))
        return Basis(vbasis, cbasis)

    def run_barrier(self) -> None:
        """Run the barrier method with crossover."""
        self.model.setParam("Method", 2)
        self.model.setParam("Crossover", -1)
        self._run()

    def run_barrier_no_crossover(self) -> None:
        """Run the barrier method without crossover."""
        self.model.setParam("Method", 2)
        self.model.setParam("Crossover", 0)
        self._run()

    def run_default(self) -> None:
        """Run the default method."""
        self.model.setParam("Method", -1)
        self._run()

    def run_simplex(self) -> None:
        """Run the simplex method."""
        self.model.setParam("Method", -1)
        self._run()

    def run_primal_simplex(self) -> None:
        """Run the primal simplex method."""
        self.model.setParam("Method", 0)
        self._run()

    def run_dual_simplex(self) -> None:
        """Run the dual simplex method."""
        self.model.setParam("Method", 1)
        self._run()

    def run_network_simplex(self) -> None:
        """Run the network simplex method."""
        self.model.setParam("NetworkAlg", 1)
        self._run()

    def reset_model(self) -> None:
        """Reset the model."""
        self.model.reset()

    def return_x(self) -> np.ndarray:
        """Return the solution vector x."""
        assert self.model.Status == GRB.OPTIMAL, "The model is not solved to optimal!"
        return np.array(self.model.getAttr("X", self.model.getVars()))

    def return_y(self) -> np.ndarray:
        """Return the dual solution vector y."""
        return np.array(self.model.getAttr("Pi", self.model.getConstrs()))

    def return_barx(self) -> Optional[np.ndarray]:
        """Return the barrier solution vector x."""
        if self.model.Params.Method != 2:
            return None
        return np.array(self.model.getAttr("BarX", self.model.getVars()))

    def return_obj_val(self) -> float:
        """Return the objective value."""
        return self.model.ObjVal

    def return_runtime(self) -> datetime.timedelta:
        """Return the runtime of the solver."""
        return datetime.timedelta(seconds=self.model.Runtime)

    def return_iter_count(self) -> int:
        """Return the number of iterations."""
        return round(self.model.IterCount)

    def return_bar_iter_count(self) -> int:
        """Return the number of barrier iterations."""
        return self.model.BarIterCount

    def return_reduced_cost(self) -> np.ndarray:
        """Return the reduced cost."""
        return np.array(self.model.getAttr("RC", self.model.getVars()))

    def return_status(self) -> str:
        """Return the status of the solution.

        Returns:
            The status of the solution, e.g. 'OPTIMAL', 'INFEASIBLE', 'UNBOUNDED' and 'UNKNOWN'.
        """
        if self.model.getAttr('Status') == GRB.OPTIMAL:
            return "OPTIMAL"
        elif self.model.getAttr('Status') == GRB.INFEASIBLE:
            return "INFEASIBLE"
        elif self.model.getAttr('Status') == GRB.UNBOUNDED:
            return "UNBOUNDED"
        else:
            return "UNKNOWN"

    def _run(self) -> None:
        """Run the model with the specified settings."""
        self._set_log()
        self._set_tol()
        self._set_time_limit()
        self._set_iter_limit()
        self._set_presolve()
        self._set_pricing()
        self.model.optimize()

    def _set_presolve(self) -> None:
        """Set the presolve option."""
        if self.settings.presolve == "off":
            self.model.setParam("Presolve", 0)
        else:
            self.model.setParam("Presolve", -1)

    def _set_tol(self) -> None:
        """Set the tolerances."""
        self.model.setParam("BarConvTol", self.settings.barrierTol)
        self.model.setParam("OptimalityTol", self.settings.optimalityTol)

    def _set_log(self) -> None:
        """Set the log settings."""
        self.model.setParam("LogFile", self.settings.log_file)
        self.model.setParam("LogToConsole", self.settings.log_console)

    def _set_time_limit(self) -> None:
        """Set the time limit."""
        self.model.setParam("TimeLimit", self.settings.timeLimit)

    def _set_iter_limit(self) -> None:
        """Set the iteration limit."""
        self.model.setParam("BarIterLimit", self.settings.iterLimit)

    def _set_pricing(self) -> None:
        """Set the simplex pricing option"""
        if self.settings.simplexPricing == 'SE':
            self.model.setParam("SimplexPricing", 1)
        if self.settings.simplexPricing == 'PP':
            self.model.setParam("SimplexPricing", 0)

    def get_model_report(self) -> str:
        """Get a brief report of the model info."""
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
