import datetime
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import mosek

from smart_crossover.formats import MinCostFlow, StandardLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class MskCaller(SolverCaller):

    def __init__(self, solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "mosek"
        self.settings = solver_settings
        self.env = mosek.Env()
        self.model = self.env.Task()

    @property
    def task(self):
        return self.model.getTask()

    def read_model_from_file(self, path: str) -> None:
        self.task.readdata(path)

    def read_lp(self, lp: StandardLP) -> None:
        num_vars = lp.c.size
        num_constraints = lp.A.shape[0]

        self.task.appendvars(num_vars)
        self.task.appendcons(num_constraints)

        for i, (ai, bi) in enumerate(zip(lp.A, lp.b)):
            self.task.putarow(i, range(num_vars), ai)
            self.task.putconbound(i, mosek.boundkey.fx, bi, bi)

        self.task.putclist(range(num_vars), lp.c)
        self.task.putvarboundslice(0, num_vars, [mosek.boundkey.ra] * num_vars, [0.0] * num_vars, lp.u)

        self.task.putobjsense(mosek.objsense.minimize)

    def read_mcf(self, mcf: MinCostFlow) -> None:
        num_vars = mcf.c.size
        num_constraints = mcf.A.shape[0]

        self.task.appendvars(num_vars)
        self.task.appendcons(num_constraints)

        for i, (ai, bi) in enumerate(zip(mcf.A, mcf.b)):
            self.task.putarow(i, range(num_vars), ai)
            self.task.putconbound(i, mosek.boundkey.fx, bi, bi)

        self.task.putclist(range(num_vars), mcf.c)
        self.task.putvarboundslice(0, num_vars, [mosek.boundkey.ra] * num_vars, [0.0] * num_vars, mcf.u)

        self.task.putobjsense(mosek.objsense.minimize)

    def get_A(self) -> sp.csr_matrix:
        row_indices = []
        col_indices = []
        data = []
        for i in range(self.task.getnumcon()):
            asub, aval = self.task.getarow(i)
            row_indices.extend([i] * len(aval))
            col_indices.extend(asub)
            data.extend(aval)

        A_coo = sp.coo_matrix((data, (row_indices, col_indices)), shape=(self.task.getnumcon(), self.task.getnumvar()))
        return A_coo.tocsr()

    def get_b(self) -> np.ndarray:
        return np.array(self.task.getconboundslice(0, self.task.getnumcon())[1])

    def get_sense(self) -> np.ndarray:
        return np.array(self.task.getconboundslice(0, self.task.getnumcon())[0])

    def get_c(self) -> np.ndarray:
        return np.array(self.task.getclist())

    def get_l(self) -> np.ndarray:
        return np.array(self.task.variables.get_lower_bounds())

    def get_u(self) -> np.ndarray:
        return np.array(self.task.variables.get_upper_bounds())

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        skc = np.array([mosek.stakey.bas if basis.vbasis[i] == 0 else mosek.stakey.low for i in range(basis.vbasis.size)])
        skx = np.array([mosek.stakey.low if basis.vbasis[i] == -1 else mosek.stakey.upr if basis.vbasis[i] == -2 else mosek.stakey.bas for i in range(basis.vbasis.size)])
        self.task.putskc(skc)
        self.task.putskx(skx)

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        self.task.putxx(start_solution[0])
        self.task.puty(start_solution[1])

    def return_basis(self) -> Optional[Basis]:
        skc = self.task.getskc()
        cbasis = np.array([0 if skc[i] == mosek.stakey.bas else -1 for i in range(self.task.getnumcon())])
        skx = self.task.getskx()
        vbasis = np.array([0 if skx[i] == mosek.stakey.bas else -1 if skx[i] == mosek.stakey.low else -2 for i in range(self.task.getnumvar())])
        return Basis(vbasis=vbasis, cbasis=cbasis)

    def return_x(self) -> np.ndarray:
        return np.array(self.task.getxx())

    def return_y(self) -> np.ndarray:
        return np.array(self.task.gety())

    def return_barx(self) -> Optional[np.ndarray]:
        # assert the crossover is off, to be checked.
        # if mosek.iparam.intpnt_basis is set to mosek.basindtype.never, then return the x
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
            return self.return_x()

    def return_obj_val(self) -> float:
        return self.task.getprimalobj(mosek.soltype.bas)

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.task.getdouinf(mosek.dinfitem.optimizer_time))

    def return_iter_count(self) -> float:
        return self.task.getintinf(mosek.iinfitem.sim_primal_iter + mosek.iinfitem.sim_dual_iter)

    def return_bar_iter_count(self) -> int:
        return self.task.getintinf(mosek.iinfitem.intpnt_iter)

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.task.getreducedcosts(mosek.soltype.bas))

    def run_default(self) -> None:
        """Run default algorithm on the current model."""
        # set the method to automatic
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.free)
        self._run()

    def run_barrier(self) -> None:
        """Run barrier algorithm on the current model, crossover on."""
        # set the method to barrier
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
        self._run()

    def run_barrier_no_crossover(self) -> None:
        """Run barrier algorithm on the current model, crossover off."""
        # set the method to barrier
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
        # set the crossover to off
        self.task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
        self._run()

    def run_primal_simplex(self) -> None:
        """Run simplex/network simplex algorithm on the current model."""
        # set the method to simplex
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
        self._run()

    def run_dual_simplex(self) -> None:
        """Run dual simplex algorithm on the current model."""
        # set the method to dual
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.dual_simplex)
        self._run()

    def run_network_simplex(self) -> None:
        """Run simplex/network simplex algorithm on the current model."""
        # NOTE: Mosek has no network simplex algorithm, so we use the general simplex algorithm instead.
        self.task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.free_simplex)
        self._run()

    def reset_model(self) -> None:
        """Reset the model to the original state."""
        self.task.clear()

    def _run(self) -> None:
        """Run the solver with the current settings."""
        self._set_tol()
        self._set_presolve()
        self._set_log()
        self._set_time_limit()
        self.task.optimize()

    def _set_tol(self) -> None:
        self.task.putdouparam(mosek.dparam.intpnt_tol_rel_gap, self.settings.barrierTol)
        self.task.putdouparam(mosek.dparam.simplex_tol_rel_gap, self.settings.optimalityTol)

    def _set_presolve(self) -> None:
        # turn off the presolve if setting.presolve is "off"
        if self.settings.presolve == "off":
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        else:
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.free)

    def _set_log(self) -> None:
        ...

    def _set_time_limit(self) -> None:
        self.task.putdouparam(mosek.dparam.optimizer_max_time, self.settings.timeLimit)
