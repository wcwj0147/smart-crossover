import datetime
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import mosek

from smart_crossover.formats import MinCostFlow, StandardLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


MSK_INF = np.inf  # Only a symbolic constant, used for readability.


class MskCaller(SolverCaller):

    def __init__(self, solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "mosek"
        self.settings = solver_settings
        self.env = mosek.Env()
        self.model = self.env.Task()

    @property
    def task(self):
        return self.model

    def read_model_from_file(self, path: str) -> None:
        self.task.readdata(path)

    def read_lp(self, lp: MinCostFlow) -> None:
        num_vars = lp.c.size
        num_constraints = lp.A.shape[0]

        self.task.appendvars(num_vars)
        self.task.appendcons(num_constraints)

        for i in range(num_constraints):
            ai = lp.A.getrow(i).toarray().ravel()
            bi = lp.b[i]
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

        for i in range(num_constraints):
            ai = mcf.A.getrow(i).toarray().ravel()
            bi = mcf.b[i]
            self.task.putarow(i, range(num_vars), ai)
            self.task.putconbound(i, mosek.boundkey.fx, bi, bi)

        self.task.putclist(range(num_vars), mcf.c)

        var_list_lo = [ind for ind in range(num_vars) if mcf.u[ind] == np.inf]
        var_list_ra = [ind for ind in range(num_vars) if mcf.u[ind] != np.inf and mcf.l[ind] != -np.inf]
        self.task.putvarboundlist(var_list_ra, [mosek.boundkey.ra] * len(var_list_ra), [0.0] * len(var_list_ra), mcf.u[var_list_ra])
        self.task.putvarboundlist(var_list_lo, [mosek.boundkey.lo] * len(var_list_lo), [0.0] * len(var_list_lo), [MSK_INF] * len(var_list_lo))

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
        return np.array(self.task.getvarboundslice(0, self.task.getnumvar())[1])

    def get_u(self) -> np.ndarray:
        return np.array(self.task.getvarboundslice(0, self.task.getnumvar())[2])

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        skc = np.array([mosek.stakey.fix if basis.cbasis[i] == -1 else mosek.stakey.bas for i in range(basis.cbasis.size)])
        skx = np.array([mosek.stakey.low if basis.vbasis[i] == -1 else mosek.stakey.upr if basis.vbasis[i] == -2 else mosek.stakey.bas for i in range(basis.vbasis.size)])
        self.task.putskc(mosek.soltype.bas, skc)
        self.task.putskx(mosek.soltype.bas, skx)
        self.task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.free)

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        self.task.putxx(start_solution[0])
        self.task.puty(start_solution[1])

    def return_basis(self) -> Optional[Basis]:
        skc = [mosek.stakey.unk] * self.task.getnumcon()
        self.task.getskc(mosek.soltype.bas, skc)
        cbasis = np.array([0 if skc[i] == mosek.stakey.bas else -1 for i in range(self.task.getnumcon())])
        skx = [mosek.stakey.unk] * self.task.getnumvar()
        self.task.getskx(mosek.soltype.bas, skx)
        vbasis = np.array([0 if skx[i] == mosek.stakey.bas else -1 if skx[i] == mosek.stakey.low else -2 for i in range(self.task.getnumvar())])
        return Basis(vbasis=vbasis, cbasis=cbasis)

    def return_x(self) -> np.ndarray:
        xx = np.zeros(self.task.getnumvar())
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
            self.task.getxx(mosek.soltype.itr, xx)
        else:
            self.task.getxx(mosek.soltype.bas, xx)
        return np.array(xx)

    def return_y(self) -> np.ndarray:
        y = np.zeros(self.task.getnumcon())
        if self.task.getintparam(mosek.iparam.intpnt_basis == mosek.basindtype.never):
            self.task.gety(mosek.soltype.itr, y)
        else:
            self.task.gety(mosek.soltype.bas, y)
        return np.array(y)

    def return_barx(self) -> Optional[np.ndarray]:
        # assert the crossover is off,
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
            xx = np.zeros(self.task.getnumvar())
            self.task.getxx(mosek.soltype.itr, xx)
            return np.array(xx)
        return None

    def return_obj_val(self) -> float:
        return self.task.getprimalobj(mosek.soltype.bas)

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.task.getdouinf(mosek.dinfitem.optimizer_time))

    def return_iter_count(self) -> int:
        return self.task.getintinf(mosek.iinfitem.sim_primal_iter) + self.task.getintinf(mosek.iinfitem.sim_dual_iter)

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
        self.task.putdouparam(mosek.dparam.simplex_abs_tol_piv, self.settings.optimalityTol)

    def _set_presolve(self) -> None:
        # turn off the presolve if setting.presolve is "off"
        if self.settings.presolve == "off":
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        else:
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.free)

    def _set_log(self) -> None:
        self.env.set_Stream(mosek.streamtype.log, _stream_printer)
        self.task.set_Stream(mosek.streamtype.log, _stream_printer)

    def _set_time_limit(self) -> None:
        self.task.putdouparam(mosek.dparam.optimizer_max_time, self.settings.timeLimit)


def _stream_printer(text):
    sys.stdout.write(text)
    sys.stdout.flush()
