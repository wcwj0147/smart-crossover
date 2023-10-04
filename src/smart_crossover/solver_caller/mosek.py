import datetime
import sys
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import mosek

from smart_crossover.formats import MinCostFlow, StandardLP, GeneralLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


MSK_INF = np.inf  # Only a symbolic constant, used for readability.


class MskCaller(SolverCaller):

    def __init__(self, solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "MSK"
        self.settings = solver_settings
        self.env = mosek.Env()
        self.model = self.env.Task()

    @property
    def task(self):
        return self.model

    def read_model_from_file(self, path: str) -> None:
        self.task.readdata(path)

    def read_stdlp(self, stdlp: MinCostFlow) -> None:
        num_vars = stdlp.c.size
        num_constraints = stdlp.A.shape[0]

        self.task.appendvars(num_vars)
        self.task.appendcons(num_constraints)

        row_indices, col_indices = stdlp.A.nonzero()
        values = stdlp.A.data

        self.task.putaijlist(row_indices, col_indices, values)
        self.task.putconboundlist(range(num_constraints), [mosek.boundkey.fx] * num_constraints, stdlp.b, stdlp.b)

        self.task.putclist(range(num_vars), stdlp.c)
        self.task.putobjsense(mosek.objsense.minimize)

        var_list_lo = [ind for ind in range(num_vars) if stdlp.u[ind] == np.inf]
        var_list_ra = [ind for ind in range(num_vars) if stdlp.u[ind] != np.inf and stdlp.l[ind] != -np.inf]
        self.task.putvarboundlist(var_list_ra, [mosek.boundkey.ra] * len(var_list_ra), [0.0] * len(var_list_ra), stdlp.u[var_list_ra])
        self.task.putvarboundlist(var_list_lo, [mosek.boundkey.lo] * len(var_list_lo), [0.0] * len(var_list_lo), [MSK_INF] * len(var_list_lo))

    def read_mcf(self, mcf: MinCostFlow) -> None:
        self.read_stdlp(mcf)

    def read_genlp(self, genlp: GeneralLP) -> None:
        num_vars = genlp.c.size
        num_constraints = genlp.A.shape[0]

        self.task.appendvars(num_vars)
        self.task.appendcons(num_constraints)

        row_indices, col_indices = genlp.A.nonzero()
        values = genlp.A.data

        self.task.putaijlist(row_indices, col_indices, values)
        constraints_bound_keys = [mosek.boundkey.fx if sense == '=' else mosek.boundkey.up if sense == '<' else mosek.boundkey.lo for sense in genlp.sense]
        self.task.putconboundlist(range(num_constraints), constraints_bound_keys, genlp.b, genlp.b)

        self.task.putclist(range(num_vars), genlp.c)
        self.task.putobjsense(mosek.objsense.minimize)

        var_list_lo = [ind for ind in range(num_vars) if genlp.u[ind] == np.inf and genlp.l[ind] != -np.inf]
        var_list_up = [ind for ind in range(num_vars) if genlp.u[ind] != np.inf and genlp.l[ind] == -np.inf]
        var_list_ra = [ind for ind in range(num_vars) if genlp.u[ind] != np.inf and genlp.l[ind] != -np.inf]
        var_list_fr = [ind for ind in range(num_vars) if genlp.u[ind] == np.inf and genlp.l[ind] == -np.inf]
        self.task.putvarboundlist(var_list_ra, [mosek.boundkey.ra] * len(var_list_ra), genlp.l[var_list_ra], genlp.u[var_list_ra])
        self.task.putvarboundlist(var_list_lo, [mosek.boundkey.lo] * len(var_list_lo), genlp.l[var_list_lo], [MSK_INF] * len(var_list_lo))
        self.task.putvarboundlist(var_list_up, [mosek.boundkey.up] * len(var_list_up), genlp.l[var_list_lo], genlp.u[var_list_up])
        self.task.putvarboundlist(var_list_fr, [mosek.boundkey.fr] * len(var_list_fr), [MSK_INF] * len(var_list_lo), [MSK_INF] * len(var_list_lo))

    def get_A(self) -> sp.csr_matrix:
        num_con = self.task.getnumcon()
        num_var = self.task.getnumvar()
        row_indices = []
        col_indices = []
        data = []

        for i in range(num_con):
            asub = [0] * self.task.getarownumnz(i)
            aval = [0.0] * self.task.getarownumnz(i)
            nzi = self.task.getarow(i, asub, aval)
            asub, aval = asub[:nzi], aval[:nzi]

            row_indices.extend([i] * len(aval))
            col_indices.extend(asub)
            data.extend(aval)

        A_coo = sp.coo_matrix((data, (row_indices, col_indices)), shape=(num_con, num_var))
        return A_coo.tocsr()

    def get_b(self) -> np.ndarray:
        m = self.task.getnumcon()
        b = np.zeros(m)
        self.task.getconboundslice(0, m, [mosek.boundkey.fx] * m, b, b)
        return b

    def get_sense(self) -> np.ndarray:
        m = self.task.getnumcon()
        sense = np.array([mosek.boundkey.fx] * m)
        self.task.getconboundslice(0, m, sense, [0.0]*m, [0.0]*m)
        sense = np.array(['=' if s == mosek.boundkey.fx else '<' if s == mosek.boundkey.up else '>' for s in sense])
        return sense

    def get_c(self) -> np.ndarray:
        num_var = self.task.getnumvar()
        c = np.zeros(num_var)
        self.task.getclist(range(num_var), c)
        return c

    def get_l(self) -> np.ndarray:
        num_var = self.task.getnumvar()
        bk, bl, bu = [0] * num_var, [0.0] * num_var, [0.0] * num_var
        self.task.getvarboundslice(0, num_var, bk, bl, bu)
        return np.array(bl)

    def get_u(self) -> np.ndarray:
        num_var = self.task.getnumvar()
        bk, bl, bu = [0] * num_var, [0.0] * num_var, [0.0] * num_var
        self.task.getvarboundslice(0, num_var, bk, bl, bu)
        return np.array(bu)

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        skc = np.array([mosek.stakey.fix if basis.cbasis[i] == -1 else mosek.stakey.bas for i in range(basis.cbasis.size)])
        skx = np.array([mosek.stakey.low if basis.vbasis[i] == -1 else mosek.stakey.upr if basis.vbasis[i] == -2 else mosek.stakey.bas for i in range(basis.vbasis.size)])
        self.task.putskc(mosek.soltype.bas, skc)
        self.task.putskx(mosek.soltype.bas, skx)
        self.task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.free)

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        self.task.putxx(mosek.soltype.bas, start_solution[0])
        self.task.puty(mosek.soltype.bas, start_solution[1])
        self.task.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.free)

    def return_basis(self) -> Optional[Basis]:
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
            return None
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
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
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
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never:
            return self.task.getprimalobj(mosek.soltype.itr)
        return self.task.getprimalobj(mosek.soltype.bas)

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.task.getdouinf(mosek.dinfitem.optimizer_time))

    def return_iter_count(self) -> int:
        return self.task.getintinf(mosek.iinfitem.sim_primal_iter) + self.task.getintinf(mosek.iinfitem.sim_dual_iter)

    def return_bar_iter_count(self) -> int:
        return self.task.getintinf(mosek.iinfitem.intpnt_iter)

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.task.getreducedcosts(mosek.soltype.bas))

    def return_status(self) -> str:
        if self.task.getintparam(mosek.iparam.intpnt_basis) == mosek.basindtype.never or self.task.getsolsta(mosek.soltype.bas) == mosek.solsta.optimal:
            return "OPTIMAL"
        if self.task.getprosta(mosek.soltype.bas) == mosek.prosta.prim_infeas or self.task.getprosta(mosek.iparam.intpnt_basis) == mosek.prosta.prim_infeas:
            return "INFEASIBLE"
        if self.task.getprosta(mosek.soltype.bas) == mosek.prosta.dual_infeas or self.task.getprosta(mosek.iparam.intpnt_basis) == mosek.prosta.dual_infeas:
            return "UNBOUNDED"
        return "OTHER"

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
        self.run_simplex()

    def run_simplex(self) -> None:
        """Run simplex algorithm on the current model."""
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
        self.task.putdouparam(mosek.dparam.basis_tol_s, self.settings.optimalityTol)

    def _set_presolve(self) -> None:
        # turn off the presolve if setting.presolve is "off"
        if self.settings.presolve == "off":
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.off)
        else:
            self.task.putintparam(mosek.iparam.presolve_use, mosek.presolvemode.free)

    def _set_log(self) -> None:
        self.env.set_Stream(mosek.streamtype.log, _stream_printer)
        self.task.set_Stream(mosek.streamtype.log, _stream_printer)
        if self.settings.log_file != '':
            self.env.linkfiletostream(mosek.streamtype.log, self.settings.log_file, 1)
            self.task.linkfiletostream(mosek.streamtype.log, self.settings.log_file, 1)

    def _set_time_limit(self) -> None:
        self.task.putdouparam(mosek.dparam.optimizer_max_time, self.settings.timeLimit)


def _stream_printer(text):
    sys.stdout.write(text)
    sys.stdout.flush()
