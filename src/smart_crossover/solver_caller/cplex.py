import datetime
from typing import Optional, Tuple

import cplex
import numpy as np
import scipy.sparse as sp

from smart_crossover.formats import MinCostFlow, StandardLP, GeneralLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class CplCaller(SolverCaller):
    def __init__(self, solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "cplex"
        self.settings = solver_settings
        self.model = cplex.Cplex()
        self.runtime = 0

    def read_model_from_file(self, path: str) -> None:
        self.model.read(path)

    def read_stdlp(self, stdlp: StandardLP) -> None:
        # Add variables
        stdlp.A = stdlp.A.tocsr()
        self.model.variables.add(obj=stdlp.c.tolist(), ub=stdlp.u.tolist(), names=['x_{}'.format(i) for i in range(stdlp.c.size)])

        # Precompute variable names
        var_names = ['x_{}'.format(i) for i in range(stdlp.A.shape[1])]
        constraint_names = ['Ax_{}'.format(i) for i in range(stdlp.A.shape[0])]
        nonzero_ind = [stdlp.A.indices[stdlp.A.indptr[i]:stdlp.A.indptr[i + 1]] for i in range(stdlp.A.shape[0])]

        # Add constraints
        self.model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=[var_names[j] for j in nonzero_ind[i]], val=[stdlp.A[i, j] for j in nonzero_ind[i]])
                for i in range(stdlp.A.shape[0])],
            senses=['E' for _ in range(stdlp.A.shape[0])],
            rhs=stdlp.b,
            names=constraint_names
        )

    def read_mcf(self, mcf: MinCostFlow) -> None:
        self.read_stdlp(mcf)

    def read_genlp(self, genlp: GeneralLP) -> None:
        genlp.A = genlp.A.tocsr()
        self.model.variables.add(obj=genlp.c.tolist(), ub=genlp.u.tolist(), lb=genlp.l.tolist(), names=['x_{}'.format(i) for i in range(genlp.c.size)])

        # Precompute variable names
        var_names = ['x_{}'.format(i) for i in range(genlp.A.shape[1])]
        constraint_names = ['Ax_{}'.format(i) for i in range(genlp.A.shape[0])]
        nonzero_ind = [genlp.A.indices[genlp.A.indptr[i]:genlp.A.indptr[i + 1]] for i in range(genlp.A.shape[0])]

        # Add constraints
        self.model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=[var_names[j] for j in nonzero_ind[i]], val=[genlp.A[i, j] for j in nonzero_ind[i]])
                for i in range(genlp.A.shape[0])],
            senses=['E' if genlp.sense[i] == '=' else 'L' if genlp.sense[i] == '<' else 'G' for i in range(genlp.A.shape[0])],
            rhs=genlp.b,
            names=constraint_names
        )

    def get_A(self) -> sp.csr_matrix:
        num_rows = self.model.linear_constraints.get_num()
        num_cols = self.model.variables.get_num()

        row_indices = []
        col_indices = []
        values = []

        for i in range(num_rows):
            sparse_pair = self.model.linear_constraints.get_rows(i)
            row_indices.extend([i] * len(sparse_pair.ind))
            col_indices.extend(sparse_pair.ind)
            values.extend(sparse_pair.val)

        A_coo = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_rows, num_cols))

        return A_coo.tocsr()

    def get_b(self) -> np.ndarray:
        return np.array(self.model.linear_constraints.get_rhs())

    def get_sense(self) -> np.ndarray:
        sense = np.array(self.model.linear_constraints.get_senses())
        sense = np.array(['=' if s == 'E' else '<' if s == 'L' else '>' for s in sense])
        return sense

    def get_c(self) -> np.ndarray:
        return np.array(self.model.objective.get_linear())

    def get_l(self) -> np.ndarray:
        return np.array(self.model.variables.get_lower_bounds())

    def get_u(self) -> np.ndarray:
        return np.array(self.model.variables.get_upper_bounds())

    def add_warm_start_basis(self,
                             basis: Basis) -> None:
        s = self.model.start.status
        col_status = [s.basic if basis.vbasis[i] == 0 else s.at_lower_bound if basis.vbasis[i] == -1 else s.at_upper_bound if basis.vbasis[i] == -2 else s.free_nonbasic for i in range(len(basis.vbasis))]
        row_status = [s.basic if basis.cbasis[i] == 0 else s.at_lower_bound for i in range(len(basis.cbasis))]
        self.model.start.set_start(col_status=col_status, row_status=row_status, col_primal=[], row_primal=[], col_dual=[], row_dual=[])

    def add_warm_start_solution(self,
                                start_solution: Tuple[np.ndarray, np.ndarray]):
        self.model.start.set_start(col_primal=start_solution[0], row_primal=start_solution[1], col_status=[], row_status=[], col_dual=[], row_dual=[])

    def return_basis(self) -> Optional[Basis]:
        if self.model.parameters.barrier.crossover.get() == -1:
            return None
        s = self.model.start.status
        basis_status = self.model.solution.basis.get_basis()
        col_status = basis_status[0]
        row_status = basis_status[1]
        vbasis = np.array([s.basic if col_status[i] == 0 else s.at_lower_bound if col_status[i] == -1 else s.at_upper_bound if col_status[i] == -2 else s.free_nonbasic for i in range(len(col_status))])
        cbasis = np.array([s.basic if row_status[i] == 0 else s.at_lower_bound for i in range(len(row_status))])
        return Basis(vbasis=vbasis, cbasis=cbasis)

    def return_x(self) -> np.ndarray:
        return np.array(self.model.solution.get_values())

    def return_y(self) -> np.ndarray:
        return np.array(self.model.solution.get_dual_values())

    def return_barx(self) -> Optional[np.ndarray]:
        # assert the crossover is off, to be checked.
        if self.model.parameters.barrier.crossover.get() == -1:
            return None
        return np.array(self.model.solution.get_values())

    def return_obj_val(self) -> float:
        return self.model.solution.get_objective_value()

    def return_runtime(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.runtime)

    def return_iter_count(self) -> int:
        return round(self.model.solution.progress.get_num_iterations())

    def return_bar_iter_count(self) -> int:
        return self.model.solution.progress.get_num_barrier_iterations()

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.model.solution.get_reduced_costs())

    def run_default(self) -> None:
        """Run default algorithm on the current model."""
        # set the method to automatic
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.auto)
        self._run()

    def run_barrier(self) -> None:
        """Run barrier algorithm on the current model, crossover on."""
        # set the method to barrier
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.barrier)
        self._run()

    def run_barrier_no_crossover(self) -> None:
        """Run barrier algorithm on the current model, crossover off."""
        # set the method to barrier
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.barrier)
        # set the crossover to off
        self.model.parameters.barrier.crossover.set(-1)
        self._run()

    def run_primal_simplex(self) -> None:
        """Run simplex/network simplex algorithm on the current model."""
        # set the method to simplex
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.primal)
        self._run()

    def run_dual_simplex(self) -> None:
        """Run dual simplex algorithm on the current model."""
        # set the method to dual
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.dual)
        self._run()

    def run_network_simplex(self) -> None:
        """Run simplex/network simplex algorithm on the current model."""
        # set the method to network
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.network)
        self._run()

    def run_simplex(self) -> None:
        self.model.parameters.lpmethod.set(self.model.parameters.lpmethod.values.auto)
        self._run()

    def reset_model(self) -> None:
        """Reset the model to the original state."""
        self.model = cplex.Cplex()

    def _run(self) -> None:
        """Run the solver with the current settings."""
        self._set_tol()
        self._set_presolve()
        self._set_log()
        self._set_time_limit()
        start = self.model.get_time()
        self.model.solve()
        end = self.model.get_time()
        self.runtime = end - start

    def _set_tol(self) -> None:
        self.model.parameters.barrier.convergetol.set(self.settings.barrierTol)
        self.model.parameters.simplex.tolerances.optimality.set(self.settings.optimalityTol)

    def _set_presolve(self) -> None:
        if self.settings.presolve == "off":
            self.model.parameters.preprocessing.presolve.set(0)
        else:
            self.model.parameters.preprocessing.presolve.set(1)

    def _set_log(self) -> None:
        self.model.parameters.simplex.display.set(2)
        self.model.parameters.barrier.display.set(2)

    def _set_time_limit(self) -> None:
        self.model.parameters.setTimeLimit = self.settings.timeLimit
