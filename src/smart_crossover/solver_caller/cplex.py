import re
import uuid
import datetime
from typing import Optional, Tuple

import cplex
import numpy as np
import scipy.sparse as sp

from smart_crossover import get_project_root
from smart_crossover.formats import MinCostFlow, StandardLP, GeneralLP
from smart_crossover.output import Basis
from smart_crossover.solver_caller.caller import SolverCaller, SolverSettings


class CplCaller(SolverCaller):
    def __init__(self, solver_settings: Optional[SolverSettings] = SolverSettings()) -> None:
        self.solver_name = "CPL"
        self.settings = solver_settings
        self.model = cplex.Cplex()
        self.runtime = 0
        self.iterations = None

    def read_model_from_file(self, path: str) -> None:
        self.model.read(path)

    def read_stdlp(self, stdlp: StandardLP) -> None:
        # Add variables
        A_coo = stdlp.A.tocoo()
        self.model.variables.add(obj=stdlp.c.tolist(), ub=stdlp.u.tolist(), names=['x_{}'.format(i) for i in range(stdlp.c.size)])

        a_rows = A_coo.row.tolist()
        a_cols = A_coo.col.tolist()
        a_vals = A_coo.data
        self.model.linear_constraints.add(rhs=stdlp.b, senses=['E'] * stdlp.b.size)
        self.model.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))

    def read_mcf(self, mcf: MinCostFlow) -> None:
        self.read_stdlp(mcf)

    def read_genlp(self, genlp: GeneralLP) -> None:
        A_coo = genlp.A.tocoo()
        self.model.variables.add(obj=genlp.c.tolist(), ub=genlp.u.tolist(), lb=genlp.l.tolist(), names=['x_{}'.format(i) for i in range(genlp.c.size)])

        sense_cpl = ['E' if sense == '=' else 'L' if sense == '<' else 'G' for sense in genlp.sense]
        a_rows = A_coo.row.tolist()
        a_cols = A_coo.col.tolist()
        a_vals = A_coo.data
        self.model.linear_constraints.add(rhs=genlp.b, senses=sense_cpl, names=['c_{}'.format(i) for i in range(genlp.b.size)])
        self.model.linear_constraints.set_coefficients(zip(a_rows, a_cols, a_vals))

    def get_A(self) -> sp.csr_matrix:
        num_rows = self.model.linear_constraints.get_num()
        num_cols = self.model.variables.get_num()

        all_constraints = self.model.linear_constraints.get_rows()

        row_indices = np.repeat(range(num_rows), [len(constraint.ind) for constraint in all_constraints])
        col_indices = np.concatenate([constraint.ind for constraint in all_constraints])
        values = np.concatenate([constraint.val for constraint in all_constraints])

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
        row_status = [s.basic if basis.cbasis[i] == 0 else s.at_lower_bound if basis.cbasis[i] == -1 else s.at_upper_bound if basis.cbasis[i] == -2 else s.free_nonbasic for i in range(len(basis.cbasis))]
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
        vbasis = np.array([0 if col_status[i] == s.basic else -1 if col_status[i] == s.at_lower_bound else -2 if col_status[i] == s.at_upper_bound else -3 for i in range(len(col_status))])
        cbasis = np.array([0 if row_status[i] == s.basic else -1 if row_status[i] == s.at_lower_bound else -2 if row_status[i] == s.at_upper_bound else -3 for i in range(len(row_status))])
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
        return round(self.iterations)

    def return_bar_iter_count(self) -> int:
        return self.model.solution.progress.get_num_barrier_iterations()

    def return_reduced_cost(self) -> np.ndarray:
        return np.array(self.model.solution.get_reduced_costs())

    def return_status(self) -> str:
        if self.model.solution.get_status_string() == 'optimal':
            return 'OPTIMAL'
        else:
            return 'OTHER'

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
        self._set_pricing()
        self._set_log()
        self._set_time_limit()
        self.model.solve()
        self.runtime, self.iterations = get_runtime_and_iterations_from_log(self.settings.log_file)

    def _set_tol(self) -> None:
        self.model.parameters.barrier.convergetol.set(self.settings.barrierTol)
        self.model.parameters.simplex.tolerances.optimality.set(self.settings.optimalityTol)

    def _set_presolve(self) -> None:
        if self.settings.presolve == "off":
            self.model.parameters.preprocessing.presolve.set(0)
        else:
            self.model.parameters.preprocessing.presolve.set(1)

    def _set_pricing(self) -> None:
        if self.settings.simplexPricing == 'PP':
            self.model.parameters.network.pricing.set(1)

    def _set_log(self) -> None:
        self.model.parameters.simplex.display.set(1)
        self.model.parameters.barrier.display.set(1)
        if self.settings.log_file != '':
            log_file = open(self.settings.log_file, 'a')  # Open the file in append mode
        else:
            self.settings.log_file = get_project_root() / 'results/logs/cplex/cplex_log.log'
            # open the file in rewrite mode
            log_file = open(self.settings.log_file, 'w+')
        self.model.set_results_stream(log_file)
        self.model.set_log_stream(log_file)

    def _set_time_limit(self) -> None:
        self.model.parameters.setTimeLimit = self.settings.timeLimit


def get_runtime_and_iterations_from_log(log_file_path: str):
    # Regular expression to match lines like
    # "Network time = 0.00 sec. (0.47 ticks)  Iterations = 838 (395)"
    pattern = re.compile(r"Network time = (\d+\.\d+) sec.*Iterations = (\d+)")

    runtime = None
    iterations = None

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                runtime = float(match.group(1))
                iterations = int(match.group(2))

    return runtime, iterations

