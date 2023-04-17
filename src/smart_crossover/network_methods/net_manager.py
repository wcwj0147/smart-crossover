from typing import Tuple, runtime_checkable, Protocol

import numpy as np
from scipy import sparse as sp

from smart_crossover.formats import MinCostFlow, StandardLP, OptTransport
from smart_crossover.lp_methods.lp_manager import LPManager
from smart_crossover.output import Basis, Output
from smart_crossover.parameters import TOLERANCE_FOR_ARTIFICIAL_VARS, TOLERANCE_FOR_REDUCED_COSTS
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.solving import solve_mcf


@runtime_checkable
class NetworkManager(Protocol):
    """ A protocol for network problem managers, including MinCostFlow and OptimalTransport. """

    m: int
    n: int
    basis: Basis

    def get_sorted_flows(self, x: np.ndarray[np.float_]) -> Tuple[np.ndarray[np.int_], np.ndarray[np.float_]]:
        ...

    def recover_x_from_sub_x(self, x_sub: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        ...

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        ...

    def solve_subproblem(self, solver: str, solver_settings: SolverSettings) -> Output:
        ...

    def recover_obj_val(self, obj_val: float) -> float:
        ...

    def check_optimality_condition(self, x: np.ndarray[np.float_], y: np.ndarray[np.float_]) -> bool:
        ...

    def add_free_variables(self, ind_free: np.ndarray) -> None:
        ...

    def update_subproblem(self):
        ...

    def set_basis(self, basis: Basis) -> None:
        ...


class MCFManager(LPManager):
    """A class to manage the MCF problem and its subproblem in network crossover algorithms.

    Attributes:
        mcf (MinCostFlow): The current MCF problem.
        mcf_sub (MinCostFlow): The subproblem of the current MCF problem.

    """

    mcf: MinCostFlow

    def __init__(self, mcf: MinCostFlow) -> None:
        super().__init__(mcf)
        self.mcf = mcf

    @property
    def lp(self) -> StandardLP:
        return self.mcf

    @lp.setter
    def lp(self, value: StandardLP) -> None:
        if isinstance(value, MinCostFlow):
            self.mcf = value
        else:
            raise ValueError("Expected a MinCostFlow instance")

    def extend_by_bigM(self, bigM: float) -> None:
        """ Extend the MCF problem by the bigM method. """
        mask_fix_up = np.zeros(self.n, dtype=bool)
        mask_fix_up[self.var_info['fix_up']] = True
        b_true = self.mcf.b - self.mcf.A.multiply(mask_fix_up) @ (self.mcf.u * mask_fix_up)
        b_sign = np.sign(b_true)
        b_sign[b_sign == 0] = 1
        c_1 = np.concatenate([self.mcf.c, bigM * np.ones(self.m)])
        u_1 = np.concatenate([self.mcf.u, np.Inf * np.ones(self.n)])
        A_1 = sp.hstack((self.mcf.A, sp.diags(b_sign)))
        A_1 = sp.vstack((A_1, sp.csr_matrix(np.concatenate([np.zeros(self.n), -b_sign]))))
        A_1 = A_1.tocsr()
        b_1 = np.concatenate([self.mcf.b, np.array([0])])
        self.mcf = MinCostFlow(A_1, b_1, c_1, u_1)
        self.artificial_vars = np.array(range(self.n, self.n + self.m), dtype=int)
        self.var_info['free'] = np.append(self.var_info['free'], np.array(range(self.n, self.n + self.m), dtype=np.int64))

    def get_sorted_flows(self, x: np.ndarray[np.float_]) -> Tuple[np.ndarray[np.int_], np.ndarray[np.float_]]:
        """ Get the sorted flows by calculating flow indicators. """
        # Reverse large flow.
        mask_large_x = x > self.mcf.u / 2
        x_hat = x * (~mask_large_x) + self.mcf.u * mask_large_x - x * mask_large_x
        x_hat[(x < 0) | (x > self.mcf.u)] = 0
        A_bar = self.mcf.A.multiply(~mask_large_x) - self.mcf.A.multiply(mask_large_x)

        # Calculate flows indicators (use r to represent at the first stage).
        A_barplus = A_bar.maximum(sp.csc_matrix((self.m, self.n)))
        A_barminus = (-A_bar).maximum(sp.csc_matrix((self.m, self.n)))
        f_1 = A_barplus @ x_hat
        f_2 = A_barminus @ x_hat
        f = np.maximum(f_1, f_2)
        f_inv = np.divide(1, f, out=np.zeros_like(f), where=f != 0)
        row, col, a = sp.find(A_bar)
        val = f_inv[row] * x_hat[col]
        r = sp.csc_matrix((val * a, (row, col)), shape=(self.m, self.n))
        r_1 = sp.csr_matrix.max(r.multiply(sp.csr_matrix.sign(r)), axis=0)
        flow_indicators = r_1.toarray().reshape((self.n,))
        # sort flows by indicators
        return np.argsort(flow_indicators)[::-1], flow_indicators

    def set_initial_basis(self) -> None:
        vbasis = np.concatenate((-np.ones(self.n), np.zeros(self.m)))
        vbasis[self.var_info['fix_up']] = -2
        cbasis = np.concatenate([-np.ones(self.m), np.zeros(1)])
        self.set_basis(Basis(vbasis, cbasis))


class OTManager:
    """A class to manage the OT problem and its subproblem in network crossover algorithms.

    Attributes:
        ot (OptimalTransport): The current OT problem.
        ot_sub (OptimalTransport): The subproblem of the current OT problem.
        m (int): The number of constraints in the original OT.
        n (int): The number of variables in the original OT.
        mask_sub_ot (np.ndarray[np.bool_]): The mask of the subproblem of the current OT problem.
        basis (Basis): A feasible basis for the current OT.
        mcf (MinCostFlow): The MCF form corresponding to the current OT.

    """

    ot: OptTransport
    m: int
    n: int
    m_ext: int
    n_ext: int
    mask_sub_ot: np.ndarray[np.bool_]
    basis: Basis
    artificial_vars: np.ndarray[np.int64]
    mcf: MinCostFlow

    def __init__(self, ot: OptTransport) -> None:
        self.ot = ot
        self.m = ot.s.size + ot.d.size
        self.n = ot.s.size * ot.d.size
        self.mask_sub_ot = np.zeros(self.n, dtype=bool)
        self.mcf = ot.to_MCF()
        self.artificial_vars = np.array([])

    def get_X(self, x: np.ndarray) -> np.ndarray:
        """ Get X from the vector x, such that X[i][j] = the flow from source i to destination j. """
        return x.reshape((self.ot.s.size, self.ot.d.size))

    def get_sorted_flows(self, x: np.ndarray) -> Tuple[np.ndarray[np.int_], np.ndarray[np.float_]]:
        """ Get the sorted flows by calculating flow indicators. """
        X = self.get_X(x)
        flow_indicators = np.maximum(X / self.ot.s.reshape((self.ot.s.size, 1)), X / self.ot.d.reshape((1, self.ot.d.size)))
        return np.argsort(flow_indicators.flatten())[::-1], flow_indicators.flatten()

    def extend_by_bigM(self, bigM: float) -> None:
        """ Extend the OT problem with bigM method. """
        s_appended = np.append(self.ot.s, np.sum(self.ot.d))
        d_appended = np.append(self.ot.d, np.sum(self.ot.s))
        M_appended = np.vstack([
            np.hstack([self.ot.M, bigM * bigM * np.ones((self.ot.s.size, 1))]),
            np.hstack([bigM * np.ones((1, self.ot.d.size)), np.array([[0]])])
        ])
        self.mask_sub_ot = np.vstack([
            np.hstack([np.zeros((self.ot.s.size, self.ot.d.size),  dtype=np.bool_), np.ones((self.ot.s.size, 1), dtype=np.bool_)]),
            np.hstack([np.ones((1, self.ot.d.size), dtype=np.bool_), np.array([[1]], dtype=np.bool_)]),
        ])
        # self.artificial_vars all the variables in the self.mask_sub_ot such that it is 1:
        self.artificial_vars = np.where(self.mask_sub_ot.ravel())[0]
        self.ot = OptTransport(s_appended, d_appended, M_appended)
        self.mcf = self.ot.to_MCF()

    def add_free_variables(self, ind_free: np.ndarray) -> None:
        """ Add free variables in the sub OT problem. """
        if self.artificial_vars.size > 0:
            mask_on_original_problem = self.mask_sub_ot[:-1, :-1]
            rows, cols = np.unravel_index(ind_free, mask_on_original_problem.shape)
            mask_on_original_problem[rows, cols] = True
            self.mask_sub_ot[:-1, :-1] = mask_on_original_problem
        else:
            self.mask_sub_ot[ind_free.ravel()] = True

    def set_basis(self, basis: Basis) -> None:
        """ Add a feasible basis to the current OT problem. """
        self.basis = basis

    def recover_x_from_sub_x(self, x_sub: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        x = np.zeros(self.ot.s.size * self.ot.d.size)
        x[self.mask_sub_ot.ravel()] = x_sub
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        vbasis = -np.ones(self.ot.s.size * self.ot.d.size)
        vbasis[self.mask_sub_ot.ravel()] = basis_sub.vbasis
        return Basis(vbasis, basis_sub.cbasis)

    def get_sub_problem(self) -> MinCostFlow:
        mcf = self.ot.to_MCF()
        mcf.A = mcf.A.tocsc()[:, self.mask_sub_ot.ravel()]
        mcf.c = self.ot.M.flatten()[self.mask_sub_ot.ravel()]
        mcf.u = mcf.u[self.mask_sub_ot.ravel()]
        return mcf

    def solve_subproblem(self, solver: str, solver_settings: SolverSettings) -> Output:
        return solve_mcf(self.get_sub_problem(), solver=solver, warm_start_basis=Basis(self.basis.vbasis[self.mask_sub_ot.ravel()], self.basis.cbasis), presolve=solver_settings.presolve)

    def recover_obj_val(self, obj_val):
        return obj_val

    def get_reduced_cost_for_original_OT(self, y: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        return self.mcf.c - self.mcf.A.T @ y

    def check_optimality_condition(self, x: np.ndarray[np.float_], y: np.ndarray[np.float_]) -> bool:
        artificial_vars_condition = np.all(x[self.artificial_vars][:-1] < TOLERANCE_FOR_ARTIFICIAL_VARS) if self.artificial_vars.size > 0 else True
        rcost_condition = np.all(self.get_reduced_cost_for_original_OT(y) >= -TOLERANCE_FOR_REDUCED_COSTS)
        return artificial_vars_condition and rcost_condition

    def update_subproblem(self):
        pass

    def set_initial_basis(self) -> None:
        vbasis = -np.ones(self.ot.s.size * self.ot.d.size)
        vbasis[self.artificial_vars] = 0
        cbasis = np.concatenate([-np.ones(self.m + 1), np.zeros(1)])
        self.basis = Basis(vbasis, cbasis)
