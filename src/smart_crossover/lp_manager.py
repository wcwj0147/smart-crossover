import numpy as np
from scipy import sparse as sp

from smart_crossover.parameters import TOLERANCE_FOR_ARTIFICIAL_VARS, TOLERANCE_FOR_REDUCED_COSTS
from smart_crossover.formats import StandardLP, MinCostFlow
from smart_crossover.output import Basis, Output
from smart_crossover.solver_caller.utils import solve_lp


class LPManager:
    """
    A class to manage the LPs and its sub problems used in the smart crossover algorithm:

    Attributes:
        m: The number of constraints in the original LP.
        n: The number of variables in the original LP.
        lp: The current LP.
        lp_sub: A sub problem for the current LP. Usually, it is derived by fixing some variables, and getting a sub-matrix (sub-columns) of lp.A, which is called column-generation.
        var_info: A dictionary containing the information of variables in the sub problem (fixed to upper/lower bound or free).
        basis: The current basis for the current LP.
        artificial_vars: The indices of artificial variables in the current (extended / rescaled) LP.
        c_rescaling_factor: The factor used to rescale the objective function of the current LP.
    """

    m: int
    n: int
    lp: StandardLP
    var_info: dict[str, np.ndarray[np.int64]]
    lp_sub: StandardLP
    basis: Basis

    def __init__(self, lp: StandardLP) -> None:
        self.lp = lp
        self.m = self.lp.b.size
        self.n = self.lp.c.size
        self.var_info = {'free': np.array(range(self.n), dtype=np.int64)}
        self.artificial_vars = np.array([])
        self.c_rescaling_factor = None

    def extend_lp_bigM(self, bigM: float) -> None:
        self.lp.A = sp.vstack([self.lp.A, np.eye(len(self.lp.b))])
        self.lp.c = np.hstack([self.lp.c, np.ones(len(self.lp.b)) * bigM])
        self.lp.u = np.hstack([self.lp.u, np.inf * np.ones(len(self.lp.b))])
        self.artificial_vars = np.array(range(self.n, self.n + self.m), dtype=int)
        self.var_info['free'] = np.append(self.var_info['free'], np.array(range(self.n, self.n + self.m), dtype=np.int64))

    def fix_variables(self, ind_fix_to_low: np.ndarray, ind_fix_to_up: np.ndarray) -> None:
        self.var_info['fix_low'] = ind_fix_to_low
        self.var_info['fix_up'] = ind_fix_to_up
        self.var_info['free'] = np.setdiff1d(range(len(self.lp.c)), np.append(ind_fix_to_low, ind_fix_to_up))
        self.var_info['fix'] = np.setdiff1d(range(len(self.lp.c)), self.var_info['free'])

    def add_free_variables(self, ind_free_new: np.ndarray) -> None:
        self.var_info['free'] = np.append(self.var_info['free'], ind_free_new)
        self.var_info['fix'] = np.setdiff1d(self.var_info['fix'], ind_free_new)
        self.var_info['fix_low'] = np.setdiff1d(self.var_info['fix_low'], ind_free_new)
        self.var_info['fix_up'] = np.setdiff1d(self.var_info['fix_up'], ind_free_new)

    def update_subproblem(self) -> None:
        self.lp_sub = StandardLP(
            A=self.lp.A[:, self.var_info['free']],
            b=self.lp.b - self.lp.A[:, self.var_info['fix_up']] @ self.lp.u[self.var_info['fix_up']],
            c=self.lp.c[self.var_info['free']],
            u=self.lp.u[self.var_info['free']]
        )

    def add_basis(self, basis: Basis) -> None:
        self.basis = basis

    def rescale_cost(self, factor: float) -> None:
        self.lp.c = self.lp.c / factor
        self.c_rescaling_factor = factor

    def recover_x_from_sub_x(self, x_sub: np.ndarray) -> np.ndarray:
        x = np.zeros(self.lp.c.size)
        x[self.var_info['free']] = x_sub
        x[self.var_info['fix_up']] = self.lp.u[self.var_info['fix_up']]
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        vbasis = -np.ones(self.lp.c.size, dtype=int)
        vbasis[self.var_info['free']] = basis_sub.vbasis
        vbasis[self.var_info['fix_up']] = -2
        cbasis = basis_sub.cbasis
        return Basis(vbasis, cbasis)

    def recover_obj_val(self, obj_val: float) -> float:
        return obj_val * self.c_rescaling_factor

    def get_reduced_cost_for_original_lp(self, y: np.ndarray) -> np.ndarray:
        rcost = self.lp.c - self.lp.A.T @ y
        rcost[self.basis.vbasis == -2] = -rcost[self.basis.vbasis == -2]
        return rcost

    def get_values_for_artificial_vars(self, x: np.ndarray) -> np.ndarray:
        return x[self.artificial_vars]

    def check_optimality_condition(self, x: np.ndarray, y: np.ndarray) -> bool:
        artificial_vars_condition = np.all(self.get_values_for_artificial_vars(x) < TOLERANCE_FOR_ARTIFICIAL_VARS) if self.artificial_vars.size > 0 else True
        rcost_condition = np.all(self.get_reduced_cost_for_original_lp(y) >= -TOLERANCE_FOR_REDUCED_COSTS)
        return artificial_vars_condition and rcost_condition

    def solve_subproblem(self, solver: str = 'GRB') -> Output:
        return solve_lp(self.lp_sub, solver=solver, warm_start_basis=Basis(self.basis.vbasis[self.var_info['free']], self.basis.cbasis))


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

    def extend_mcf_bigM(self, bigM: float) -> None:
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

    def get_sorted_flows(self, x: np.ndarray) -> np.ndarray[int]:
        # Reverse large flow.
        mask_large_x = x > self.mcf.u / 2
        x_hat = x * (~mask_large_x) + self.mcf.u * mask_large_x - x * mask_large_x
        x_hat[(x < 0) | (x > self.mcf.u)] = 0
        A_bar = self.mcf.A.multiply(~mask_large_x) - self.mcf.A.multiply(mask_large_x)

        # Calculate flows indicators r_1.
        A_barplus = A_bar.maximum(sp.csc_matrix((self.m, self.n)))
        A_barminus = (-A_bar).maximum(sp.csc_matrix((self.m, self.n)))
        f_1 = A_barplus @ x_hat
        f_2 = A_barminus @ x_hat
        f = np.maximum(f_1, f_2)
        f_inv = 1 / f
        row, col, a = sp.find(A_bar)
        val = f_inv[row] * x_hat[col]
        r = sp.csc_matrix((val * a, (row, col)), shape=(self.m, self.n))
        r_1 = sp.csr_matrix.max(r.multiply(sp.csr_matrix.sign(r)), axis=0)
        r_1 = r_1.toarray().reshape((self.n,))
        # sort flows by indicators
        return np.argsort(-r_1)
