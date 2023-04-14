import numpy as np
from scipy import sparse as sp

from smart_crossover.parameters import TOLERANCE_FOR_ARTIFICIAL_VARS, TOLERANCE_FOR_REDUCED_COSTS
from smart_crossover.formats import StandardLP
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

    def extend_by_bigM(self, bigM: float) -> None:
        """ Extend the current LP by adding artificial variables and a bigM term to the objective function. """
        self.lp.A = sp.vstack([self.lp.A, np.eye(len(self.lp.b))])
        self.lp.c = np.hstack([self.lp.c, np.ones(len(self.lp.b)) * bigM])
        self.lp.u = np.hstack([self.lp.u, np.inf * np.ones(len(self.lp.b))])
        self.artificial_vars = np.array(range(self.n, self.n + self.m), dtype=int)
        self.var_info['free'] = np.append(self.var_info['free'], np.array(range(self.n, self.n + self.m), dtype=np.int64))

    def fix_variables(self, ind_fix_to_low: np.ndarray, ind_fix_to_up: np.ndarray) -> None:
        """ Fix some variables to lower/upper bounds in the sub problem. """
        self.var_info['fix_low'] = ind_fix_to_low
        self.var_info['fix_up'] = ind_fix_to_up
        self.var_info['free'] = np.setdiff1d(range(len(self.lp.c)), np.append(ind_fix_to_low, ind_fix_to_up))
        self.var_info['fix'] = np.setdiff1d(range(len(self.lp.c)), self.var_info['free'])

    def add_free_variables(self, ind_free_new: np.ndarray) -> None:
        """ Add some variables to the free variables in the sub problem. """
        self.var_info['free'] = np.append(self.var_info['free'], ind_free_new)
        self.var_info['fix'] = np.setdiff1d(self.var_info['fix'], ind_free_new)
        self.var_info['fix_low'] = np.setdiff1d(self.var_info['fix_low'], ind_free_new)
        self.var_info['fix_up'] = np.setdiff1d(self.var_info['fix_up'], ind_free_new)

    def update_subproblem(self) -> None:
        """ Update the sub problem by the information of variables. """
        self.lp_sub = StandardLP(
            A=self.lp.A[:, self.var_info['free']],
            b=self.lp.b - self.lp.A[:, self.var_info['fix_up']] @ self.lp.u[self.var_info['fix_up']],
            c=self.lp.c[self.var_info['free']],
            u=self.lp.u[self.var_info['free']]
        )

    def set_basis(self, basis: Basis) -> None:
        """ Add a feasible basis for the current LP. """
        self.basis = basis

    def rescale_cost(self, factor: float) -> None:
        """ Rescale the objective function of the current LP. """
        self.lp.c = self.lp.c / factor
        self.c_rescaling_factor = factor

    def recover_x_from_sub_x(self, x_sub: np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        """ Recover the solution of the current LP from a solution of the sub problem. """
        x = np.zeros(self.lp.c.size)
        x[self.var_info['free']] = x_sub
        x[self.var_info['fix_up']] = self.lp.u[self.var_info['fix_up']]
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        """ Recover the basis of the current LP from a basis of the sub problem. """
        vbasis = -np.ones(self.lp.c.size, dtype=int)
        vbasis[self.var_info['free']] = basis_sub.vbasis
        vbasis[self.var_info['fix_up']] = -2
        cbasis = basis_sub.cbasis
        return Basis(vbasis, cbasis)

    def recover_obj_val(self, obj_val: float) -> float:
        """ Recover the objective value of the original LP from the objective value of the current LP. """
        return obj_val * self.c_rescaling_factor

    def get_reduced_cost_for_original_lp(self, y: np.ndarray) -> np.ndarray:
        """ Get the reduced cost for the current LP. """
        rcost = self.lp.c - self.lp.A.T @ y
        rcost[self.basis.vbasis == -2] = -rcost[self.basis.vbasis == -2]
        return rcost

    def check_optimality_condition(self, x: np.ndarray[np.float_], y: np.ndarray[np.float_]) -> bool:
        """ Check the optimality condition for the current LP and a pair of primal-dual solution (x, y). """
        artificial_vars_condition = np.all(x[self.artificial_vars] < TOLERANCE_FOR_ARTIFICIAL_VARS) if self.artificial_vars.size > 0 else True
        rcost_condition = np.all(self.get_reduced_cost_for_original_lp(y) >= -TOLERANCE_FOR_REDUCED_COSTS)
        return artificial_vars_condition and rcost_condition

    def solve_subproblem(self, solver: str = 'GRB') -> Output:
        """ Solve the sub problem. """
        return solve_lp(self.lp_sub, solver=solver, warm_start_basis=Basis(self.basis.vbasis[self.var_info['free']], self.basis.cbasis))
