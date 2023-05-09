import numpy as np
from typing import Dict

from smart_crossover.formats import StandardLP, GeneralLP
from smart_crossover.output import Basis


class LPManager:
    """
    A class to manage the LPs and its sub problems used in the smart crossover algorithm:

    Attributes:
        m: The number of constraints in the original LP.
        n: The number of variables in the original LP.
        lp: The current LP.
        lp_sub: A sub problem for the current LP. Usually, it is derived by fixing some variables, and getting a sub-matrix (sub-columns) of lp.A, which is called column-generation.
        var_info: A dictionary containing the information of variables in the sub problem (fixed to upper/lower bound or non_fix).
        basis: The current basis for the current LP.
        artificial_vars: The indices of artificial variables in the current (extended / rescaled) LP.
        c_rescaling_factor: The factor used to rescale the objective function of the current LP.
    """

    m: int
    n: int
    lp: GeneralLP
    var_info: Dict[str, np.ndarray]
    lp_sub: GeneralLP
    basis: Basis

    def __init__(self, lp: GeneralLP) -> None:
        self.lp = lp
        self.m = self.lp.b.size
        self.n = self.lp.c.size
        self.var_info = {'non_fix': np.array(range(self.n), dtype=np.int64)}

    def fix_variables(self, ind_fix_to_low: np.ndarray, ind_fix_to_up: np.ndarray) -> None:
        """ Fix some variables to lower/upper bounds in the sub problem. """
        self.var_info['fix_low'] = ind_fix_to_low
        self.var_info['fix_up'] = ind_fix_to_up
        self.var_info['non_fix'] = np.setdiff1d(range(self.n), np.append(ind_fix_to_low, ind_fix_to_up))
        self.var_info['fix'] = np.setdiff1d(range(self.n), self.var_info['non_fix'])

    def update_subproblem(self) -> None:
        """ Update the sub problem by the information of variables. """
        self.lp_sub = GeneralLP(
            A=self.lp.A[:, self.var_info['non_fix']],
            b=self.lp.b - self.lp.A[:, self.var_info['fix_up']] @ self.lp.u[self.var_info['fix_up']] + self.lp.A[:, self.var_info['fix_low']] @ self.lp.l[self.var_info['fix_low']],
            c=self.lp.c[self.var_info['non_fix']],
            l=self.lp.l[self.var_info['non_fix']],
            u=self.lp.u[self.var_info['non_fix']],
            sense=self.lp.sense
        )

    def recover_x_from_sub_x(self, x_sub: np.ndarray) -> np.ndarray:
        """ Recover the solution of the current LP from a solution of the sub problem. """
        x = np.zeros(self.lp.c.size)
        x[self.var_info['non_fix']] = x_sub
        x[self.var_info['fix_up']] = self.lp.u[self.var_info['fix_up']]
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        """ Recover the basis of the current LP from a basis of the sub problem. """
        vbasis = -np.ones(self.lp.c.size, dtype=int)
        vbasis[self.var_info['non_fix']] = basis_sub.vbasis
        vbasis[self.var_info['fix_up']] = -2
        cbasis = basis_sub.cbasis
        return Basis(vbasis, cbasis)
