from typing import Union

import numpy as np
import scipy

from smart_crossover.output import Basis


class StandardLP:
    """
    Information of a standard form LP model:
             min      c^T x
             s.t.     A x = b
                   0 <= x <= u
    """

    A: Union[scipy.sparse.csr_matrix, np.ndarray]
    b: np.ndarray
    c: np.ndarray

    def __init__(self,
                 A: Union[scipy.sparse.csr_matrix, np.ndarray],
                 b: np.ndarray,
                 c: np.ndarray,
                 u: np.ndarray) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.u = u


class MinCostFlow(StandardLP):
    """
    Information of the LP model for a general MCF problem:
             min      c^T x
             s.t.     A x  = b
                   0 <= x <= u
    """

    def __init__(self,
                 A: Union[scipy.sparse.csr_matrix, np.ndarray],
                 b: np.ndarray,
                 c: np.ndarray,
                 u: np.ndarray) -> None:
        if np.sum(b) != 0:
            raise ValueError("The sum of the b array must be equal to 0.")
        super().__init__(A, b, c, u)


class OptTrans:
    pass


class SubLPManager:
    """
    Information of a sub LP problem:
    """

    lp: StandardLP
    ind_fix_to_low: np.ndarray
    ind_fix_to_up: np.ndarray
    ind_free: np.ndarray

    def __init__(self, lp: StandardLP,
                 ind_fix_to_low: np.ndarray, ind_fix_to_up: np.ndarray) -> None:
        self.ind_fix_to_low = ind_fix_to_low
        self.ind_fix_to_up = ind_fix_to_up
        self.ind_fix = list(set(ind_fix_to_up).union(set(ind_fix_to_low)))
        self.ind_free = np.array(list(set(range(len(lp.c))) - set(self.ind_fix)))
        self.lp_sub = StandardLP(
            A=lp.A[:, self.ind_free],
            b=lp.b - lp.A[:, ind_fix_to_up] @ lp.u[ind_fix_to_up],
            c=lp.c[self.ind_free],
            u=lp.u[self.ind_free]
        )

    def recover_x(self, x_sub: np.ndarray) -> np.ndarray:
        x = np.zeros(len(self.lp.c))
        x[self.ind_free] = x_sub
        x[self.ind_fix_to_up] = self.lp.u[self.ind_fix_to_up]
        return x

    def recover_basis(self, basis_sub: Basis) -> Basis:
        ...


class SubMCFManager(SubLPManager):
    ...
