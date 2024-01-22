from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Optional

import numpy as np
import scipy.sparse as sp


@dataclass
class GeneralLP:
    """
    Information of a Gurobi presolved LP model:
                min      c^T x
                s.t.     A x (sense) b
                        l <= x <= u
    """
    A: Union[sp.csr_matrix, np.ndarray]
    b: np.ndarray
    c: np.ndarray
    l: np.ndarray
    u: np.ndarray
    sense: np.ndarray
    name: str = "lp_instance"

    def __post_init__(self) -> None:
        assert np.all(np.logical_or(self.sense == '=', self.sense == '<'))

    def get_free_ind(self) -> np.ndarray:
        return np.where(np.logical_and(self.l == -np.inf, self.u == np.inf))[0]

    def get_nonfree_ind(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.get_standard_c().size), self.get_free_ind())

    def get_free_var_matrix(self) -> sp.csr_matrix:
        return self.A[:, self.get_free_ind()]

    def get_nonfree_var_matrix(self) -> sp.csr_matrix:
        return self.get_standard_A().tocsr()[:, self.get_nonfree_ind()]

    def get_standard_A(self) -> sp.csr_matrix:
        """ Added slack variables to the original A matrix, such that the LP can be transfered in the form A_new x = b. """
        m, n = self.A.shape
        A_new = sp.hstack([self.A, sp.eye(m, format="csr")[:, np.where(self.sense == "<")[0]]])
        return A_new.tocsr()

    def get_standard_c(self) -> np.ndarray:
        return np.hstack([self.c, np.zeros(np.sum(self.sense == "<"))])

    def get_standard_x(self, x: np.ndarray) -> np.ndarray:
        """ Added slack variables to the original variable vector, which may represent a cost vector or a solution vector."""
        assert np.all(np.logical_or(self.sense == '=', self.sense == '<'))
        slack_ind = np.where(self.sense == "<")[0]
        return np.hstack([x, self.b[slack_ind] - self.A[slack_ind, :] @ x])

    def get_dual_slack(self, y: np.ndarray) -> np.ndarray:
        return self.c - self.A.transpose() @ y

    def get_primal_slack(self, x: np.ndarray) -> np.ndarray:
        return self.b - self.A @ x

    def copy(self) -> GeneralLP:
        return GeneralLP(self.A.copy(), self.b.copy(), self.c.copy(), self.l.copy(), self.u.copy(), self.sense.copy(), self.name)


@dataclass
class StandardLP:
    """
    Information of a standard form LP model:
             min      c^T x
             s.t.     A x = b
                   0 <= x <= u
    """

    A: Union[sp.csr_matrix, np.ndarray]
    b: np.ndarray
    c: np.ndarray
    u: np.ndarray
    name: str = "lp_instance"
    l: Optional[np.ndarray] = None  # Entries of l can only be 0 or -inf (free variables).

    def __post_init__(self) -> None:
        if self.l is None:
            self.l = np.zeros_like(self.u)


@dataclass
class MinCostFlow(StandardLP):
    """
    Information of the LP model for a minimum cost flow problem:
             min      c^T x
             s.t.     A x  = b
                   0 <= x <= u
    """

    name: str = "mcf_instance"

    def __post_init__(self) -> None:
        if self.l is None:
            self.l = np.zeros_like(self.u)
        self.A = self.A.tocsr()
        if not np.isclose(np.sum(self.b), 0, atol=1e-8):
            raise ValueError("The sum of the b array must be equal to 0.")


@dataclass
class OptTransport:
    """
    Information of an optimal transport problem:
    We use a different format from the LP / MCF format to make full use of the special structure of Optimal Transport.

    Attributes:
        s: The suppliers distribution.
        d: The demanders distribution.
        M: The cost matrix; M[i, j]: cost transferring a unit from s[i] to d[j].

    """

    s: np.ndarray
    d: np.ndarray
    M: Union[sp.csr_matrix, np.ndarray]
    name: str = "ot_instance"

    def __post_init__(self) -> None:
        if not np.isclose(np.sum(self.s), np.sum(self.d), atol=1e-8):
            raise ValueError("The sum of the s and d arrays must be the same.")

    def to_MCF(self) -> MinCostFlow:
        """
        Convert the optimal transport problem to a minimum cost flow problem.

        Returns:
            The minimum cost flow problem.
        """
        m = self.s.size + self.d.size
        n = self.s.size * self.d.size
        A = sp.vstack([
            -sp.kron(sp.eye(self.s.size).toarray(), np.ones((1, self.d.size))),
            sp.kron(np.ones((1, self.s.size)), sp.eye(self.d.size).toarray())
        ])
        b = np.hstack([-self.s, self.d])
        return MinCostFlow(A=A, b=b, c=self.M.flatten(), u=np.full(n, np.inf))
