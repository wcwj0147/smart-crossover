from dataclasses import dataclass
from typing import Union

import numpy as np
import scipy.sparse as sp


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


@dataclass
class MinCostFlow(StandardLP):
    """
    Information of the LP model for a minimum cost flow problem:
             min      c^T x
             s.t.     A x  = b
                   0 <= x <= u
    """

    def __post_init__(self) -> None:
        if np.sum(self.b) != 0:
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

    s: np.ndarray[np.float64]
    d: np.ndarray[np.float64]
    M: Union[sp.csr_matrix, np.ndarray[np.float64]]

    def __post_init__(self) -> None:
        if np.sum(self.s) != np.sum(self.d):
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
            sp.kron(sp.eye(self.d.size).toarray(), np.ones((1, self.s.size))),
            -sp.kron(np.ones((1, self.d.size)), sp.eye(self.s.size).toarray())
        ])
        b = np.hstack([self.d, -self.s])
        return MinCostFlow(A=A, b=b, c=self.M.flatten(), u=np.inf(n))
