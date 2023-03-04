from typing import Union

import numpy as np
import scipy


class MCFInput:
    """
    Information of the LP model for a general MCF problem:
             min      c^T x
             s.t.     A x  = b
                   0 <= x <= u
    """

    A: Union[scipy.sparse.csr_matrix, np.ndarray]
    b: np.ndarray
    c: np.ndarray
    u: np.ndarray

    def __init__(self,
                 A: Union[scipy.sparse.csr_matrix, np.ndarray],
                 b: np.ndarray,
                 c: np.ndarray,
                 u: np.ndarray) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.u = u


class OTInput:
    pass


class StandardLPInput:
    """
    Information of a standard form LP model:
             min  c^T x
             s.t. A x  = b
                    x >= 0
    """

    A: Union[scipy.sparse.csr_matrix, np.ndarray]
    b: np.ndarray
    c: np.ndarray

    def __init__(self,
                 A: Union[scipy.sparse.csr_matrix, np.ndarray],
                 b: np.ndarray,
                 c: np.ndarray) -> None:
        self.A = A
        self.b = b
        self.c = c
