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
    Information of the LP model for a general MCF problem:
             min      c^T x
             s.t.     A x  = b
                   0 <= x <= u
    """

    def __post_init__(self) -> None:
        if np.sum(self.b) != 0:
            raise ValueError("The sum of the b array must be equal to 0.")


@dataclass
class OptTrans:
    pass
