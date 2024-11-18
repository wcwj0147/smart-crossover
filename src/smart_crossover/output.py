import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Basis:
    """Information of: vbasis and cbasis."""

    vbasis: np.ndarray
    cbasis: np.ndarray

    def __post_init__(self):
        self.vbasis = self.vbasis.astype(int)
        self.cbasis = self.cbasis.astype(int)


@dataclass(frozen=True)
class Output:
    """Output of solution of a LP model

    Attributes:
        x: Vertex solution of the primal.
        y: Vertex solution of the dual.
        x_bar: Interior-point solution of the primal.
        obj_val: Objective value.
        runtime: Runtime of the algorithm.
        iter_count: Number of iterations.
        bar_iter_count: Number of barrier iterations.
        rcost: Reduced costs.
        basis: Basis information.
        status: Status of the solution.

    """

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    x_bar: Optional[np.ndarray] = None
    obj_val: Optional[float] = None
    runtime: Optional[datetime.timedelta] = None
    iter_count: Optional[float] = None
    bar_iter_count: Optional[int] = None
    rcost: Optional[np.ndarray] = None
    basis: Optional[Basis] = None
    status: Optional[str] = None

    def __str__(self) -> str:
        return (f"Output(obj_val={self.obj_val}, "
                f"runtime={self.runtime}, "
                f"iter_count={self.iter_count}, "
                f"bar_iter_count={self.bar_iter_count})")
