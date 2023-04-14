import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Basis:
    """Information of: vbasis and cbasis."""

    vbasis: np.ndarray[np.float_]
    cbasis: np.ndarray[np.float_]

    def __post_init__(self):
        self.vbasis = self.vbasis.astype(int)
        self.cbasis = self.cbasis.astype(int)


@dataclass(frozen=True)
class Output:
    x: Optional[np.ndarray[np.float_]] = None
    y: Optional[np.ndarray[np.float_]] = None
    x_bar: Optional[np.ndarray[np.float_]] = None
    obj_val: Optional[float] = None
    runtime: Optional[datetime.timedelta] = None
    iter_count: Optional[float] = None
    bar_iter_count: Optional[int] = None
    rcost: Optional[np.ndarray[float]] = None
    basis: Optional[Basis] = None
