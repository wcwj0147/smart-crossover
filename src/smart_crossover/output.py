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
    x: Optional[np.ndarray[float]] = None
    y: Optional[np.ndarray[float]] = None
    obj_val: Optional[float] = None
    runtime: Optional[float] = None
    iter_count: Optional[float] = None
    rcost: Optional[np.ndarray[float]] = None
    basis: Optional[Basis] = None
