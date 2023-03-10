from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Output:
    x: Optional[np.ndarray] = None
    obj_val: Optional[float] = None
    runtime: Optional[float] = None
    iter_count: Optional[float] = None
    rcost: Optional[np.ndarray] = None
    vbasis: Optional[np.ndarray] = None
    cbasis: Optional[np.ndarray] = None
