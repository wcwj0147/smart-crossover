from typing import Optional

import numpy as np


class Output:
    x: np.ndarray
    obj_val: float
    runtime: float
    iter_count: float

    def __init__(self,
                 x: Optional[np.ndarray] = None,
                 obj_val: Optional[float] = None,
                 runtime: Optional[float] = None,
                 iter_count: Optional[float] = None) -> None:
        self.x = x
        self.obj_val = obj_val
        self.runtime = runtime
        self.iter_count = iter_count
