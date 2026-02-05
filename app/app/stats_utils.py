# analyzer_app/stats_utils.py
from __future__ import annotations

import numpy as np

def mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))) / 0.6745) if x.size else 0.0
