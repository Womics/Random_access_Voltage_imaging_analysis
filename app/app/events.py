import numpy as np
from .signal_proc import estimate_fs, zscore

def detect_events_simple(t: np.ndarray, y: np.ndarray,
                         thr_z: float = 4.0,
                         refractory_ms: float = 20.0,
                         polarity: str = "positive"):
    fs = estimate_fs(t)
    refr = int(round((refractory_ms / 1000.0) * fs))
    refr = max(1, refr)

    yz = zscore(y)
    events = []
    for j in range(y.shape[1]):
        s = yz[:, j]
        if polarity == "positive":
            idx = np.where(s >= thr_z)[0]
        elif polarity == "negative":
            idx = np.where(s <= -thr_z)[0]
        else:
            idx = np.where(np.abs(s) >= thr_z)[0]

        if idx.size == 0:
            continue
        keep = [idx[0]]
        last = idx[0]
        for k in idx[1:]:
            if k - last >= refr:
                keep.append(k)
                last = k
        keep = np.array(keep, dtype=int)
        for k in keep:
            events.append((j, float(t[k]), float(y[k, j])))
    return events
