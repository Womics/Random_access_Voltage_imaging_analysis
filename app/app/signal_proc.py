import numpy as np
from .deps import HAVE_SCIPY, spsig

def estimate_fs(t: np.ndarray) -> float:
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 1.0
    return 1.0 / dt

def butterworth_filter(y: np.ndarray, fs: float, ftype: str, order: int,
                       f1: float, f2: float) -> np.ndarray:
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for Butterworth filters.")
    y = np.asarray(y, dtype=float)
    nyq = 0.5 * fs
    if ftype == "lowpass":
        Wn = f1 / nyq
        b, a = spsig.butter(order, Wn, btype="low", analog=False)
    elif ftype == "highpass":
        Wn = f1 / nyq
        b, a = spsig.butter(order, Wn, btype="high", analog=False)
    elif ftype == "bandpass":
        lo = f1 / nyq
        hi = f2 / nyq
        b, a = spsig.butter(order, [lo, hi], btype="band", analog=False)
    else:
        return y
    return spsig.filtfilt(b, a, y, axis=0)

def zscore(y: np.ndarray) -> np.ndarray:
    mu = np.nanmean(y, axis=0, keepdims=True)
    sd = np.nanstd(y, axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (y - mu) / sd
