import numpy as np
from .deps import HAVE_SCIPY, sp, spsolve

def mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))) / 0.6745) if x.size else 0.0

def arpls(y: np.ndarray, lam: float = 1e8, ratio: float = 1e-6, itermax: int = 20) -> np.ndarray:
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy is required for arPLS (scipy.sparse, spsolve).")
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 5:
        return np.zeros_like(y)

    D = sp.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), format='csc')
    H = (lam * (D @ D.transpose())).tocsc()
    w = np.ones(L)

    z = np.zeros_like(y)
    for _ in range(itermax):
        W = sp.diags(w, 0, format='csc')
        Z = (W + H).tocsc()
        z = spsolve(Z, w * y)
        d = y - z
        dn = d[d < 0]
        m = dn.mean() if dn.size else 0.0
        s = dn.std() if dn.size else 1.0
        if s == 0:
            s = 1.0
        wt = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        denom = np.linalg.norm(w) + 1e-12
        if np.linalg.norm(w - wt) / denom < ratio:
            w = wt
            break
        w = wt
    return np.asarray(z, dtype=float)