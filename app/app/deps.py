# Optional deps (scipy, matplotlib, pyqtgraph)

HAVE_SCIPY = False
HAVE_PG = False
HAVE_MPL = False

try:
    import scipy.signal as spsig
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    HAVE_SCIPY = True
except Exception:
    spsig = None
    sp = None
    spsolve = None

try:
    import pyqtgraph as pg
    HAVE_PG = True
except Exception:
    pg = None

try:
    import matplotlib
    matplotlib.use("Agg")  # export-only
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAVE_MPL = True
except Exception:
    plt = None
    FuncAnimation = None
    PillowWriter = None
