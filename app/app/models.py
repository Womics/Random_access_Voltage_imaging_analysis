from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import numpy as np

@dataclass(frozen=True)
class RawModel:
    path: str
    t_raw: np.ndarray          # (N,)
    y_raw: np.ndarray          # (N, M)
    labels: List[str]
    dt: float

@dataclass(frozen=True)
class ProcessOptions:
    sel_idx: List[int]

    filter_type: str
    filter_order: int
    f1: float
    f2: float

    use_arpls: bool
    arpls_lam: float
    arpls_ratio: float
    arpls_itermax: int

    use_artifact: bool
    slope_pos: float
    slope_neg: float
    amp_pos: float
    amp_neg: float
    min_len: int
    merge_gap_samples: int
    calm_window: float
    calm_sigma_k: float
    mean_k: float
    max_extend: float
    post_silence: float
    extend_before: float
    extend_after: float
    merge_gap_sec: float

    norm: str
    baseline_n: int

    use_events: bool
    thr_z: float
    refractory_ms: float
    polarity: str
    events_on_normalized: bool

    # Artifact dialog override (full-precision blocks).
    # If provided and fingerprint matches, ProcWorker will use these blocks instead of auto-detection.
    artifact_override_blocks: Optional[List[Tuple[float, float]]] = None
    artifact_override_fingerprint: Optional[Tuple[Any, ...]] = None

@dataclass(frozen=True)
class ProcessedResult:
    t: np.ndarray
    y: np.ndarray
    disp: np.ndarray
    labels: List[str]
    baseline: Optional[np.ndarray]
    blocks: List[Tuple[float, float]]
    used_compressed: bool
    events: List[Tuple[int, float, float]]
    fs: float