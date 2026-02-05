# analyzer_app/processing.py
from __future__ import annotations

import numpy as np

from .signal_proc import estimate_fs, butterworth_filter, zscore
from .baseline import arpls
from .artifact import (
    detect_artifact_blocks_strict,
    extend_artifact_until_calm_light,
    merge_blocks,
    apply_blocks,
)
from .events import detect_events_simple

def preprocess_until_artifact(t_raw: np.ndarray, y_raw: np.ndarray, opts: dict):
    """Apply selection + filter + arPLS. Returns (t, y_corr, baseline)."""
    t = np.asarray(t_raw, dtype=float).copy()
    y = np.asarray(y_raw, dtype=float).copy()

    fs = estimate_fs(t)

    ftype = opts.get("filter_type", "none")
    if ftype != "none":
        y = butterworth_filter(
            y, fs=fs, ftype=ftype,
            order=int(opts.get("filter_order", 3)),
            f1=float(opts.get("f1", 1.0)),
            f2=float(opts.get("f2", 50.0)),
        )

    baseline = None
    y_corr = y
    if bool(opts.get("use_arpls", False)):
        lam = float(opts.get("arpls_lam", 1e8))
        ratio = float(opts.get("arpls_ratio", 1e-6))
        it = int(opts.get("arpls_itermax", 20))
        baseline = np.zeros_like(y)
        for j in range(y.shape[1]):
            baseline[:, j] = arpls(y[:, j], lam=lam, ratio=ratio, itermax=it)
        y_corr = y - baseline

    return t, y_corr, baseline


def compute_artifact_blocks(t: np.ndarray, y_corr: np.ndarray, opts: dict):
    """Detect/extend/merge across columns -> global blocks."""
    all_blocks = []
    for j in range(y_corr.shape[1]):
        blocks = detect_artifact_blocks_strict(
            t, y_corr[:, j],
            slope_pos=float(opts.get("slope_pos", 5.0)),
            slope_neg=float(opts.get("slope_neg", 8.0)),
            amp_pos=float(opts.get("amp_pos", 3.0)),
            amp_neg=float(opts.get("amp_neg", 12.0)),
            min_len=int(opts.get("min_len", 10)),
            merge_gap=int(opts.get("merge_gap_samples", 2)),
        )
        blocks_ext = extend_artifact_until_calm_light(
            t, y_corr[:, j], blocks,
            calm_window=float(opts.get("calm_window", 1.0)),
            calm_sigma_k=float(opts.get("calm_sigma_k", 1.5)),
            mean_k=float(opts.get("mean_k", 2.0)),
            max_extend=float(opts.get("max_extend", 8.0)),
            post_silence=float(opts.get("post_silence", 0.5)),
            extend_before=float(opts.get("extend_before", 0.2)),
            extend_after=float(opts.get("extend_after", 0.3)),
        )
        all_blocks.append(blocks_ext)

    blocks_global = merge_blocks(all_blocks, merge_gap=float(opts.get("merge_gap_sec", 0.5)))
    return blocks_global


def apply_full_pipeline(model: dict, opts: dict):
    """Main processing used by worker."""
    t_raw = model["t_raw"]
    y_raw = model["y_raw"]
    labels = model["labels"]
    dt = model["dt"]

    sel = opts["sel_idx"]
    y = y_raw[:, sel]
    sel_labels = [labels[i] for i in sel]

    # 1) filter + arPLS
    t, y_corr, baseline = preprocess_until_artifact(t_raw, y, opts)

    # 2) artifact removal + compressed time
    blocks_global = []
    if bool(opts.get("use_artifact", False)):
        manual = opts.get("manual_blocks", None)
        if manual:
            blocks_global = [(float(a), float(b)) for (a, b) in manual]
        else:
            blocks_global = compute_artifact_blocks(t, y_corr, opts)

        t_keep, _ = apply_blocks(t, np.zeros_like(t), blocks_global)
        n_clean = len(t_keep)
        t_comp = np.arange(n_clean, dtype=float) * float(dt)

        if n_clean < 10:
            raise RuntimeError(
                f"artifact除去で残点数が少なすぎます (n_clean={n_clean}). "
                "閾値を緩めるか、artifact除去をOFFにしてください。"
            )

        y_new = np.zeros((n_clean, y_corr.shape[1]), dtype=float)
        for j in range(y_corr.shape[1]):
            _, y_drop = apply_blocks(t, y_corr[:, j], blocks_global)
            y_new[:, j] = y_drop
        t = t_comp
        y_corr = y_new

    # 3) normalization for detection/display
    disp = y_corr.copy()
    norm = opts.get("norm", "raw")
    if norm == "baseline":
        n = int(opts.get("baseline_n", 200))
        n = max(1, min(n, disp.shape[0]))
        disp = disp - np.nanmean(disp[:n, :], axis=0, keepdims=True)
    elif norm == "zscore":
        disp = zscore(disp)

    # 4) event detection
    events = []
    if bool(opts.get("use_events", False)):
        events = detect_events_simple(
            t, disp if bool(opts.get("events_on_normalized", True)) else y_corr,
            thr_z=float(opts.get("thr_z", 4.0)),
            refractory_ms=float(opts.get("refractory_ms", 20.0)),
            polarity=str(opts.get("polarity", "positive")),
        )

    return dict(
        t=t,
        y=y_corr,
        disp=disp,
        labels=sel_labels,
        baseline=baseline,
        blocks=blocks_global,
        used_compressed=bool(opts.get("use_artifact", False)),
        events=events,
        fs=estimate_fs(t),
    )
def preprocess_for_artifact_preview(
    t_raw: np.ndarray,
    y_raw: np.ndarray,
    filter_type: str = "none",
    filter_order: int = 3,
    f1: float = 1.0,
    f2: float = 50.0,
    use_arpls: bool = False,
    arpls_lam: float = 1e8,
    arpls_ratio: float = 1e-6,
    arpls_itermax: int = 20,
    preview_only_first_trace: bool = True,
):
    """
    Dialogプレビュー用の軽量前処理。
      - filter は全列まとめて実行（filtfiltはベクトル化されて速い）
      - arPLS は重いので、デフォルトで「最初の1本だけ」に限定
    Returns: (t, y_corr, baseline_or_None)
    """
    t = np.asarray(t_raw, dtype=float).copy()
    y = np.asarray(y_raw, dtype=float).copy()

    fs = estimate_fs(t)

    if filter_type != "none":
        y = butterworth_filter(y, fs=fs, ftype=filter_type, order=int(filter_order), f1=float(f1), f2=float(f2))

    baseline = None
    y_corr = y
    if use_arpls:
        if preview_only_first_trace:
            baseline = np.zeros((len(t), 1), dtype=float)
            baseline[:, 0] = arpls(y[:, 0], lam=float(arpls_lam), ratio=float(arpls_ratio), itermax=int(arpls_itermax))
            y_corr = y.copy()
            y_corr[:, 0] = y[:, 0] - baseline[:, 0]
        else:
            baseline = np.zeros_like(y)
            for j in range(y.shape[1]):
                baseline[:, j] = arpls(y[:, j], lam=float(arpls_lam), ratio=float(arpls_ratio), itermax=int(arpls_itermax))
            y_corr = y - baseline

    return t, y_corr, baseline


def compute_artifact_blocks_global(
    t: np.ndarray,
    y_corr: np.ndarray,
    params: dict,
    merge_gap_sec: float = 0.5,
    use_all_traces: bool = False,
):
    """
    Dialogプレビュー用のブロック推定。
      - use_all_traces=False の場合、最初の1本のみでブロックを作る（軽い）
      - use_all_traces=True の場合、従来通り全列を merge して global blocks を作る
    """
    y_use = y_corr if use_all_traces else y_corr[:, :1]
    opts = dict(params)
    opts["merge_gap_sec"] = float(merge_gap_sec)
    return compute_artifact_blocks(np.asarray(t, float), np.asarray(y_use, float), opts)
