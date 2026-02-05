import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from .deps import HAVE_SCIPY
from .models import RawModel, ProcessOptions, ProcessedResult
from .signal_proc import estimate_fs, butterworth_filter, zscore
from .baseline import arpls
from .artifact import (
    detect_artifact_blocks_strict,
    extend_artifact_until_calm_light,
    merge_blocks,
    apply_blocks,
)
from .events import detect_events_simple


def _artifact_fingerprint_from_opts(opts: ProcessOptions) -> tuple:
    return (
        tuple(opts.sel_idx),
        opts.filter_type, int(opts.filter_order), float(opts.f1), float(opts.f2),
        bool(opts.use_arpls), float(opts.arpls_lam), float(opts.arpls_ratio), int(opts.arpls_itermax),
    )


class ProcWorker(QObject):
    finished = Signal(object)   # ProcessedResult
    failed = Signal(str)

    def __init__(self, model: RawModel, opts: ProcessOptions):
        super().__init__()
        self.model = model
        self.opts = opts

    @Slot()
    def run(self):
        try:
            t_raw = self.model.t_raw
            y_raw = self.model.y_raw
            labels = self.model.labels
            dt = self.model.dt

            sel = list(self.opts.sel_idx)
            y = y_raw[:, sel]
            sel_labels = [labels[i] for i in sel]
            t = t_raw.copy()

            fs = estimate_fs(t)

            # Butterworth filter (before baseline/artifact)
            if self.opts.filter_type != "none":
                if not HAVE_SCIPY:
                    raise RuntimeError("SciPy is required for Butterworth filters.")
                y = butterworth_filter(
                    y, fs=fs, ftype=self.opts.filter_type,
                    order=self.opts.filter_order,
                    f1=self.opts.f1, f2=self.opts.f2
                )

            # arPLS baseline and correction
            baseline = None
            y_corr = y
            if self.opts.use_arpls:
                if not HAVE_SCIPY:
                    raise RuntimeError("SciPy is required for arPLS.")
                baseline = np.zeros_like(y)
                for j in range(y.shape[1]):
                    baseline[:, j] = arpls(y[:, j], lam=self.opts.arpls_lam,
                                           ratio=self.opts.arpls_ratio,
                                           itermax=self.opts.arpls_itermax)
                y_corr = y - baseline

            # Artifact removal + compressed time
            blocks_global = []
            if self.opts.use_artifact:
                # If override exists and fingerprint matches, use it.
                fp_now = _artifact_fingerprint_from_opts(self.opts)
                if (self.opts.artifact_override_blocks is not None
                        and self.opts.artifact_override_fingerprint is not None
                        and tuple(self.opts.artifact_override_fingerprint) == tuple(fp_now)):
                    blocks_global = list(self.opts.artifact_override_blocks)
                else:
                    all_blocks = []
                    for j in range(y_corr.shape[1]):
                        blocks = detect_artifact_blocks_strict(
                            t, y_corr[:, j],
                            slope_pos=self.opts.slope_pos,
                            slope_neg=self.opts.slope_neg,
                            amp_pos=self.opts.amp_pos,
                            amp_neg=self.opts.amp_neg,
                            min_len=self.opts.min_len,
                            merge_gap=self.opts.merge_gap_samples
                        )
                        blocks_ext = extend_artifact_until_calm_light(
                            t, y_corr[:, j], blocks,
                            calm_window=self.opts.calm_window,
                            calm_sigma_k=self.opts.calm_sigma_k,
                            mean_k=self.opts.mean_k,
                            max_extend=self.opts.max_extend,
                            post_silence=self.opts.post_silence,
                            extend_before=self.opts.extend_before,
                            extend_after=self.opts.extend_after
                        )
                        all_blocks.append(blocks_ext)
                    blocks_global = merge_blocks(all_blocks, merge_gap=self.opts.merge_gap_sec)

                # build compressed time axis
                t_keep, _ = apply_blocks(t, np.zeros_like(t), blocks_global)
                n_clean = len(t_keep)
                t_comp = np.arange(n_clean, dtype=float) * dt
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

            # normalization for detection/display
            disp = y_corr.copy()
            if self.opts.norm == "baseline":
                n = int(self.opts.baseline_n)
                n = max(1, min(n, disp.shape[0]))
                disp = disp - np.nanmean(disp[:n, :], axis=0, keepdims=True)
            elif self.opts.norm == "zscore":
                disp = zscore(disp)

            # event detection
            events = []
            if self.opts.use_events:
                events = detect_events_simple(
                    t, disp if self.opts.events_on_normalized else y_corr,
                    thr_z=self.opts.thr_z,
                    refractory_ms=self.opts.refractory_ms,
                    polarity=self.opts.polarity
                )

            out = ProcessedResult(
                t=t,
                y=y_corr,
                disp=disp,
                labels=sel_labels,
                baseline=baseline,
                blocks=blocks_global,
                used_compressed=bool(self.opts.use_artifact),
                events=events,
                fs=estimate_fs(t),
            )
            self.finished.emit(out)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
