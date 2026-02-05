from pathlib import Path
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, QThread
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFileDialog,
    QTextEdit, QCheckBox, QGroupBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QMessageBox, QTabWidget, QSplitter
)

from .deps import HAVE_SCIPY, HAVE_PG, HAVE_MPL, plt, FuncAnimation, PillowWriter
from .io_utils import robust_read_table
from .models import RawModel, ProcessOptions, ProcessedResult
from .signal_proc import estimate_fs
from .views.column_selector import ColumnSelector
from .views.plot_area import PlotArea
from .views.artifact_dialog import ArtifactTuningDialog
from .worker import ProcWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyzer (fast view + robust cleaning + events + animation)")

        if not HAVE_PG:
            QMessageBox.warning(self, "Dependency", "pyqtgraph が必要です。conda/pipでインストールしてください。")
        if not HAVE_SCIPY:
            QMessageBox.warning(self, "Dependency", "SciPy が必要です（Butterworth / arPLS / artifact）。")

        self.model: RawModel | None = None
        self.proc: ProcessedResult | None = None

        self._dirty = True
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.apply_processing)

        # buttons
        self.btn_open = QPushButton("ファイルを開く")
        self.btn_apply = QPushButton("適用（処理実行）")
        self.btn_artifact_tune = QPushButton("artifact drop調整(プレビュー)…")
        self.btn_save_png = QPushButton("PNG保存")
        self.btn_save_pdf = QPushButton("PDF保存")
        self.btn_save_events = QPushButton("イベントCSV保存")
        self.btn_export_gif = QPushButton("GIF出力（アニメ）")

        for b in [self.btn_apply, self.btn_artifact_tune, self.btn_save_png, self.btn_save_pdf, self.btn_save_events, self.btn_export_gif]:
            b.setEnabled(False)

        self.lbl_status = QLabel("ファイルを開いてください")

        self.btn_open.clicked.connect(self.open_file)
        self.btn_apply.clicked.connect(self.apply_processing)
        self.btn_artifact_tune.clicked.connect(self.open_artifact_tuner)
        self.btn_save_png.clicked.connect(lambda: self.save_static("png"))
        self.btn_save_pdf.clicked.connect(lambda: self.save_static("pdf"))
        self.btn_save_events.clicked.connect(self.save_events_csv)
        self.btn_export_gif.clicked.connect(self.export_gif)

        top = QHBoxLayout()
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_apply)
        top.addWidget(self.btn_artifact_tune)
        top.addWidget(self.btn_save_png)
        top.addWidget(self.btn_save_pdf)
        top.addWidget(self.btn_save_events)
        top.addWidget(self.btn_export_gif)
        top.addStretch(1)

        # tabs
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(360)

        # columns tab
        self.colsel = ColumnSelector()
        self.colsel.selection_changed.connect(self.mark_dirty)
        tab_cols = QWidget()
        lay_cols = QVBoxLayout()
        lay_cols.addWidget(self.colsel)
        tab_cols.setLayout(lay_cols)
        self.tabs.addTab(tab_cols, "Columns")

        # display tab
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["overlay", "separate"])
        self.cmb_mode.currentIndexChanged.connect(self.refresh_plot_only)

        self.chk_offset = QCheckBox("オフセット表示（stacked相当）")
        self.chk_offset.setChecked(False)
        self.chk_offset.stateChanged.connect(self.refresh_plot_only)

        self.sp_offset = QDoubleSpinBox()
        self.sp_offset.setRange(0.0, 1e12)
        self.sp_offset.setDecimals(6)
        self.sp_offset.setValue(1.0)
        self.sp_offset.valueChanged.connect(self.refresh_plot_only)

        self.sp_max_points = QSpinBox()
        self.sp_max_points.setRange(500, 2_000_000)
        self.sp_max_points.setValue(100000)
        self.sp_max_points.valueChanged.connect(self.refresh_plot_only)

        tab_disp = QWidget()
        lay_disp = QVBoxLayout()
        lay_disp.addWidget(QLabel("表示モード"))
        lay_disp.addWidget(self.cmb_mode)
        lay_disp.addWidget(self.chk_offset)
        lay_disp.addWidget(QLabel("offset step"))
        lay_disp.addWidget(self.sp_offset)
        lay_disp.addWidget(QLabel("max points（表示用間引き）"))
        lay_disp.addWidget(self.sp_max_points)
        lay_disp.addStretch(1)
        tab_disp.setLayout(lay_disp)
        self.tabs.addTab(tab_disp, "Display")

        # filter tab
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(["none", "lowpass", "highpass", "bandpass"])
        self.cmb_filter.currentIndexChanged.connect(self.mark_dirty)

        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 10)
        self.sp_order.setValue(3)
        self.sp_order.valueChanged.connect(self.mark_dirty)

        self.sp_f1 = QDoubleSpinBox()
        self.sp_f1.setRange(0.0001, 1e6)
        self.sp_f1.setDecimals(4)
        self.sp_f1.setValue(1.0)
        self.sp_f1.valueChanged.connect(self.mark_dirty)

        self.sp_f2 = QDoubleSpinBox()
        self.sp_f2.setRange(0.0001, 1e6)
        self.sp_f2.setDecimals(4)
        self.sp_f2.setValue(50.0)
        self.sp_f2.valueChanged.connect(self.mark_dirty)

        tab_filt = QWidget()
        lay_filt = QVBoxLayout()
        lay_filt.addWidget(QLabel("Filter type"))
        lay_filt.addWidget(self.cmb_filter)
        lay_filt.addWidget(QLabel("Order"))
        lay_filt.addWidget(self.sp_order)
        lay_filt.addWidget(QLabel("f1 (Hz)"))
        lay_filt.addWidget(self.sp_f1)
        lay_filt.addWidget(QLabel("f2 (Hz)  ※bandpassのみ"))
        lay_filt.addWidget(self.sp_f2)
        lay_filt.addStretch(1)
        tab_filt.setLayout(lay_filt)
        self.tabs.addTab(tab_filt, "Filter")

        # baseline/artifact tab
        self.chk_arpls = QCheckBox("arPLSでbaseline推定して除去")
        self.chk_arpls.setChecked(False)
        self.chk_arpls.stateChanged.connect(self.mark_dirty)

        self.sp_lam = QDoubleSpinBox()
        self.sp_lam.setRange(1e2, 1e20)
        self.sp_lam.setDecimals(0)
        self.sp_lam.setValue(1e8)
        self.sp_lam.valueChanged.connect(self.mark_dirty)

        self.sp_ratio = QDoubleSpinBox()
        self.sp_ratio.setRange(1e-12, 1e-1)
        self.sp_ratio.setDecimals(12)
        self.sp_ratio.setValue(1e-6)
        self.sp_ratio.valueChanged.connect(self.mark_dirty)

        self.sp_iter = QSpinBox()
        self.sp_iter.setRange(1, 200)
        self.sp_iter.setValue(20)
        self.sp_iter.valueChanged.connect(self.mark_dirty)

        self.chk_artifact = QCheckBox("artifact除去＋時間圧縮（drop区間を詰める）")
        self.chk_artifact.setChecked(False)
        self.chk_artifact.stateChanged.connect(self.mark_dirty)

        self.sp_slope_pos = QDoubleSpinBox(); self.sp_slope_pos.setRange(0.1, 1000); self.sp_slope_pos.setValue(5.0); self.sp_slope_pos.valueChanged.connect(self.mark_dirty)
        self.sp_slope_neg = QDoubleSpinBox(); self.sp_slope_neg.setRange(0.1, 1000); self.sp_slope_neg.setValue(8.0); self.sp_slope_neg.valueChanged.connect(self.mark_dirty)
        self.sp_amp_pos = QDoubleSpinBox(); self.sp_amp_pos.setRange(0.1, 1000); self.sp_amp_pos.setValue(3.0); self.sp_amp_pos.valueChanged.connect(self.mark_dirty)
        self.sp_amp_neg = QDoubleSpinBox(); self.sp_amp_neg.setRange(0.1, 1000); self.sp_amp_neg.setValue(12.0); self.sp_amp_neg.valueChanged.connect(self.mark_dirty)

        self.sp_merge_gap_sec = QDoubleSpinBox(); self.sp_merge_gap_sec.setRange(0.0, 10.0); self.sp_merge_gap_sec.setValue(0.5); self.sp_merge_gap_sec.valueChanged.connect(self.mark_dirty)
        self.sp_calm_window = QDoubleSpinBox(); self.sp_calm_window.setRange(0.01, 10.0); self.sp_calm_window.setValue(1.0); self.sp_calm_window.valueChanged.connect(self.mark_dirty)

        tab_base = QWidget()
        lay_base = QVBoxLayout()
        lay_base.addWidget(self.chk_arpls)
        lay_base.addWidget(QLabel("arPLS lam"))
        lay_base.addWidget(self.sp_lam)
        lay_base.addWidget(QLabel("arPLS ratio"))
        lay_base.addWidget(self.sp_ratio)
        lay_base.addWidget(QLabel("arPLS itermax"))
        lay_base.addWidget(self.sp_iter)
        lay_base.addSpacing(8)
        lay_base.addWidget(self.chk_artifact)

        gb_art = QGroupBox("artifact params (簡易)")
        ga = QVBoxLayout()
        row1 = QHBoxLayout(); row1.addWidget(QLabel("slope+")); row1.addWidget(self.sp_slope_pos); row1.addWidget(QLabel("slope-")); row1.addWidget(self.sp_slope_neg)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("amp+")); row2.addWidget(self.sp_amp_pos); row2.addWidget(QLabel("amp-")); row2.addWidget(self.sp_amp_neg)
        row3 = QHBoxLayout(); row3.addWidget(QLabel("merge_gap(s)")); row3.addWidget(self.sp_merge_gap_sec); row3.addWidget(QLabel("calm_win(s)")); row3.addWidget(self.sp_calm_window)
        ga.addLayout(row1); ga.addLayout(row2); ga.addLayout(row3)
        gb_art.setLayout(ga)
        lay_base.addWidget(gb_art)
        lay_base.addStretch(1)
        tab_base.setLayout(lay_base)
        self.tabs.addTab(tab_base, "Baseline/Artifacts")

        # events tab
        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["raw", "baseline", "zscore"])
        self.cmb_norm.currentIndexChanged.connect(self.mark_dirty)

        self.sp_baseline_n = QSpinBox()
        self.sp_baseline_n.setRange(1, 10_000_000)
        self.sp_baseline_n.setValue(200)
        self.sp_baseline_n.valueChanged.connect(self.mark_dirty)

        self.chk_events = QCheckBox("event detection（閾値）")
        self.chk_events.setChecked(False)
        self.chk_events.stateChanged.connect(self.mark_dirty)

        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(0.1, 100.0)
        self.sp_thr.setValue(4.0)
        self.sp_thr.valueChanged.connect(self.mark_dirty)

        self.sp_refr = QDoubleSpinBox()
        self.sp_refr.setRange(0.0, 5000.0)
        self.sp_refr.setValue(20.0)
        self.sp_refr.valueChanged.connect(self.mark_dirty)

        self.cmb_pol = QComboBox()
        self.cmb_pol.addItems(["positive", "negative", "both"])
        self.cmb_pol.currentIndexChanged.connect(self.mark_dirty)

        self.chk_events_norm = QCheckBox("検出は正規化後（z等）で行う")
        self.chk_events_norm.setChecked(True)
        self.chk_events_norm.stateChanged.connect(self.mark_dirty)

        tab_evt = QWidget()
        lay_evt = QVBoxLayout()
        lay_evt.addWidget(QLabel("Normalization (表示/検出用)"))
        lay_evt.addWidget(self.cmb_norm)
        lay_evt.addWidget(QLabel("baseline N点（baseline引き用）"))
        lay_evt.addWidget(self.sp_baseline_n)
        lay_evt.addSpacing(8)
        lay_evt.addWidget(self.chk_events)
        lay_evt.addWidget(QLabel("threshold z"))
        lay_evt.addWidget(self.sp_thr)
        lay_evt.addWidget(QLabel("refractory ms"))
        lay_evt.addWidget(self.sp_refr)
        lay_evt.addWidget(QLabel("polarity"))
        lay_evt.addWidget(self.cmb_pol)
        lay_evt.addWidget(self.chk_events_norm)
        lay_evt.addStretch(1)
        tab_evt.setLayout(lay_evt)
        self.tabs.addTab(tab_evt, "Events")

        # animation tab
        self.sp_win = QDoubleSpinBox(); self.sp_win.setRange(0.1, 120.0); self.sp_win.setValue(10.0)
        self.sp_step = QDoubleSpinBox(); self.sp_step.setRange(0.01, 10.0); self.sp_step.setValue(0.5)
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(1, 60); self.sp_fps.setValue(10)

        tab_anim = QWidget()
        lay_anim = QVBoxLayout()
        lay_anim.addWidget(QLabel("window sec"))
        lay_anim.addWidget(self.sp_win)
        lay_anim.addWidget(QLabel("step sec"))
        lay_anim.addWidget(self.sp_step)
        lay_anim.addWidget(QLabel("fps"))
        lay_anim.addWidget(self.sp_fps)
        lay_anim.addStretch(1)
        tab_anim.setLayout(lay_anim)
        self.tabs.addTab(tab_anim, "Animation")

        # log + plot
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.plot = PlotArea()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        center = QWidget()
        lay = QVBoxLayout()
        lay.addLayout(top)
        lay.addWidget(self.lbl_status)
        lay.addWidget(splitter, 1)
        lay.addWidget(self.log, 0)
        center.setLayout(lay)
        self.setCentralWidget(center)

        self.thread: QThread | None = None
        self.worker: ProcWorker | None = None

    # ---------
    def logln(self, s: str):
        self.log.append(s)

    def mark_dirty(self):
        if self.model is None:
            return
        self._dirty = True
        self.btn_apply.setEnabled(True)
        self.btn_artifact_tune.setEnabled(True)
        self.lbl_status.setText("変更あり：適用を押すか、少し待つと自動で適用します")
        self._debounce.start(400)

    def refresh_plot_only(self):
        if self.proc is None:
            return
        self._plot_proc()

    # ---------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "データファイルを選択", "", "Data (*.txt *.tsv *.csv);;All (*.*)"
        )
        if not path:
            return
        try:
            df = robust_read_table(path)
            time_col = df.columns[0]
            t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
            y_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
            valid = np.isfinite(t)
            t = t[valid]
            y = y_df.to_numpy(dtype=float)[valid, :]
            labels = [str(c) for c in y_df.columns]
            dt = float(np.median(np.diff(t)))

            self.model = RawModel(path=path, t_raw=t, y_raw=y, labels=labels, dt=dt)
            self.colsel.set_columns(labels)
            for i in range(min(3, len(labels))):
                self.colsel._checks[i].setChecked(True)

            self.btn_apply.setEnabled(True)
            self.btn_artifact_tune.setEnabled(True)
            self.lbl_status.setText(f"Loaded: {Path(path).name}  (rows={len(t):,}, cols={y.shape[1]})")
            self.log.setPlainText(f"Loaded: {path}\nrows={len(t):,}, cols={y.shape[1]}\n")
            self.mark_dirty()
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{type(e).__name__}: {e}")

    def _collect_opts(self) -> ProcessOptions:
        sel = self.colsel.selected_indices()
        if len(sel) == 0:
            raise ValueError("列が選択されていません。")

        return ProcessOptions(
            sel_idx=sel,
            filter_type=self.cmb_filter.currentText(),
            filter_order=int(self.sp_order.value()),
            f1=float(self.sp_f1.value()),
            f2=float(self.sp_f2.value()),
            use_arpls=self.chk_arpls.isChecked(),
            arpls_lam=float(self.sp_lam.value()),
            arpls_ratio=float(self.sp_ratio.value()),
            arpls_itermax=int(self.sp_iter.value()),
            use_artifact=self.chk_artifact.isChecked(),
            slope_pos=float(self.sp_slope_pos.value()),
            slope_neg=float(self.sp_slope_neg.value()),
            amp_pos=float(self.sp_amp_pos.value()),
            amp_neg=float(self.sp_amp_neg.value()),
            min_len=10,
            merge_gap_samples=2,
            calm_window=float(self.sp_calm_window.value()),
            calm_sigma_k=1.5,
            mean_k=2.0,
            max_extend=8.0,
            post_silence=0.5,
            extend_before=0.2,
            extend_after=0.3,
            merge_gap_sec=float(self.sp_merge_gap_sec.value()),
            norm=self.cmb_norm.currentText(),
            baseline_n=int(self.sp_baseline_n.value()),
            use_events=self.chk_events.isChecked(),
            thr_z=float(self.sp_thr.value()),
            refractory_ms=float(self.sp_refr.value()),
            polarity=self.cmb_pol.currentText(),
            events_on_normalized=self.chk_events_norm.isChecked(),
        )

    def apply_processing(self):
        if self.model is None:
            return
        try:
            opts = self._collect_opts()
        except Exception as e:
            QMessageBox.warning(self, "Option", str(e))
            return

        if opts.filter_type != "none" and not HAVE_SCIPY:
            QMessageBox.warning(self, "Dependency", "SciPyが無いのでfilterが使えません。")
            return
        if (opts.use_arpls or opts.use_artifact) and not HAVE_SCIPY:
            QMessageBox.warning(self, "Dependency", "SciPyが無いのでarPLS/artifactが使えません。")
            return
        if opts.use_arpls and len(opts.sel_idx) > 30:
            QMessageBox.information(self, "arPLS", "arPLSは重いので、まずは選択列を30本以下にするのを推奨します。")

        self.btn_apply.setEnabled(False)
        self.lbl_status.setText("処理中…")
        self._dirty = False

        self.thread = QThread()
        self.worker = ProcWorker(self.model, opts)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_processed)
        self.worker.failed.connect(self._on_proc_fail)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_processed(self, result: ProcessedResult):
        self.proc = result
        self.lbl_status.setText("完了")
        for b in [self.btn_save_png, self.btn_save_pdf, self.btn_save_events, self.btn_export_gif]:
            b.setEnabled(True)
        self._plot_proc()
        self.logln(f"\nProcessed: n={len(result.t):,}, m={result.y.shape[1]}, fs={result.fs:.2f} Hz, compressed={result.used_compressed}")
        if result.blocks:
            self.logln(f"artifact blocks: {len(result.blocks)}")

    def _on_proc_fail(self, msg: str):
        self.lbl_status.setText("エラー")
        self.logln(f"\n[Error] {msg}")
        QMessageBox.critical(self, "Processing failed", msg)

    def _downsample_for_display(self, t: np.ndarray, y: np.ndarray):
        N = len(t)
        max_points = int(self.sp_max_points.value())
        if N > max_points:
            step = max(1, N // max_points)
            return t[::step], y[::step, :]
        return t, y

    def _plot_proc(self):
        if self.proc is None:
            return
        t = self.proc.t
        y = self.proc.disp
        labels = self.proc.labels
        events = self.proc.events if self.chk_events.isChecked() else None

        t_d, y_d = self._downsample_for_display(t, y)

        mode = self.cmb_mode.currentText()
        offset = self.chk_offset.isChecked()
        offset_step = float(self.sp_offset.value())

        thin = True
        alpha = 180 if offset else 220

        self.plot.plot(
            t_d, y_d, labels,
            mode=mode,
            offset=offset,
            offset_step=offset_step,
            thin=thin,
            alpha=alpha,
            show_events=events
        )

    # ---------- exports（元のまま移植）
    def open_artifact_tuner(self):
        if self.model is None:
            return

        # まず、選択列だけ取り出す
        sel = self.colsel.selected_indices()
        if len(sel) == 0:
            QMessageBox.warning(self, "Artifact", "列が選択されていません。")
            return

        t = self.model.t_raw
        y = self.model.y_raw[:, sel]

        # プレビューは「本処理に近い状態」で見たいので、
        # 可能なら軽く前処理（filter + arPLS）をここで適用してから渡すのがベスト。
        # ただし重くなりうるので、ここでは「filterのみ」 or 「現状そのまま」でもOK。
        # ここは最初はシンプルに raw のまま（必要なら後で一致させる）。
        # ---- 一致させたいなら：worker.processの前半を共通化して呼ぶのが理想 ----

        init_params = dict(
            slope_pos=float(self.sp_slope_pos.value()),
            slope_neg=float(self.sp_slope_neg.value()),
            amp_pos=float(self.sp_amp_pos.value()),
            amp_neg=float(self.sp_amp_neg.value()),
            merge_gap_sec=float(self.sp_merge_gap_sec.value()),
            calm_window=float(self.sp_calm_window.value()),
            calm_sigma_k=1.5,
            mean_k=2.0,
            max_extend=8.0,
            post_silence=0.5,
            extend_before=0.2,
            extend_after=0.3,
            min_len=10,
            merge_gap_samples=2,
        )

        params = ArtifactTuningDialog.get_params(t, y, init_params, parent=self)
        if params is None:
            return  # cancel

        # 確定：MainWindow側のスピンボックスへ反映
        self.sp_slope_pos.setValue(params["slope_pos"])
        self.sp_slope_neg.setValue(params["slope_neg"])
        self.sp_amp_pos.setValue(params["amp_pos"])
        self.sp_amp_neg.setValue(params["amp_neg"])
        self.sp_merge_gap_sec.setValue(params["merge_gap_sec"])
        self.sp_calm_window.setValue(params["calm_window"])

        # 拡張パラメータもUIに出したくなったら、MainWindow側にもSpinBox増設して同期
        # 今回はDialog側で調整してもoptsに反映されないので、opts固定値をMainWindowに増やすか、
        # いったんはDialog側もMainWindow固定値に合わせて使うようにするのが確実。

        # ここで mark_dirty すれば、確定ボタン＝処理に反映、になる
        self.mark_dirty()

    def save_static(self, fmt: str):
        if self.proc is None:
            return
        if not HAVE_MPL:
            QMessageBox.warning(self, "Dependency", "matplotlib が無いので保存できません。")
            return

        default = Path(self.model.path).with_suffix(f".{fmt}").name if self.model else f"plot.{fmt}"
        if fmt == "png":
            out, _ = QFileDialog.getSaveFileName(self, "PNG保存", default, "PNG (*.png)")
        else:
            out, _ = QFileDialog.getSaveFileName(self, "PDF保存", default, "PDF (*.pdf)")
        if not out:
            return

        try:
            t = self.proc.t
            y = self.proc.disp
            labels = self.proc.labels
            mode = self.cmb_mode.currentText()
            offset = self.chk_offset.isChecked()
            offset_step = float(self.sp_offset.value())

            if mode == "separate":
                fig, axes = plt.subplots(len(labels), 1, figsize=(10, max(2, 1.6 * len(labels))), sharex=True)
                if len(labels) == 1:
                    axes = [axes]
                for i, ax in enumerate(axes):
                    ax.plot(t, y[:, i], linewidth=0.8, alpha=0.9)
                    ax.set_ylabel(labels[i])
                    ax.grid(True, alpha=0.2)
                axes[-1].set_xlabel("time")
                fig.tight_layout()
            else:
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(111)
                for i in range(y.shape[1]):
                    yy = y[:, i] + (i * offset_step if offset else 0.0)
                    ax.plot(t, yy, linewidth=0.8, alpha=0.8)
                ax.set_xlabel("time")
                ax.set_ylabel("value" + (" (offset)" if offset else ""))
                ax.grid(True, alpha=0.2)
                fig.tight_layout()

            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            QMessageBox.information(self, "Saved", f"Saved: {out}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")

    def save_events_csv(self):
        if self.proc is None:
            return
        events = self.proc.events
        if not events:
            QMessageBox.information(self, "Events", "イベントがありません（検出ON + 適用が必要）")
            return
        default = "events.csv"
        out, _ = QFileDialog.getSaveFileName(self, "イベントCSV保存", default, "CSV (*.csv)")
        if not out:
            return
        rows = [{"col": self.proc.labels[j], "time_s": te, "value": ve} for (j, te, ve) in events]
        pd.DataFrame(rows).to_csv(out, index=False)
        QMessageBox.information(self, "Saved", f"Saved: {out}")

    def export_gif(self):
        if self.proc is None:
            return
        if not HAVE_MPL:
            QMessageBox.warning(self, "Dependency", "matplotlib が無いのでGIF出力できません。")
            return

        default = "animation.gif"
        out, _ = QFileDialog.getSaveFileName(self, "GIFとして保存", default, "GIF (*.gif)")
        if not out:
            return

        try:
            t = self.proc.t
            y = self.proc.disp
            labels = self.proc.labels

            fs = estimate_fs(t)
            window_sec = float(self.sp_win.value())
            step_sec = float(self.sp_step.value())
            fps = int(self.sp_fps.value())

            window_samples = max(5, int(round(window_sec * fs)))
            step = max(1, int(round(step_sec * fs)))

            total = len(t)
            if total <= window_samples + 2:
                raise ValueError("データが短すぎてアニメにできません。")

            frames = np.arange(0, total - window_samples, step, dtype=int)

            n_tr = len(labels)
            fig_h = min(2.0 * n_tr, 16.0)
            fig, axes = plt.subplots(n_tr, 1, figsize=(10, fig_h), sharex=True)
            if n_tr == 1:
                axes = [axes]

            lines = []
            ymin = float(np.nanmin(y))
            ymax = float(np.nanmax(y))
            pad = 0.05 * (ymax - ymin + 1e-12)
            for ax, lab in zip(axes, labels):
                (ln,) = ax.plot([], [], lw=1.0, alpha=0.9)
                ax.set_ylabel(lab)
                ax.set_ylim(ymin - pad, ymax + pad)
                ax.grid(True, alpha=0.2)
                lines.append(ln)
            axes[-1].set_xlabel("time (processed)")

            def init():
                for ln in lines:
                    ln.set_data([], [])
                return lines

            def update(fi):
                start = int(fi)
                end = min(total, start + window_samples)
                x = t[start:end]
                for j, ln in enumerate(lines):
                    ln.set_data(x, y[start:end, j])
                if len(x) > 1:
                    for ax in axes:
                        ax.set_xlim(x[0], x[-1])
                return lines

            ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000 / fps)
            writer = PillowWriter(fps=fps)
            ani.save(out, writer=writer)
            plt.close(fig)
            QMessageBox.information(self, "Saved", f"Saved: {out}")
        except Exception as e:
            QMessageBox.critical(self, "GIF failed", f"{type(e).__name__}: {e}")