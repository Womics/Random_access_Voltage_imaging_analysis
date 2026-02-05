# app/views/artifact_dialog.py
import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QDoubleSpinBox, QWidget, QMessageBox
)

from ..deps import HAVE_PG, HAVE_SCIPY, pg
from ..artifact import (
    detect_artifact_blocks_strict,
    extend_artifact_until_calm_light,
    merge_blocks,
)
from ..signal_proc import estimate_fs


class ArtifactTuningDialog(QDialog):
    """
    artifact除去のパラメータを別ウィンドウで調整し、
    dropブロック（時間区間）を可視化しながら確定できるDialog。

    - 入力: t, y (N x M)  ※できれば「baseline除去後」(y_corr) を渡すのが一致する
    - 出力: dict(params) または None（キャンセル）
    """
    def __init__(self, t: np.ndarray, y: np.ndarray, init_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Artifact drop tuning")
        self.setModal(True)
        self.resize(1100, 700)

        self.t = np.asarray(t, dtype=float)
        self.y = np.asarray(y, dtype=float)
        if self.y.ndim == 1:
            self.y = self.y[:, None]

        self._result_params = None
        self._regions = []  # pg.LinearRegionItem list

        # ---------- plot ----------
        if not HAVE_PG:
            QMessageBox.warning(self, "Dependency", "pyqtgraph が無いので表示できません。")
            self.accept()
            return

        self.plotw = pg.PlotWidget()
        self.plotw.setBackground("w")
        self.plotw.showGrid(x=True, y=True, alpha=0.2)

        # trace preview（重いときは1本だけでOK）
        self.pen = pg.mkPen(color=(0, 0, 0, 180), width=0.9)

        # ---------- controls ----------
        ctrl = QWidget()
        ctrl_lay = QVBoxLayout()
        ctrl.setLayout(ctrl_lay)

        gb = QGroupBox("artifact params")
        gb_lay = QVBoxLayout()
        gb.setLayout(gb_lay)

        def mk_spin(label, lo, hi, val, decimals=3):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(decimals)
            sp.setValue(val)
            sp.setSingleStep((hi - lo) / 200.0 if hi > lo else 0.1)
            row.addWidget(sp, 1)
            gb_lay.addLayout(row)
            return sp

        # init params（MainWindowのoptsに合わせる）
        self.sp_slope_pos = mk_spin("slope_pos", 0.1, 1000.0, init_params.get("slope_pos", 5.0), 3)
        self.sp_slope_neg = mk_spin("slope_neg", 0.1, 1000.0, init_params.get("slope_neg", 8.0), 3)
        self.sp_amp_pos   = mk_spin("amp_pos",   0.1, 1000.0, init_params.get("amp_pos", 3.0), 3)
        self.sp_amp_neg   = mk_spin("amp_neg",   0.1, 1000.0, init_params.get("amp_neg", 12.0), 3)

        self.sp_merge_gap_sec = mk_spin("merge_gap_sec", 0.0, 10.0, init_params.get("merge_gap_sec", 0.5), 3)
        self.sp_calm_window   = mk_spin("calm_window",   0.01, 10.0, init_params.get("calm_window", 1.0), 3)

        # 伸長系（UIは最低限だけ）
        self.sp_calm_sigma_k = mk_spin("calm_sigma_k", 0.1, 10.0, init_params.get("calm_sigma_k", 1.5), 3)
        self.sp_mean_k       = mk_spin("mean_k",       0.1, 10.0, init_params.get("mean_k", 2.0), 3)
        self.sp_max_extend   = mk_spin("max_extend",   0.0, 60.0, init_params.get("max_extend", 8.0), 3)
        self.sp_post_silence = mk_spin("post_silence", 0.0, 10.0, init_params.get("post_silence", 0.5), 3)
        self.sp_extend_before= mk_spin("extend_before",0.0, 10.0, init_params.get("extend_before", 0.2), 3)
        self.sp_extend_after = mk_spin("extend_after", 0.0, 10.0, init_params.get("extend_after", 0.3), 3)

        # strict detect側の固定値（必要なら後でUI化）
        self.min_len = int(init_params.get("min_len", 10))
        self.merge_gap_samples = int(init_params.get("merge_gap_samples", 2))

        # 状態表示
        self.lbl_info = QLabel("")
        self.lbl_info.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # buttons
        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("確定（反映）")
        self.btn_cancel = QPushButton("キャンセル")
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_cancel)

        ctrl_lay.addWidget(gb)
        ctrl_lay.addWidget(self.lbl_info)
        ctrl_lay.addStretch(1)
        ctrl_lay.addLayout(btn_row)

        # root layout
        root = QHBoxLayout()
        root.addWidget(self.plotw, 2)
        root.addWidget(ctrl, 1)

        main = QVBoxLayout()
        main.addLayout(root)
        self.setLayout(main)

        # debounce update
        self._deb = QTimer()
        self._deb.setSingleShot(True)
        self._deb.timeout.connect(self._update_preview)

        for sp in [
            self.sp_slope_pos, self.sp_slope_neg, self.sp_amp_pos, self.sp_amp_neg,
            self.sp_merge_gap_sec, self.sp_calm_window,
            self.sp_calm_sigma_k, self.sp_mean_k, self.sp_max_extend,
            self.sp_post_silence, self.sp_extend_before, self.sp_extend_after
        ]:
            sp.valueChanged.connect(self._schedule_update)

        self.btn_apply.clicked.connect(self._on_accept)
        self.btn_cancel.clicked.connect(self.reject)

        # initial draw
        self._update_preview()

    def _schedule_update(self):
        self._deb.start(120)

    def _clear_regions(self):
        for r in self._regions:
            try:
                self.plotw.removeItem(r)
            except Exception:
                pass
        self._regions = []

    def _draw_regions(self, blocks):
        # 半透明の縦帯
        self._clear_regions()
        for (a, b) in blocks:
            r = pg.LinearRegionItem(values=(a, b), movable=False)
            r.setBrush(pg.mkBrush(255, 200, 0, 80))
            try:
                region.setPen(pg.mkPen(255, 165, 0, 180))
            except Exception:
                pass
            self.plotw.addItem(r)
            self._regions.append(r)

    def _compute_blocks(self):
        # M本のうち全てのブロックを作ってmerge（本処理と同じ思想）
        t = self.t
        y = self.y

        all_blocks = []
        for j in range(y.shape[1]):
            blocks = detect_artifact_blocks_strict(
                t, y[:, j],
                slope_pos=float(self.sp_slope_pos.value()),
                slope_neg=float(self.sp_slope_neg.value()),
                amp_pos=float(self.sp_amp_pos.value()),
                amp_neg=float(self.sp_amp_neg.value()),
                min_len=self.min_len,
                merge_gap=self.merge_gap_samples
            )
            blocks_ext = extend_artifact_until_calm_light(
                t, y[:, j], blocks,
                calm_window=float(self.sp_calm_window.value()),
                calm_sigma_k=float(self.sp_calm_sigma_k.value()),
                mean_k=float(self.sp_mean_k.value()),
                max_extend=float(self.sp_max_extend.value()),
                post_silence=float(self.sp_post_silence.value()),
                extend_before=float(self.sp_extend_before.value()),
                extend_after=float(self.sp_extend_after.value()),
            )
            all_blocks.append(blocks_ext)

        blocks_global = merge_blocks(all_blocks, merge_gap=float(self.sp_merge_gap_sec.value()))
        return blocks_global

    def _update_preview(self):
        if not HAVE_SCIPY:
            # artifact系はscipy不要だけど、アプリ側の依存関係方針に合わせてメッセージ出すならここ
            pass

        t = self.t
        y = self.y

        self.plotw.clear()
        self._regions = []

        # プレビューは表示負荷軽減のため「1本目」だけ描画（必要なら全本に拡張可）
        y0 = y[:, 0]
        item = self.plotw.plot(t, y0, pen=self.pen)
        item.setDownsampling(auto=True, method="peak")
        item.setClipToView(True)

        blocks = self._compute_blocks()
        self._draw_regions(blocks)

        # info
        fs = estimate_fs(t)
        total = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        drop = sum((b - a) for (a, b) in blocks) if blocks else 0.0
        self.lbl_info.setText(
            f"blocks={len(blocks)} / total={total:.3f}s / drop≈{drop:.3f}s / keep≈{max(0.0, total-drop):.3f}s / fs≈{fs:.1f}Hz"
        )

        self._latest_blocks = blocks

    def _on_accept(self):
        self._result_params = self.result_params()
        self.accept()

    def result_params(self) -> dict:
        # MainWindow側のスピンボックスへ反映する値
        return dict(
            slope_pos=float(self.sp_slope_pos.value()),
            slope_neg=float(self.sp_slope_neg.value()),
            amp_pos=float(self.sp_amp_pos.value()),
            amp_neg=float(self.sp_amp_neg.value()),
            merge_gap_sec=float(self.sp_merge_gap_sec.value()),
            calm_window=float(self.sp_calm_window.value()),
            calm_sigma_k=float(self.sp_calm_sigma_k.value()),
            mean_k=float(self.sp_mean_k.value()),
            max_extend=float(self.sp_max_extend.value()),
            post_silence=float(self.sp_post_silence.value()),
            extend_before=float(self.sp_extend_before.value()),
            extend_after=float(self.sp_extend_after.value()),
            min_len=int(self.min_len),
            merge_gap_samples=int(self.merge_gap_samples),
        )

    @staticmethod
    def get_params(t, y, init_params: dict, parent=None):
        dlg = ArtifactTuningDialog(t, y, init_params, parent=parent)
        ok = (dlg.exec() == QDialog.Accepted)
        if not ok:
            return None
        return dlg._result_params