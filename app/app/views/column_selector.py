from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QLineEdit, QPushButton, QHBoxLayout, QVBoxLayout, QScrollArea, QCheckBox

class ColumnSelector(QWidget):
    selection_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.search = QLineEdit()
        self.search.setPlaceholderText("列を検索…")
        self.search.textChanged.connect(self._apply_filter)

        self.btn_all = QPushButton("全選択")
        self.btn_none = QPushButton("全解除")
        self.btn_all.clicked.connect(self.select_all)
        self.btn_none.clicked.connect(self.select_none)

        bar = QHBoxLayout()
        bar.addWidget(self.btn_all)
        bar.addWidget(self.btn_none)

        self.box_container = QWidget()
        self.box_layout = QVBoxLayout()
        self.box_layout.setAlignment(Qt.AlignTop)
        self.box_container.setLayout(self.box_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.box_container)

        lay = QVBoxLayout()
        lay.addWidget(self.search)
        lay.addLayout(bar)
        lay.addWidget(self.scroll)
        self.setLayout(lay)

        self._checks: list[QCheckBox] = []
        self._labels: list[str] = []

    def set_columns(self, labels: list[str]):
        for cb in self._checks:
            cb.deleteLater()
        self._checks.clear()
        self._labels = labels

        for lab in labels:
            cb = QCheckBox(lab)
            cb.setChecked(False)
            cb.stateChanged.connect(lambda _=None: self.selection_changed.emit())
            self.box_layout.addWidget(cb)
            self._checks.append(cb)

        self._apply_filter(self.search.text())

    def selected_indices(self) -> list[int]:
        return [i for i, cb in enumerate(self._checks) if cb.isChecked()]

    def select_all(self):
        for cb in self._checks:
            if cb.isVisible():
                cb.setChecked(True)
        self.selection_changed.emit()

    def select_none(self):
        for cb in self._checks:
            if cb.isVisible():
                cb.setChecked(False)
        self.selection_changed.emit()

    def _apply_filter(self, txt: str):
        q = (txt or "").strip().lower()
        for cb in self._checks:
            cb.setVisible((q in cb.text().lower()) if q else True)
