# analyzer_app/proc_worker.py
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot
from .processing import apply_full_pipeline

class ProcWorker(QObject):
    finished = Signal(object)   # dict result
    failed = Signal(str)

    def __init__(self, model: dict, opts: dict):
        super().__init__()
        self.model = model
        self.opts = opts

    @Slot()
    def run(self):
        try:
            out = apply_full_pipeline(self.model, self.opts)
            self.finished.emit(out)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
