# analyzer_app/app.py
from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 850)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
