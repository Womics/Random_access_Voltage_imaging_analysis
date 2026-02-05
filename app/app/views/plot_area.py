import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from . import __init__  # noqa
from ..deps import HAVE_PG, pg

class PlotArea(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        if not HAVE_PG:
            lay = QVBoxLayout()
            lay.addWidget(QLabel("pyqtgraphが見つかりません。pyqtgraphをインストールしてください。"))
            self.setLayout(lay)
            self.pg = None
            return

        pg.setConfigOptions(antialias=False, useOpenGL=True)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground('w')

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.glw)
        self.setLayout(lay)

        self.plots = []
        self.items = []

    def clear(self):
        if not HAVE_PG:
            return
        self.glw.clear()
        self.plots = []
        self.items = []

    def plot(self, t, y, labels, mode="overlay", offset=False, offset_step=1.0,
             thin=True, alpha=200, show_events=None):
        if not HAVE_PG:
            return
        self.clear()
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)

        if mode == "separate":
            base_plot = None
            for i in range(y.shape[1]):
                p = self.glw.addPlot(row=i, col=0)
                if base_plot is None:
                    base_plot = p
                else:
                    p.setXLink(base_plot)
                p.showGrid(x=True, y=True, alpha=0.2)
                p.setLabel('left', labels[i])

                col = pg.intColor(i, hues=max(8, y.shape[1]))
                col.setAlpha(alpha)
                pen = pg.mkPen(col, width=0.8 if thin else 1.2)

                item = p.plot(t, y[:, i], pen=pen)
                item.setDownsampling(auto=True, method="peak")
                item.setClipToView(True)

                self.plots.append(p)
                self.items.append(item)

            if show_events:
                for (j, te, ve) in show_events:
                    if 0 <= j < len(self.plots):
                        sc = pg.ScatterPlotItem([te], [ve], size=6,
                                                pen=pg.mkPen(None),
                                                brush=pg.mkBrush(255, 0, 0, 180))
                        self.plots[j].addItem(sc)
        else:
            p = self.glw.addPlot(row=0, col=0)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel('bottom', "time")
            p.setLabel('left', "value")
            penw = 0.8 if thin else 1.2

            for i in range(y.shape[1]):
                yy = y[:, i] + (i * offset_step if offset else 0.0)
                col = pg.intColor(i, hues=max(8, y.shape[1]))
                col.setAlpha(alpha)
                pen = pg.mkPen(col, width=penw)

                item = p.plot(t, yy, pen=pen)
                item.setDownsampling(auto=True, method="peak")
                item.setClipToView(True)
                self.items.append(item)

            self.plots = [p]

            if show_events:
                xs = [te for (_, te, _) in show_events]
                ys = [ve for (_, _, ve) in show_events]
                sc = pg.ScatterPlotItem(xs, ys, size=6,
                                        pen=pg.mkPen(None),
                                        brush=pg.mkBrush(255, 0, 0, 180))
                p.addItem(sc)
