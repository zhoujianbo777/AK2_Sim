"""Helper: write new window_classification.py"""
import pathlib, textwrap

src = textwrap.dedent("""\
    # window_classification.py  -- W3 Obstacle Classification View
    # Bar chart: pyqtgraph (fast per-frame). Radar: matplotlib (click-only).
    import numpy as np
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import pyqtgraph as pg
    from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor, QFont
    from modules.result_fusion import AlgoResult
    from modules.config_loader import ConfigLoader

    CLASS_COLORS_9 = ["#607D8B","#2196F3","#F44336","#FF9800","#FFFFFF","#FFF9C4","#9C27B0","#4CAF50","#00BCD4"]
    CLASS_NAMES_9  = ["Wall","Vehicle","Pedestrian","Soft","Open","Clutter","Overhead","Curb","Wet"]
    CLASS_ABBR_9   = ["Wa","Ve","Pe","So","Op","Cl","Ov","Cu","We"]
    CLASS_COLORS_5 = ["#607D8B","#FF9800","#FFFFFF","#FFF9C4","#E0E0E0"]
    CLASS_NAMES_5  = ["Hard","Soft","Open","Clutter","Unknown"]
    CLASS_ABBR_5   = ["H","S","O","G","?"]
    _PEN_SEL  = pg.mkPen("#F44336", width=2)
    _PEN_NORM = pg.mkPen("#555555", width=1)


    class _BarRow(QWidget):
        N = 6
        def __init__(self, sensor_labels, row_title, class_colors, class_abbrs,
                     ch_offset=0, parent=None):
            super().__init__(parent)
            self._colors = class_colors
            self._abbrs  = class_abbrs
            self._offset = ch_offset
            self._click_callback = None
            lay = QVBoxLayout(self)
            lay.setContentsMargins(2, 0, 2, 0)
            lay.setSpacing(0)
            lbl = QLabel(row_title)
            lbl.setFont(QFont("Arial", 8, QFont.Weight.Bold))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(lbl)
            self._pw = pg.PlotWidget()
            self._pw.setBackground("w")
            self._pw.setYRange(0, 1.15)
            self._pw.showGrid(y=True, alpha=0.3)
            self._pw.getAxis("left").setWidth(28)
            self._pw.getAxis("left").setStyle(tickTextOffset=2)
            self._pw.getAxis("bottom").setStyle(tickTextOffset=2)
            self._pw.getAxis("bottom").setTicks([[(i, sensor_labels[i]) for i in range(self.N)]])
            self._pw.setXRange(-0.6, self.N - 0.4)
            self._pw.setMouseEnabled(x=False, y=False)
            self._pw.setMenuEnabled(False)
            self._bar_items = []
            for i in range(self.N):
                bar = pg.BarGraphItem(x=[float(i)], height=[0.0], width=0.65,
                                      brush=pg.mkBrush("#BDBDBD"), pen=_PEN_NORM)
                self._pw.addItem(bar)
                self._bar_items.append(bar)
            self._text_items = []
            for i in range(self.N):
                t = pg.TextItem(text="", anchor=(0.5, 1.0), color=(50, 50, 50))
                t.setFont(QFont("Arial", 8))
                self._pw.addItem(t)
                self._text_items.append(t)
            self._pw.scene().sigMouseClicked.connect(self._on_scene_click)
            lay.addWidget(self._pw)

        def update(self, class_ids, confs, selected_ch_abs):
            for local in range(self.N):
                ch    = local + self._offset
                h     = float(confs[ch])
                cls   = int(class_ids[ch])
                color = QColor(self._colors[cls] if cls < len(self._colors) else "#BDBDBD")
                pen   = _PEN_SEL if ch == selected_ch_abs else _PEN_NORM
                self._bar_items[local].setOpts(height=[h], brush=pg.mkBrush(color), pen=pen)
                abbr  = self._abbrs[cls] if cls < len(self._abbrs) else "?"
                self._text_items[local].setText(abbr)
                self._text_items[local].setPos(local, min(h + 0.04, 1.05))

        def clear(self):
            for local in range(self.N):
                self._bar_items[local].setOpts(height=[0.0],
                                               brush=pg.mkBrush("#BDBDBD"), pen=_PEN_NORM)
                self._text_items[local].setText("")

        def _on_scene_click(self, event):
            pos   = self._pw.plotItem.vb.mapSceneToView(event.scenePos())
            local = int(round(pos.x()))
            if 0 <= local < self.N and self._click_callback:
                self._click_callback(local + self._offset)


    class WindowClassification(QWidget):
        def __init__(self, cfg: ConfigLoader, mode: str = "ai", parent=None):
            super().__init__(parent)
            self.cfg  = cfg
            self._mode = mode
            self._selected_ch = 0
            self._last_result = None
            if mode == "traditional":
                self._class_colors = CLASS_COLORS_5
                self._class_names  = CLASS_NAMES_5
                self._class_abbr   = CLASS_ABBR_5
            else:
                self._class_colors = CLASS_COLORS_9
                self._class_names  = CLASS_NAMES_9
                self._class_abbr   = CLASS_ABBR_9
            self._sensor_labels = [
                cfg.get_sensor_config(f"S{i:02d}").get("label", f"S{i:02d}")
                for i in range(1, 13)
            ]
            self.setWindowTitle("W3 - Obstacle Classification")
            self.resize(800, 480)
            self._build_ui()

        def _build_ui(self):
            root = QHBoxLayout(self)
            root.setContentsMargins(4, 4, 4, 4)
            left = QWidget()
            left_lay = QVBoxLayout(left)
            left_lay.setSpacing(2)
            left_lay.setContentsMargins(0, 0, 0, 0)
            self._row_front = _BarRow(self._sensor_labels[0:6], "Front  (S01-S06)",
                                       self._class_colors, self._class_abbr, ch_offset=0)
            self._row_rear  = _BarRow(self._sensor_labels[6:12], "Rear   (S07-S12)",
                                       self._class_colors, self._class_abbr, ch_offset=6)
            self._row_front._click_callback = self._on_channel_clicked
            self._row_rear._click_callback  = self._on_channel_clicked
            left_lay.addWidget(self._row_front, stretch=1)
            left_lay.addWidget(self._row_rear,  stretch=1)
            root.addWidget(left, stretch=3)
            self._fig_radar    = Figure(figsize=(3.2, 3.2), tight_layout=True)
            self._ax_radar     = self._fig_radar.add_subplot(111, projection="polar")
            self._canvas_radar = FigureCanvas(self._fig_radar)
            self._canvas_radar.setMinimumWidth(200)
            root.addWidget(self._canvas_radar, stretch=2)
            self._draw_empty_radar()

        def _update_bar(self, result):
            cids  = result.class_ids
            confs = np.array([result.class_probs[i, cids[i]] for i in range(12)], dtype=float)
            self._row_front.update(cids, confs, self._selected_ch)
            self._row_rear.update(cids,  confs, self._selected_ch)

        def _on_channel_clicked(self, ch):
            self._selected_ch = ch
            if self._last_result is not None:
                self._update_bar(self._last_result)
                self._update_radar(self._last_result, ch)

        def _draw_empty_radar(self):
            ax = self._ax_radar
            ax.clear()
            n   = len(self._class_names)
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
            ax.set_xticks(ang)
            ax.set_xticklabels(self._class_names, fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"Prob. ({self._sensor_labels[0]})", fontsize=9)
            self._canvas_radar.draw()

        def _update_radar(self, result, ch):
            ax     = self._ax_radar
            ax.clear()
            n_cls  = len(self._class_names)
            probs  = result.class_probs[ch][:n_cls].tolist()
            angles = np.linspace(0, 2 * np.pi, n_cls, endpoint=False).tolist()
            angles += angles[:1]; probs += probs[:1]
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self._class_names, fontsize=7)
            ax.set_ylim(0, 1)
            ax.plot(angles, probs, "o-", color="#1565C0", linewidth=2)
            ax.fill(angles, probs, alpha=0.2, color="#1565C0")
            ax.set_title(f"Prob. ({self._sensor_labels[ch]})", fontsize=9)
            self._canvas_radar.draw()

        def on_result_updated(self, result: AlgoResult) -> None:
            self._last_result = result
            self._update_bar(result)

        def reset(self) -> None:
            self._last_result = None
            self._selected_ch = 0
            self._row_front.clear()
            self._row_rear.clear()
            self._draw_empty_radar()
""")

out = pathlib.Path("F:/CODE/AK2_Sim/views/window_classification.py")
out.write_text(src, encoding="utf-8")
print(f"Written {len(src)} bytes to {out}")
