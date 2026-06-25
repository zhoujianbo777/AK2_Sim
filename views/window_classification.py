# window_classification.py  -- W3 Obstacle Classification View
# Bar chart: pyqtgraph (fast per-frame).
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from modules.result_fusion import AlgoResult
from modules.config_loader import ConfigLoader
from modules.obstacle_classes import CLASS_COLORS, CLASS_NAMES, CLASS_ABBR
_PEN_SEL     = pg.mkPen("#F44336", width=2)
_PEN_NORM    = pg.mkPen("#555555", width=1)
_PEN_SA_DIFF = pg.mkPen("#FF9800", width=2)   # orange border: SA changed SC's class


def _desaturate(hex_color: str, ratio: float = 0.55) -> QColor:
    """Return a desaturated version of *hex_color* (used for SC 'before' bars)."""
    c = QColor(hex_color)
    h, s, v, a = c.getHsvF()
    c.setHsvF(h, s * ratio, min(v + 0.10, 1.0), a)
    return c


class _BarRow(QWidget):
    N = 6
    def __init__(self, sensor_labels, row_title, class_colors, class_abbrs,
                 ch_offset=0, desat=False, parent=None):
        super().__init__(parent)
        self._abbrs  = class_abbrs
        self._offset = ch_offset
        self._click_callback = None
        # Pre-compute per-class brush colors (desaturated for SC preview rows)
        if desat:
            self._qcolors = [_desaturate(c) for c in class_colors]
        else:
            self._qcolors = [QColor(c) for c in class_colors]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(0)
        lbl = QLabel(row_title)
        lbl.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(lbl)
        self._pw = pg.PlotWidget()
        self._pw.setBackground("w")
        self._pw.setYRange(0, 1.20)
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
        # Class abbreviation above bar
        self._text_items = []
        for i in range(self.N):
            t = pg.TextItem(text="", anchor=(0.5, 1.0), color=(50, 50, 50))
            t.setFont(QFont("Arial", 8))
            self._pw.addItem(t)
            self._text_items.append(t)
        # Head-2: material hardness inside bar body
        self._hard_items = []
        for i in range(self.N):
            t = pg.TextItem(text="", anchor=(0.5, 0.5), color=(255, 255, 255))
            t.setFont(QFont("Arial", 7))
            self._pw.addItem(t)
            self._hard_items.append(t)
        # Head-3: suspension height inside bar (Overhead only, yellow)
        self._height_items = []
        for i in range(self.N):
            t = pg.TextItem(text="", anchor=(0.5, 0.5), color=(255, 230, 80))
            t.setFont(QFont("Arial", 7))
            self._pw.addItem(t)
            self._height_items.append(t)
        # Delta labels: "SC→SA" shown above SA bars when class changed (orange)
        self._delta_items = []
        for i in range(self.N):
            t = pg.TextItem(text="", anchor=(0.5, 0.0), color=(255, 120, 0))
            t.setFont(QFont("Arial", 7))
            self._pw.addItem(t)
            self._delta_items.append(t)
        self._pw.scene().sigMouseClicked.connect(self._on_scene_click)
        lay.addWidget(self._pw)

    def update(self, class_ids, confs, selected_ch_abs,
               hardness=None, suspension_height=None,
               sc_class_ids=None, sc_confs=None):
        for local in range(self.N):
            ch    = local + self._offset
            h     = float(confs[ch])
            cls   = int(class_ids[ch])
            color = self._qcolors[cls] if cls < len(self._qcolors) else QColor("#BDBDBD")

            # Determine bar pen and delta label
            delta_text = ""
            if sc_class_ids is not None:
                sc_cls = int(sc_class_ids[ch])
                sc_h   = float(sc_confs[ch]) if sc_confs is not None else h
                if cls != sc_cls:
                    pen = _PEN_SA_DIFF
                    sc_a = self._abbrs[sc_cls] if sc_cls < len(self._abbrs) else "?"
                    sa_a = self._abbrs[cls]    if cls    < len(self._abbrs) else "?"
                    delta_text = f"{sc_a}→{sa_a}"
                else:
                    pen = _PEN_SEL if ch == selected_ch_abs else _PEN_NORM
                    diff = h - sc_h
                    if abs(diff) > 0.10:
                        delta_text = "↑" if diff > 0 else "↓"
            else:
                pen = _PEN_SEL if ch == selected_ch_abs else _PEN_NORM

            self._bar_items[local].setOpts(height=[h], brush=pg.mkBrush(color), pen=pen)
            abbr = self._abbrs[cls] if cls < len(self._abbrs) else "?"
            self._text_items[local].setText(abbr)
            self._text_items[local].setPos(local, min(h + 0.04, 1.05))

            # Delta label above class abbreviation
            self._delta_items[local].setText(delta_text)
            if delta_text:
                self._delta_items[local].setPos(local, min(h + 0.13, 1.18))

            # Head-2: hardness
            h_val = float(hardness[ch]) if hardness is not None else -1.0
            if h_val > 0.0 and h >= 0.30 and cls != 4:
                self._hard_items[local].setText(f"H:{h_val:.2f}")
                self._hard_items[local].setPos(local, h * 0.65)
            else:
                self._hard_items[local].setText("")
            # Head-3: suspension height (Overhead only)
            s_val = float(suspension_height[ch]) if suspension_height is not None else -1.0
            if cls == 6 and s_val >= 0.0 and h >= 0.45:
                self._height_items[local].setText(f"\u21d5{s_val:.2f}m")
                self._height_items[local].setPos(local, h * 0.30)
            else:
                self._height_items[local].setText("")

    def clear(self):
        for local in range(self.N):
            self._bar_items[local].setOpts(height=[0.0],
                                           brush=pg.mkBrush("#BDBDBD"), pen=_PEN_NORM)
            self._text_items[local].setText("")
            self._hard_items[local].setText("")
            self._height_items[local].setText("")
            self._delta_items[local].setText("")

    def _on_scene_click(self, event):
        pos   = self._pw.plotItem.vb.mapSceneToView(event.scenePos())
        local = int(round(pos.x()))
        if 0 <= local < self.N and self._click_callback:
            self._click_callback(local + self._offset)


class WindowClassification(QWidget):
    # Emitted when the user clicks any bar; carries the channel index (0-11)
    channel_clicked = pyqtSignal(int)

    def __init__(self, cfg: ConfigLoader, mode: str = "ai", parent=None):
        super().__init__(parent)
        self.cfg  = cfg
        self._mode = mode
        self._selected_ch = 0
        self._last_result = None
        # SA dual-bar mode: only for AI instance when config flag is set
        self._sa_enabled = (mode == "ai") and cfg.get("ai.m2_sa_enabled", False)
        # Both modes use the same unified 9-class taxonomy
        self._class_colors = CLASS_COLORS
        self._class_names  = CLASS_NAMES
        self._class_abbr   = CLASS_ABBR
        self._sensor_labels = [
            cfg.get_sensor_config(f"S{i:02d}").get("label", f"S{i:02d}")
            for i in range(1, 13)
        ]
        self.setWindowTitle("W3 - Obstacle Classification")
        self.resize(800, 480)
        self._build_ui()

    def _build_ui(self):
        # Always 2-row layout (Front / Rear) regardless of SA mode.
        # SA comparison is shown inline via delta labels and orange bar borders
        # when SA changes SC's classification — no extra rows needed.
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)
        front_title = "Front  (S01-S06)" + (" ▏SA" if self._sa_enabled else "")
        rear_title  = "Rear   (S07-S12)" + (" ▏SA" if self._sa_enabled else "")
        self._row_front = _BarRow(self._sensor_labels[0:6], front_title,
                                   self._class_colors, self._class_abbr, ch_offset=0)
        self._row_rear  = _BarRow(self._sensor_labels[6:12], rear_title,
                                   self._class_colors, self._class_abbr, ch_offset=6)
        self._row_front._click_callback = self._on_channel_clicked
        self._row_rear._click_callback  = self._on_channel_clicked
        root.addWidget(self._row_front, stretch=1)
        root.addWidget(self._row_rear,  stretch=1)

    def _update_bar(self, result):
        cids  = result.class_ids
        confs = np.array([result.class_probs[i, cids[i]] for i in range(12)], dtype=float)
        hard  = result.material_hardness
        susp  = result.suspension_height_m
        if self._sa_enabled:
            sc_ids   = result.sc_class_ids
            sc_confs = np.array(
                [result.sc_class_probs[i, sc_ids[i]] for i in range(12)], dtype=float)
            # SA final output; bars with orange border + "SC→SA" delta label
            # where SA changed SC's classification
            self._row_front.update(cids, confs, self._selected_ch, hard, susp,
                                   sc_class_ids=sc_ids, sc_confs=sc_confs)
            self._row_rear.update( cids, confs, self._selected_ch, hard, susp,
                                   sc_class_ids=sc_ids, sc_confs=sc_confs)
        else:
            self._row_front.update(cids, confs, self._selected_ch, hard, susp)
            self._row_rear.update( cids, confs, self._selected_ch, hard, susp)

    def _on_channel_clicked(self, ch):
        self._selected_ch = ch
        self.channel_clicked.emit(ch)  # propagate to W6 zoom view
        if self._last_result is not None:
            self._update_bar(self._last_result)

    def on_result_updated(self, result: AlgoResult) -> None:
        self._last_result = result
        self._update_bar(result)

    def reset(self) -> None:
        self._last_result = None
        self._selected_ch = 0
        self._row_front.clear()
        self._row_rear.clear()
