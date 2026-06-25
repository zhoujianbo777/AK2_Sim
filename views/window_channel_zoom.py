"""
window_channel_zoom.py  —  W6 Channel Zoom View

Shows the full 256-point envelope of the currently selected channel with rich
annotations for debugging purposes.  Channel selection is driven by clicks in
W2 (envelope grid) or W3 (classification bar chart).

Layout:
  ┌─ info bar: channel name │ dist / valid / class ──────────────────────────┐
  │                                                                           │
  │              large pyqtgraph envelope plot                                │
  │  history traces (faded blue)                                              │
  │  current frame  (solid blue)                                              │
  │  peak marker    (red triangle)                                            │
  │  edi dist line  (green dashed vertical)                                   │
  │                                                                           │
  └───────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from modules.config_loader import ConfigLoader
from modules.obstacle_classes import CLASS_NAMES, CLASS_COLORS

ENVELOPE_POINTS = 256
_MAX_HISTORY    = 5          # number of previous-frame traces to keep


class WindowChannelZoom(QWidget):
    """
    W6 Channel Zoom View: full 256-point envelope for one selected channel.
    Automatically refreshed on every frame; channel selection comes from
    W2/W3 click signals wired in window_display.py.
    """

    def __init__(self, cfg: ConfigLoader, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._selected_ch: int = 0
        self._sensor_labels: list[str] = []

        # Per-frame history: list of full [12, 256] envelope arrays (newest first)
        self._history_all: list[np.ndarray] = []

        # Last data received (needed for immediate redraw on channel change)
        self._last_envelopes:   np.ndarray | None = None
        self._last_distances:   np.ndarray | None = None
        self._last_valid_flags: np.ndarray | None = None
        self._last_result                         = None

        # Load sensor labels (S01..S12)
        for sid in [f"S{i:02d}" for i in range(1, 13)]:
            sc = cfg.get_sensor_config(sid)
            self._sensor_labels.append(sc.get("label", sid))

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)

        # ── Info bar ──────────────────────────────────────────────────────
        info_bar = QHBoxLayout()
        info_bar.setSpacing(8)

        self._lbl_channel = QLabel("  Ch01  —")
        self._lbl_channel.setStyleSheet(
            "color:#aad4ff; font-weight:bold; font-size:12px;"
        )

        self._lbl_class = QLabel("")
        self._lbl_class.setStyleSheet(
            "color:#ffffff; font-weight:bold; font-size:11px;"
            "padding: 1px 6px; border-radius: 3px;"
        )

        self._lbl_info = QLabel("")
        self._lbl_info.setStyleSheet("color:#aaaaaa; font-size:10px;")
        self._lbl_info.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        info_bar.addWidget(self._lbl_channel)
        info_bar.addWidget(self._lbl_class)
        info_bar.addStretch()
        info_bar.addWidget(self._lbl_info)
        root.addLayout(info_bar)

        # ── Main envelope plot ────────────────────────────────────────────
        self._pw = pg.PlotWidget()
        self._pw.setBackground("#1a1a2e")
        self._pw.setYRange(0, 1.05)
        self._pw.setXRange(0, ENVELOPE_POINTS - 1)
        self._pw.showGrid(x=True, y=True, alpha=0.25)
        self._pw.setMenuEnabled(False)
        self._pw.setMouseEnabled(x=False, y=False)

        # Axis styling
        for axis_name in ("left", "bottom"):
            ax = self._pw.getAxis(axis_name)
            ax.setTextPen(pg.mkPen("#888888"))
            ax.setPen(pg.mkPen("#555555"))
        self._pw.getAxis("left").setLabel("Amplitude (normalized)", color="#888888")
        self._pw.getAxis("bottom").setLabel("Sample index", color="#888888")
        self._pw.getAxis("left").setWidth(52)
        self._pw.getAxis("bottom").setHeight(28)

        # History traces (oldest = most transparent)
        self._history_curves: list[pg.PlotDataItem] = [
            self._pw.plot(
                pen=pg.mkPen(QColor(80, 140, 255, 0), width=1)
            )
            for _ in range(_MAX_HISTORY)
        ]

        # Current-frame curve
        self._curve = self._pw.plot(
            pen=pg.mkPen(QColor(100, 180, 255), width=2)
        )

        # Peak marker (red downward-pointing triangle)
        self._peak_scatter = pg.ScatterPlotItem(
            symbol="t1", size=14, pen=None,
            brush=pg.mkBrush(QColor(244, 67, 54, 220))
        )
        self._pw.addItem(self._peak_scatter)

        # EDI distance line (green dashed vertical)
        self._dist_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen(
                QColor(76, 175, 80, 180), width=1,
                style=Qt.PenStyle.DashLine
            )
        )
        self._pw.addItem(self._dist_line)

        # Distance label anchored to the distance line
        self._dist_label = pg.TextItem(
            text="", anchor=(0.0, 1.0),
            color=QColor(76, 175, 80)
        )
        self._dist_label.setFont(QFont("Arial", 8))
        self._pw.addItem(self._dist_label)

        self._pw.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self._pw)

    # ── Public interface ───────────────────────────────────────────────────

    def set_channel(self, ch: int) -> None:
        """
        Switch the zoom view to channel *ch*.
        Called when the user clicks a probe in W2 or a bar in W3.
        History is preserved (the stored per-frame arrays cover all channels).
        """
        if ch == self._selected_ch:
            return
        self._selected_ch = ch
        # Immediate redraw without waiting for next frame
        if self._last_envelopes is not None:
            self._refresh()

    def update_frame(
        self,
        envelopes:   np.ndarray,   # [12, 256] float32
        distances:   np.ndarray,   # [12]      float32  (metres, from EDI)
        valid_flags: np.ndarray,   # [12]      float32  (M1 validity 0–1)
        result=None,               # AlgoResult or None
    ) -> None:
        """
        Called every frame (from DisplayWindow.update_frame).
        Pushes the previous frame into history and redraws.
        """
        # Archive previous frame before overwriting
        if self._last_envelopes is not None:
            self._history_all.insert(0, self._last_envelopes.copy())
            if len(self._history_all) > _MAX_HISTORY:
                self._history_all.pop()

        self._last_envelopes   = envelopes
        self._last_distances   = distances
        self._last_valid_flags = valid_flags
        self._last_result      = result
        self._refresh()

    def reset(self) -> None:
        """Clear all display state when switching sessions."""
        self._history_all.clear()
        self._last_envelopes   = None
        self._last_distances   = None
        self._last_valid_flags = None
        self._last_result      = None
        self._curve.setData([], [])
        for hc in self._history_curves:
            hc.setData([], [])
        self._peak_scatter.setData([])
        self._dist_line.setValue(0)
        self._dist_label.setText("")
        self._lbl_channel.setText("  Ch01  —")
        self._lbl_class.setText("")
        self._lbl_info.setText("")

    # ── Internal rendering ─────────────────────────────────────────────────

    def _refresh(self) -> None:
        if self._last_envelopes is None:
            return

        ch  = self._selected_ch
        env = self._last_envelopes[ch]          # [256]
        dist  = float(self._last_distances[ch]) if self._last_distances is not None else 0.0
        valid = float(self._last_valid_flags[ch]) if self._last_valid_flags is not None else 0.0

        x = np.arange(ENVELOPE_POINTS)

        # ── History traces (oldest = most transparent) ─────────────────
        for i, hc in enumerate(self._history_curves):
            if i < len(self._history_all):
                # alpha: 50 → 10 as we go back in time
                alpha = max(10, 50 - i * 10)
                hc.setPen(pg.mkPen(QColor(80, 140, 255, alpha), width=1))
                hc.setData(x, self._history_all[i][ch])
            else:
                hc.setData([], [])

        # ── Current-frame envelope ──────────────────────────────────────
        self._curve.setData(x, env)

        # ── Peak marker ─────────────────────────────────────────────────
        peak_idx = int(np.argmax(env))
        peak_val = float(env[peak_idx])
        if peak_val > 0.05:
            self._peak_scatter.setData([{"pos": (peak_idx, peak_val + 0.04)}])
        else:
            self._peak_scatter.setData([])

        # ── EDI distance line ────────────────────────────────────────────
        # Show the green dashed line at peak_idx (the argmax position) which
        # is the best proxy we have for the EDI-reported range bin.
        if peak_val > 0.05:
            self._dist_line.setValue(peak_idx)
            self._dist_label.setText(f"  {dist:.2f} m")
            self._dist_label.setPos(peak_idx, 0.96)
        else:
            self._dist_line.setValue(0)
            self._dist_label.setText("")

        # ── Info bar ─────────────────────────────────────────────────────
        ch_label = self._sensor_labels[ch] if ch < len(self._sensor_labels) else f"Ch{ch}"
        self._lbl_channel.setText(f"  Ch{ch + 1:02d}  {ch_label}")

        # Class + confidence from AlgoResult (if available)
        cls_id   = -1
        cls_name = ""
        conf     = 0.0
        if self._last_result is not None:
            try:
                cls_id   = int(self._last_result.class_ids[ch])
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "?"
                conf     = float(self._last_result.class_probs[ch, cls_id])
            except Exception:
                pass

        # Class badge with the class's canonical color
        if cls_name:
            hex_color = CLASS_COLORS[cls_id] if 0 <= cls_id < len(CLASS_COLORS) else "#888888"
            self._lbl_class.setText(f" {cls_name} ({conf:.2f}) ")
            self._lbl_class.setStyleSheet(
                f"color:#ffffff; font-weight:bold; font-size:11px;"
                f"background:{hex_color}; padding:1px 6px; border-radius:3px;"
            )
        else:
            self._lbl_class.setText("")

        info_parts = [f"dist {dist:.2f} m", f"valid {valid:.2f}"]
        self._lbl_info.setText("    ".join(info_parts))
