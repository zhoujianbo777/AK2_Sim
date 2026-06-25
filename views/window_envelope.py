"""
window_envelope.py  —  W2 Envelope Waveform View
2 rows × 6 cols: Front row (ch0~ch5), Rear row (ch6~ch11), left-to-right in bird's-eye view.
See spec section 5.2.2.

Background color coding (based on M1 validity):
  valid_flag >= threshold  →  orange  #FF8C00  (obstacle detected)
  valid_flag <  threshold  →  transparent      (no obstacle)
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont

from modules.config_loader import ConfigLoader
from modules.result_fusion import AlgoResult

# Validity background colors
_COLOR_VALID   = QColor(255, 140, 0, 70)   # orange — obstacle detected
_COLOR_INVALID = QColor(0, 0, 0, 0)        # transparent — no obstacle

ENVELOPE_POINTS = 256


class EnvelopeSubplot:
    """Single-channel envelope subplot."""

    def __init__(self, plot_widget: pg.PlotWidget, sensor_label: str, history_frames: int,
                 mode: str = "ai", threshold: float = 0.5):
        self._pw = plot_widget
        self._label = sensor_label
        self._history = history_frames
        self._threshold = threshold
        self._click_callback = None  # set by WindowEnvelope to emit channel_clicked

        self._pw.setBackground("w")

        # Hide axis labels to save space; sensor name shown as in-plot text
        self._pw.getAxis("left").setWidth(36)
        self._pw.getAxis("bottom").setHeight(20)
        self._pw.getAxis("left").setStyle(tickTextOffset=2)
        self._pw.getAxis("bottom").setStyle(tickTextOffset=2)
        # Remove axis title labels (they eat too much space)
        self._pw.setLabel("left", "")
        self._pw.setLabel("bottom", "")

        # Sensor name as plot title (always anchored to top of frame)
        self._pw.setTitle(sensor_label, size="8pt", color="#333333")

        self._pw.setYRange(0, 1)
        self._pw.setXRange(0, ENVELOPE_POINTS - 1)
        self._pw.showGrid(x=False, y=True, alpha=0.2)

        # Current frame main curve (blue solid)
        self._curve = self._pw.plot(pen=pg.mkPen("#1565C0", width=2))

        # Historical frame semi-transparent curves (up to 5)
        self._history_curves = [
            self._pw.plot(pen=pg.mkPen(QColor(100, 149, 237, 60), width=1))
            for _ in range(history_frames)
        ]

        # Peak annotation (red triangle)
        self._peak_scatter = pg.ScatterPlotItem(
            symbol="t", size=12, pen=None, brush=pg.mkBrush("#F44336")
        )
        self._pw.addItem(self._peak_scatter)

        # Background rect (class color)
        self._bg = pg.LinearRegionItem(
            values=(0, ENVELOPE_POINTS - 1),
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(QColor(255, 255, 255, 40))
        )
        self._pw.addItem(self._bg, ignoreBounds=True)

        self._history_envelopes: list[np.ndarray] = []

        # Wire click on this subplot to fire the channel-selection callback
        self._pw.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, _event) -> None:
        """Any click on this subplot selects the corresponding channel."""
        if self._click_callback is not None:
            self._click_callback()

    def update(self, envelope: np.ndarray, valid_flag: float, distance_m: float) -> None:
        """Refresh current frame envelope and validity background color."""
        x = np.arange(ENVELOPE_POINTS)
        self._curve.setData(x, envelope)

        # History curves
        for i, hc in enumerate(self._history_curves):
            if i < len(self._history_envelopes):
                alpha = int(60 * (1 - i / max(len(self._history_envelopes), 1)))
                hc.setPen(pg.mkPen(QColor(100, 149, 237, alpha), width=1))
                hc.setData(x, self._history_envelopes[i])
            else:
                hc.setData([], [])

        # Update history queue
        self._history_envelopes.insert(0, envelope.copy())
        if len(self._history_envelopes) > self._history:
            self._history_envelopes.pop()

        # Peak annotation
        peak_idx = int(np.argmax(envelope))
        peak_val = float(envelope[peak_idx])
        if peak_val > 0.05:
            self._peak_scatter.setData(
                [{"pos": (peak_idx, peak_val), "data": distance_m}]
            )
        else:
            self._peak_scatter.setData([])

        # Background color — orange if obstacle detected, transparent otherwise
        self._bg.setBrush(pg.mkBrush(_COLOR_VALID if valid_flag >= self._threshold else _COLOR_INVALID))


class WindowEnvelope(QWidget):
    """W2 Envelope Waveform View: 2 rows × 6 cols (Front / Rear), left-to-right."""

    # Emitted when the user clicks any subplot; carries the channel index (0-11)
    channel_clicked = pyqtSignal(int)

    # Each row: (row_label, [channel_indices_left_to_right])
    # New numbering (left→right in bird's-eye view):
    # Front row ch0~ch5:  S01=FL-Side, S02=FL-Corner, S03=FL-Center, S04=FR-Center, S05=FR-Corner, S06=FR-Side
    # Rear  row ch6~ch11: S07=RL-Side, S08=RL-Corner, S09=RL-Center, S10=RR-Center, S11=RR-Corner, S12=RR-Side
    ROW_GROUPS = [
        ("Front  —  FL-Side | FL-Corner | FL-Center | FR-Center | FR-Corner | FR-Side",
         [0, 1, 2, 3, 4, 5]),    # S01~S06
        ("Rear   —  RL-Side | RL-Corner | RL-Center | RR-Center | RR-Corner | RR-Side",
         [6, 7, 8, 9, 10, 11]),  # S07~S12
    ]

    def __init__(self, cfg: ConfigLoader, mode: str = "ai", parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._mode = mode
        self._history_frames = cfg.get("display.envelope_history_frames", 5)
        self._m1_threshold = cfg.get("ai.m1_validity_threshold", 0.5)
        self._subplots: list[EnvelopeSubplot] = []          # ordered S01..S12
        self._sensor_labels: list[str] = []

        # Load sensor labels
        for sid in [f"S{i:02d}" for i in range(1, 13)]:
            sc = cfg.get_sensor_config(sid)
            self._sensor_labels.append(sc.get("label", sid))

        self.setWindowTitle("W2 - Envelope Waveform View")
        w = cfg.get("windows.envelope.width", 1400)
        h = cfg.get("windows.envelope.height", 560)
        self.resize(w, h)
        self._build_ui()

    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setSpacing(2)
        grid.setContentsMargins(4, 4, 4, 4)

        font = QFont()
        font.setBold(True)
        font.setPointSize(8)

        plot_row = 0
        for row_label, indices in self.ROW_GROUPS:
            # Row header label (spans all 6 columns)
            lbl = QLabel(row_label)
            lbl.setFont(font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl.setStyleSheet("color: #444; padding-left: 4px;")
            lbl.setFixedHeight(18)
            grid.addWidget(lbl, plot_row, 0, 1, 6)
            plot_row += 1

            for col, sensor_idx in enumerate(indices):
                pw = pg.PlotWidget()
                pw.setMinimumHeight(100)
                pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                sp = EnvelopeSubplot(pw, self._sensor_labels[sensor_idx], self._history_frames,
                                     self._mode, self._m1_threshold)
                # Keep _subplots indexed by sensor order (S01=0 .. S12=11)
                while len(self._subplots) <= sensor_idx:
                    self._subplots.append(None)
                self._subplots[sensor_idx] = sp
                # Wire subplot click → WindowEnvelope.channel_clicked signal
                sp._click_callback = (lambda _ch=sensor_idx: self.channel_clicked.emit(_ch))
                grid.addWidget(pw, plot_row, col)

            plot_row += 1

    def on_result_updated(self, result: AlgoResult) -> None:
        """Observer callback: called by result fusion to refresh (no-op here)."""
        pass  # requires frame data; called by main application

    def reset(self) -> None:
        """Clear all channel displays when switching sessions."""
        blank_env = np.zeros((12, 256), dtype=np.float32)
        blank_dist = np.zeros(12, dtype=np.float32)
        blank_valid = np.zeros(12, dtype=np.float32)
        self.update_frame(blank_env, blank_dist, blank_valid)

    def update_frame(self, envelopes: np.ndarray, distances: np.ndarray,
                     valid_flags: np.ndarray) -> None:
        """
        Refresh all 12 channel envelope displays.
        envelopes:   [12, 256] float32
        distances:   [12] float32 (meters)
        valid_flags: [12] float32 (M1 validity probability, 0~1)
        """
        for i, sp in enumerate(self._subplots):
            sp.update(envelopes[i], float(valid_flags[i]), float(distances[i]))
