"""
window_collision.py  —  W5 Collision Analysis View
Top: elastic wave RMS energy time series (last 50 frames)
Bottom: 4-class collision dashboard + M4 VAE anomaly score
See spec section 5.2.5.
"""

import numpy as np
import pyqtgraph as pg
from collections import deque
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

from modules.result_fusion import AlgoResult
from modules.config_loader import ConfigLoader

COLLISION_NAMES = ["No Collision", "Light Scrape", "Soft Hit", "Hard Impact"]
COLLISION_COLORS = ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336"]
WINDOW_FRAMES = 50  # time-series lookback frames

# 12-channel sensor line colors
CHANNEL_COLORS = [
    "#F44336", "#E91E63", "#9C27B0", "#673AB7",
    "#3F51B5", "#2196F3", "#03A9F4", "#00BCD4",
    "#009688", "#4CAF50", "#8BC34A", "#CDDC39",
]


class WindowCollision(QWidget):
    """W5 Collision Analysis View."""

    def __init__(self, cfg: ConfigLoader, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._time_window = cfg.get("display.collision_time_window_frames", WINDOW_FRAMES)
        # Elastic wave RMS history (one deque per channel)
        self._rms_history: list[deque] = [deque(maxlen=self._time_window) for _ in range(12)]
        self._threshold = cfg.get("traditional.elastic_rms_threshold", 0.5)

        # Load sensor position labels from config (ch0=S01, ch1=S02, ...)
        self._sensor_labels = [
            cfg.get_sensor_config(f"S{i:02d}").get("label", f"S{i:02d}")
            for i in range(1, 13)
        ]

        self.setWindowTitle("W5 - Collision Analysis")
        self.resize(620, 380)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Top: elastic wave time series ──
        grp_ts = QGroupBox("Elastic Wave RMS Energy Time Series (Last 50 Frames)")
        ts_lay = QVBoxLayout(grp_ts)
        self._pw_ts = pg.PlotWidget()
        self._pw_ts.setBackground("#12191F")
        self._pw_ts.setLabel("left", "Normalized RMS")
        self._pw_ts.setLabel("bottom", "Frame (latest)")
        self._pw_ts.setYRange(0, 1.1)
        self._pw_ts.showGrid(x=False, y=True, alpha=0.2)

        # Collision threshold reference line
        self._threshold_line = pg.InfiniteLine(
            pos=self._threshold, angle=0,
            pen=pg.mkPen("#FF9800", width=1, style=Qt.PenStyle.DashLine),
            label=f"Threshold {self._threshold:.2f}",
            labelOpts={"color": "#FF9800", "position": 0.95}
        )
        self._pw_ts.addItem(self._threshold_line)

        # 12-channel curves
        self._ts_curves = []
        for ch in range(12):
            curve = self._pw_ts.plot(
                pen=pg.mkPen(QColor(CHANNEL_COLORS[ch]), width=1.5),
                name=self._sensor_labels[ch]
            )
            self._ts_curves.append(curve)

        ts_lay.addWidget(self._pw_ts)
        root.addWidget(grp_ts, stretch=2)

        # ── Bottom: collision type dashboard ──
        grp_dash = QGroupBox("Current Frame Collision Decision")
        dash_lay = QHBoxLayout(grp_dash)

        self._collision_indicators: list[QLabel] = []
        for i, (name, color) in enumerate(zip(COLLISION_NAMES, COLLISION_COLORS)):
            indicator = QLabel(f"{name}\n--%")
            indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
            indicator.setFixedSize(110, 60)
            indicator.setStyleSheet(
                f"border: 2px solid {color}; border-radius: 6px; color: #AAA; font-size: 12px;"
            )
            self._collision_indicators.append(indicator)
            dash_lay.addWidget(indicator)

        # VAE anomaly score progress bar
        anom_col = QVBoxLayout()
        anom_col.addWidget(QLabel("M4 Anomaly Score"))
        self._anomaly_bar = QProgressBar()
        self._anomaly_bar.setRange(0, 100)
        self._anomaly_bar.setValue(0)
        self._anomaly_bar.setTextVisible(True)
        self._anomaly_bar.setFixedHeight(24)
        self._anomaly_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4CAF50, stop:0.5 #FFEB3B, stop:1 #F44336); }
        """)
        anom_col.addWidget(self._anomaly_bar)
        anom_col.addStretch()
        dash_lay.addLayout(anom_col)

        root.addWidget(grp_dash, stretch=1)

    def on_result_updated(self, result: AlgoResult) -> None:
        """Observer callback: receive latest inference result."""
        self._update_display(result)

    def reset(self) -> None:
        """Clear RMS history and curves when switching sessions."""
        for dq in self._rms_history:
            dq.clear()
        for curve in self._ts_curves:
            curve.setData([], [])

    def update_with_frame(self, result: AlgoResult, elastic_features: np.ndarray) -> None:
        """
        Refresh collision view.
        elastic_features: [12, 20] float32
        """
        # Compute per-channel RMS
        rms_per_ch = np.sqrt(np.mean(elastic_features ** 2, axis=1))  # [12]
        for ch in range(12):
            self._rms_history[ch].append(float(rms_per_ch[ch]))

        # Update time series curves
        n = len(self._rms_history[0])
        x = np.arange(n)
        for ch in range(12):
            y = np.array(list(self._rms_history[ch]), dtype=float)
            self._ts_curves[ch].setData(x, y)

        self._update_display(result)

    def _update_display(self, result: AlgoResult) -> None:
        """Refresh collision type dashboard and anomaly score."""
        ctype = int(result.collision_type)
        cprobs = result.collision_probs

        for i, (indicator, color) in enumerate(zip(self._collision_indicators, COLLISION_COLORS)):
            pct = int(cprobs[i] * 100) if i < len(cprobs) else 0
            indicator.setText(f"{COLLISION_NAMES[i]}\n{pct}%")
            if i == ctype:
                indicator.setStyleSheet(
                    f"border: 2px solid {color}; border-radius: 6px; "
                    f"background-color: {color}33; color: white; font-weight: bold; font-size: 12px;"
                )
            else:
                indicator.setStyleSheet(
                    f"border: 2px solid {color}; border-radius: 6px; color: #AAA; font-size: 12px;"
                )

        anom_pct = int(result.anomaly_score * 100)
        self._anomaly_bar.setValue(anom_pct)
        self._anomaly_bar.setFormat(f"Anomaly: {anom_pct}%")
