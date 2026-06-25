"""
window_display.py  —  Integrated display window
Embeds W2 (envelope), W3 (classification), W4 (OGM), W6 (statistics)
into a full-screen QMainWindow using a splitter grid layout:

  ┌─────────────────────────┬──────────────┐
  │  W2 Envelope (12ch)     │              │
  ├─────────────────────────│  W4 OGM      │
  │  W3 Classification      │  (full height│
  ├─────────────────────────│  = W1 height)│
  │  W6 Performance Stats    │              │
  └─────────────────────────┴──────────────┘
  W2 / W3 / W6 share the same width.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QSplitter,
    QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt

from modules.config_loader import ConfigLoader
from modules.result_fusion import AlgoResult
from modules.data_manager import DataFrame
import numpy as np

from views.window_envelope import WindowEnvelope
from views.window_classification import WindowClassification
from views.window_ogm import WindowOGM
from views.window_channel_zoom import WindowChannelZoom
from views.window_pointcloud import WindowPointCloud


class DisplayWindow(QMainWindow):
    """
    Integrated display window for a single algorithm instance, containing W2~W6 sub-views.
    Call update_frame() to refresh all sub-views simultaneously.
    """

    def __init__(self, cfg: ConfigLoader, mode: str = "traditional", parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.mode = mode

        label = "Traditional" if mode == "traditional" else "AI"
        self.setWindowTitle(f"AK2 Simulator — {label} Display Panel")
        self.setMinimumSize(600, 500)

        # ── Create sub-windows (embedded as Widgets) ──
        self.w2 = WindowEnvelope(cfg, mode=mode)
        self.w3 = WindowClassification(cfg, mode=mode)
        self.w4 = WindowOGM(cfg, mode=mode)
        self.w5 = WindowPointCloud(cfg, mode=mode)
        self.w6 = WindowChannelZoom(cfg)

        # ── Wire W2/W3 channel-click signals → W6 zoom view ──
        self.w2.channel_clicked.connect(self.w6.set_channel)
        self.w3.channel_clicked.connect(self.w6.set_channel)

        # ── Layout ──
        self._build_layout()

    def _build_layout(self):
        root = QWidget()
        self.setCentralWidget(root)

        # Left column: W2 / W3 / W6 stacked vertically (all same width)
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.addWidget(self._wrap(self.w2, "W2  Envelope Waveform"))
        left_splitter.addWidget(self._wrap(self.w3, "W3  Obstacle Classification"))
        left_splitter.addWidget(self._wrap(self.w6, "W6  Channel Zoom"))
        left_splitter.setSizes([470, 390, 415])   # ≈ 37% / 31% / 32% of 1275px

        # Right column: W4 (top) over W5 (bottom). W5 mirrors W6's height so the
        # OGM no longer occupies the full vertical span.
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setChildrenCollapsible(False)
        w4_wrap = self._wrap(self.w4, "W4  Occupancy Grid Map (OGM)")
        # W4 is an aspect-locked spatial map; guarantee enough width so it never
        # collapses into a thin strip (which forces the OGM to auto-zoom out).
        w4_wrap.setMinimumWidth(440)
        w5_wrap = self._wrap(self.w5, "W5  Echo Point Cloud (3D)")
        right_splitter.addWidget(w4_wrap)
        right_splitter.addWidget(w5_wrap)
        # W4 takes the upper part; W5 gets a taller lower band so its bottom
        # edge lines up with W6 / the control panel.
        right_splitter.setSizes([770, 505])

        # Horizontal splitter: left column | right column (W4 over W5)
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        h_splitter.setChildrenCollapsible(False)
        h_splitter.addWidget(left_splitter)
        h_splitter.addWidget(right_splitter)
        h_splitter.setSizes([620, 560])           # ≈ 53% / 47% — give the map column a near-equal share
        h_splitter.setStretchFactor(0, 1)         # left column absorbs extra width on resize
        h_splitter.setStretchFactor(1, 1)         # map column keeps its share when the panel grows

        # Root layout
        layout = QGridLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)
        layout.addWidget(h_splitter, 0, 0)

    @staticmethod
    def _wrap(widget: QWidget, title: str) -> QWidget:
        """Wrap a sub-widget with a title label in a container."""
        container = QWidget()
        container.setObjectName("SubPanel")
        container.setStyleSheet(
            "#SubPanel { border: 1px solid #444; border-radius: 3px; }"
        )
        vbox_layout = __import__("PyQt6.QtWidgets", fromlist=["QVBoxLayout"]).QVBoxLayout(container)
        vbox_layout.setContentsMargins(2, 2, 2, 2)
        vbox_layout.setSpacing(2)

        title_label = QLabel(f"  {title}")
        title_label.setFixedHeight(22)
        title_label.setStyleSheet(
            "background:#2a2a40; color:#aad4ff; font-size:11px; font-weight:bold;"
            "border-radius:2px; padding-left:4px;"
        )

        vbox_layout.addWidget(title_label)
        vbox_layout.addWidget(widget)
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return container

    # ── Unified refresh interface ──────────────────────────────

    def update_frame(self,
                     frame: "DataFrame",
                     result: "AlgoResult") -> None:
        """Called by main app to refresh all sub-views at once."""
        self.w2.update_frame(frame.envelopes, frame.edi_distance, result.valid_flags)
        self.w3.on_result_updated(result)
        self.w4.update_ogm(result, frame)
        self.w5.update_frame(frame.envelopes)
        self.w6.update_frame(frame.envelopes, frame.edi_distance, result.valid_flags, result)

    def update_with_gt(self, result: "AlgoResult", gt_ids: np.ndarray) -> None:
        """No-op: GT evaluation statistics panel has been replaced by W6 channel zoom."""
        pass

    def set_session(self, session_id: str, total_frames: int,
                    num_classes: int, engine_label: str,
                    has_gt: bool = True) -> None:
        """Reset all display panels for the new session."""
        self.w2.reset()
        self.w3.reset()
        self.w4.reset()
        self.w5.reset()
        self.w6.reset()

    def on_result_updated(self, result: "AlgoResult") -> None:
        """Compatibility callback (no-op; W6 is now updated via update_frame)."""
        pass

    def showMaximized(self) -> None:  # noqa
        super().showMaximized()
