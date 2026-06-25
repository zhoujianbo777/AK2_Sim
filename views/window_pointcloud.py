"""
window_pointcloud.py  —  W5 Echo Point-Cloud View (3D)

Renders the raw envelope matrix [12 sensors × 256 samples] as a lidar-style
3D point cloud so the echo energy distribution across all probes can be
inspected at a glance:

    X axis  →  probe / sensor index   (探头编号 0~11)
    Y axis  →  echo distance sample    (回波距离采样点编号 0~255)
    Z axis  →  echo intensity          (回波强度 amplitude 0~1)

Each point is colored by its amplitude (blue → green → yellow → red) and
strong echoes are drawn larger, making cross-probe energy ridges obvious.
"""

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtGui import QColor, QFont

from modules.config_loader import ConfigLoader

N_SENSORS = 12
ENVELOPE_LEN = 256

# Display-space spans (the three physical ranges differ wildly, so they are
# normalized into a comfortable viewing box).
X_SPAN = 22.0     # probe axis
Y_SPAN = 22.0     # sample axis
Z_SPAN = 16.0     # intensity axis

# Sensor channel names — identical abbreviations to the W2 envelope window
# (front row ch0~5, rear row ch6~11).
_SENSOR_NAMES = [
    "FL-Side", "FL-Corner", "FL-Center", "FR-Center", "FR-Corner", "FR-Side",
    "RL-Side", "RL-Corner", "RL-Center", "RR-Center", "RR-Corner", "RR-Side",
]


class WindowPointCloud(QWidget):
    """W5 — 3D echo point cloud of the 12×256 envelope matrix."""

    def __init__(self, cfg: ConfigLoader, mode: str = "ai", parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.mode = mode

        # Amplitude → RGBA color map (jet-like, no external colormap dependency)
        self._cmap = pg.ColorMap(
            pos=[0.0, 0.25, 0.50, 0.75, 1.0],
            color=[
                (25, 25, 90, 255),     # deep blue  (weak)
                (0, 120, 210, 255),    # blue
                (0, 200, 120, 255),    # green
                (235, 210, 40, 255),   # yellow
                (240, 45, 45, 255),    # red        (strong)
            ],
        )

        self._init_coords()
        self._build_ui()

    # ── Precompute fixed X/Y coordinates ───────────────────────
    def _init_coords(self):
        n = N_SENSORS * ENVELOPE_LEN
        si = np.repeat(np.arange(N_SENSORS), ENVELOPE_LEN)      # 0..11
        pj = np.tile(np.arange(ENVELOPE_LEN), N_SENSORS)        # 0..255
        self._base_pos = np.zeros((n, 3), dtype=np.float32)
        self._base_pos[:, 0] = si / (N_SENSORS - 1) * X_SPAN
        self._base_pos[:, 1] = pj / (ENVELOPE_LEN - 1) * Y_SPAN

    # ── Build the GL scene ─────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(QColor(18, 18, 28))
        self.view.setCameraPosition(distance=54, elevation=22, azimuth=-45)
        layout.addWidget(self.view)

        # Floor grid spanning the probe/sample plane
        grid = gl.GLGridItem()
        grid.setSize(x=X_SPAN, y=Y_SPAN)
        grid.setSpacing(x=X_SPAN / N_SENSORS, y=Y_SPAN / 16)
        grid.translate(X_SPAN / 2, Y_SPAN / 2, 0)
        grid.setColor((90, 90, 120, 120))
        self.view.addItem(grid)

        self._add_axes()
        self._add_labels()

        # Per-probe waveform polylines (one 256-point line per sensor) so the
        # full envelope shape is visible just like W2, not only sparse peaks.
        self._lines = []
        for ch in range(N_SENSORS):
            seg = self._base_pos[ch * ENVELOPE_LEN:(ch + 1) * ENVELOPE_LEN].copy()
            line = gl.GLLinePlotItem(pos=seg, color=(0.4, 0.4, 0.4, 0.5),
                                     width=1.4, antialias=True, mode="line_strip")
            line.setGLOptions("translucent")
            self.view.addItem(line)
            self._lines.append(line)

        # Main point cloud (colored markers on top of the waveform lines)
        self.scatter = gl.GLScatterPlotItem(
            pos=self._base_pos.copy(),
            color=(0.5, 0.5, 0.5, 0.6),
            size=3.0,
        )
        self.scatter.setGLOptions("translucent")
        self.view.addItem(self.scatter)

    def _add_axes(self):
        """Draw colored X / Y / Z reference axes from the origin."""
        axes = [
            (np.array([[0, 0, 0], [X_SPAN, 0, 0]]), (1.0, 0.5, 0.5, 1.0)),  # X red
            (np.array([[0, 0, 0], [0, Y_SPAN, 0]]), (0.5, 1.0, 0.5, 1.0)),  # Y green
            (np.array([[0, 0, 0], [0, 0, Z_SPAN]]), (0.5, 0.7, 1.0, 1.0)),  # Z blue
        ]
        for pts, col in axes:
            line = gl.GLLinePlotItem(pos=pts.astype(np.float32),
                                     color=col, width=2.0, antialias=True)
            self.view.addItem(line)

    def _add_labels(self):
        """Add axis titles plus probe / sample / intensity tick labels."""
        if not hasattr(gl, "GLTextItem"):
            return
        tick_font = QFont("Arial", 7)
        title_font = QFont("Arial", 9)
        probe_font = QFont("Arial", 9)
        probe_font.setBold(True)
        try:
            # Axis titles
            titles = [
                ("X: 探头编号 Probe", (X_SPAN + 1.5, -4.8, 0.0), (255, 170, 170)),
                ("Y: 采样点 Sample", (-6.0, Y_SPAN + 0.5, 0.0), (170, 255, 170)),
                ("Z: 回波强度", (-4.5, -2.0, Z_SPAN), (170, 200, 255)),
            ]
            for text, pos, rgb in titles:
                t = gl.GLTextItem(pos=np.array(pos, dtype=np.float32),
                                  text=text, color=QColor(*rgb), font=title_font)
                self.view.addItem(t)

            # Per-probe tick labels along the X axis: index + W2 abbreviation.
            # Single row pushed clear of the waveform plane; the bold, brighter
            # font keeps all 12 labels legible at this camera angle.
            for i, name in enumerate(_SENSOR_NAMES):
                x = i / (N_SENSORS - 1) * X_SPAN
                t = gl.GLTextItem(
                    pos=np.array([x, -3.6, 0.0], dtype=np.float32),
                    text=f"{i} {name}",
                    color=QColor(240, 240, 255),
                    font=probe_font,
                )
                self.view.addItem(t)

            # Sample-index tick labels along the Y axis (0~255)
            for s in (0, 50, 100, 150, 200, 250):
                y = s / (ENVELOPE_LEN - 1) * Y_SPAN
                t = gl.GLTextItem(
                    pos=np.array([-2.2, y, 0.0], dtype=np.float32),
                    text=f"{s}",
                    color=QColor(170, 230, 170),
                    font=tick_font,
                )
                self.view.addItem(t)

            # Intensity tick labels along the Z axis (0~1)
            for v in (0.0, 0.25, 0.5, 0.75, 1.0):
                z = v * Z_SPAN
                t = gl.GLTextItem(
                    pos=np.array([-2.6, 0.0, z], dtype=np.float32),
                    text=f"{v:.2f}",
                    color=QColor(180, 205, 255),
                    font=tick_font,
                )
                self.view.addItem(t)
        except Exception:
            # GLTextItem signature differs across pyqtgraph versions; labels are
            # cosmetic, so silently skip if unsupported.
            pass

    # ── Per-frame refresh ──────────────────────────────────────
    def update_frame(self, envelopes) -> None:
        """Update the point cloud from a [12, 256] envelope matrix."""
        try:
            env = np.asarray(envelopes, dtype=np.float32)
        except Exception:
            return
        if env.ndim != 2 or env.shape[0] != N_SENSORS or env.shape[1] != ENVELOPE_LEN:
            return

        amp = np.clip(env, 0.0, 1.0).reshape(-1)
        pos = self._base_pos.copy()
        pos[:, 2] = amp * Z_SPAN

        colors = self._cmap.map(amp, mode="float")          # (N, 4) float 0~1
        # Keep the whole 12×256 grid visible (so density matches W2), while still
        # letting strong echoes stand out via higher opacity.
        colors[:, 3] = np.clip(0.40 + amp * 0.8, 0.0, 1.0)
        # Fixed marker size (no amplitude scaling) so tall peaks don't bloat into
        # large spheres that occlude the rest of the cloud.
        self.scatter.setData(pos=pos, color=colors, size=4.0)

        # Update per-probe waveform polylines
        for ch, line in enumerate(self._lines):
            seg_pos = pos[ch * ENVELOPE_LEN:(ch + 1) * ENVELOPE_LEN]
            seg_col = colors[ch * ENVELOPE_LEN:(ch + 1) * ENVELOPE_LEN].copy()
            seg_col[:, 3] = 0.55
            line.setData(pos=seg_pos, color=seg_col)

    def reset(self) -> None:
        """Clear the point cloud back to a flat baseline."""
        pos = self._base_pos.copy()
        self.scatter.setData(
            pos=pos,
            color=(0.5, 0.5, 0.5, 0.4),
            size=3.0,
        )
        for ch, line in enumerate(self._lines):
            seg = pos[ch * ENVELOPE_LEN:(ch + 1) * ENVELOPE_LEN]
            line.setData(pos=seg, color=(0.4, 0.4, 0.4, 0.4))
