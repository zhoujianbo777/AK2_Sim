"""
window_ogm.py  —  W4 Occupancy Grid Map (OGM)
Top-down view of vehicle and obstacle spatial distribution.
See spec section 5.2.4.
"""

import math
from collections import deque
import numpy as np
from scipy.ndimage import gaussian_filter
import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QGraphicsPolygonItem
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QPolygonF, QFont

from modules.config_loader import ConfigLoader
from modules.result_fusion import AlgoResult
from modules.data_manager import DataFrame
from modules.obstacle_classes import CLASS_COLORS_RGB


class WindowOGM(QWidget):
    """W4 Occupancy Grid Map."""

    def __init__(self, cfg: ConfigLoader, mode: str = "ai", parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._mode = mode
        # Use unified 9-class colors for all modes (traditional maps to subset of IDs 0–8)
        self._class_colors = CLASS_COLORS_RGB

        self._range_front = cfg.get("display.ogm_range_front_m", 6.0)
        self._range_side  = cfg.get("display.ogm_range_side_m", 5.0)
        self._resolution  = cfg.get("display.ogm_resolution_m", 0.1)
        self._veh_len     = cfg.get("display.vehicle_length_m", 4.5)
        self._veh_wid     = cfg.get("display.vehicle_width_m", 1.9)
        self._show_fov    = True
        self._m1_threshold = cfg.get("ai.m1_validity_threshold", 0.5)

        # Sensor mount parameters
        self._sensor_configs = [
            cfg.get_sensor_config(f"S{i:02d}") for i in range(1, 13)
        ]

        self._trajectory: list[tuple[float, float]] = []  # historical trajectory points

        # Multi-frame accumulation (odometry-registered) state. Each frame's obstacle
        # segments are stored in a fixed world frame; the recent history is re-projected
        # into the current ego frame on every update so a stable boundary builds up as
        # the vehicle moves (low-speed APA odometry via a single-track / 单轨 model).
        self._accum_on = True
        self._accum_maxframes = cfg.get("display.ogm_accum_frames", 30)
        self._wheelbase = cfg.get("display.wheelbase_m", 2.7)
        self._steer_ratio = cfg.get("display.steering_ratio", 15.0)
        self._ego_x = 0.0
        self._ego_y = 0.0
        self._ego_th = 0.0
        self._last_ts: float | None = None
        self._accum_buf: deque = deque(maxlen=self._accum_maxframes)

        self.setWindowTitle("W4 - Occupancy Grid Map (OGM)")
        self.resize(620, 640)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._chk_fov = QCheckBox("Show Sensor FOV Sectors")
        self._chk_fov.setChecked(True)
        self._chk_fov.toggled.connect(self._on_fov_toggled)
        toolbar.addWidget(self._chk_fov)
        self._chk_accum = QCheckBox("Accumulate (multi-frame)")
        self._chk_accum.setChecked(True)
        self._chk_accum.toggled.connect(self._on_accum_toggled)
        toolbar.addWidget(self._chk_accum)
        toolbar.addStretch()
        lay.addLayout(toolbar)

        # pyqtgraph ViewBox
        self._pw = pg.PlotWidget()
        self._pw.setBackground("#1A1A2E")
        self._pw.setAspectLocked(True)
        self._pw.setXRange(-self._range_side, self._range_side)
        self._pw.setYRange(-self._range_front, self._range_front)
        self._pw.setLabel("left", "Longitudinal (m, fwd+)")
        self._pw.setLabel("bottom", "Lateral (m, right+)")
        self._pw.showGrid(x=True, y=True, alpha=0.15)
        lay.addWidget(self._pw)

        # Vehicle outline (blue rectangle)
        hw = self._veh_wid / 2
        hl = self._veh_len / 2
        veh_pts = np.array([
            [-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl], [-hw, -hl]
        ])
        self._veh_item = self._pw.plot(
            veh_pts[:, 0], veh_pts[:, 1],
            pen=pg.mkPen("#2196F3", width=2)
        )

        # M6 free-space heatmap (drivable space) — rendered furthest back (z=-11),
        # below the occupancy heatmap. Green LUT: low prob → transparent,
        # high prob → semi-transparent green (info M2 散点 cannot represent).
        _lut_free = np.zeros((256, 4), dtype=np.uint8)
        _lut_free[:, 1] = 200                                          # G
        _lut_free[:, 2] = 90                                           # slight cyan tint
        _lut_free[:, 3] = np.linspace(0, 110, 256).astype(np.uint8)   # alpha ramp
        self._ogm_free_heatmap = pg.ImageItem()
        self._ogm_free_heatmap.setLookupTable(_lut_free)
        self._ogm_free_heatmap.setZValue(-11)
        self._pw.addItem(self._ogm_free_heatmap)
        self._ogm_free_heatmap.setRect(QRectF(
            -self._range_side, -self._range_front,
            2 * self._range_side, 2 * self._range_front,
        ))

        # M6 OGM heatmap (rendered behind all other items, z=-10)
        # Red-to-yellow LUT: low prob → transparent red, high prob → opaque yellow
        _lut = np.zeros((256, 4), dtype=np.uint8)
        _lut[:, 0] = 255                                              # R always max
        _lut[:, 1] = np.linspace(0, 255, 256).astype(np.uint8)       # G ramp
        _lut[:, 3] = np.linspace(0, 180, 256).astype(np.uint8)       # alpha ramp
        self._ogm_heatmap = pg.ImageItem()
        self._ogm_heatmap.setLookupTable(_lut)
        self._ogm_heatmap.setZValue(-10)
        self._pw.addItem(self._ogm_heatmap)
        # Pre-set the bounding rect once (same for every frame)
        self._ogm_heatmap.setRect(QRectF(
            -self._range_side, -self._range_front,
            2 * self._range_side, 2 * self._range_front,
        ))

        # FOV sector polygons (12 channels) — geometry built once, brush updated per frame
        self._fov_poly_items: list[QGraphicsPolygonItem] = []
        for _ in self._sensor_configs:
            item = QGraphicsPolygonItem()
            item.setPen(pg.mkPen(QColor(255, 255, 255, 30), width=1))
            item.setBrush(pg.mkBrush(QColor(0, 0, 0, 0)))
            self._pw.addItem(item)
            self._fov_poly_items.append(item)
        self._build_fov_geometry()

        # Accumulated (multi-frame, odometry-registered) obstacle history — faint
        # segments per class, re-projected into the current ego frame each update so the
        # boundary builds up into a stable contour as the vehicle moves. Drawn beneath
        # the bright current-frame segments (z below default 0, above the heatmap).
        self._accum_curves: list[pg.PlotCurveItem] = []
        for _ in range(len(self._class_colors)):
            cv = pg.PlotCurveItem()
            cv.setZValue(-5)
            self._pw.addItem(cv)
            self._accum_curves.append(cv)

        # Obstacle segments — each detected obstacle is rendered as a straight line
        # segment placed at its measured distance, oriented perpendicular to the sensor
        # line-of-sight, with length = object_width_m. One curve item per channel so each
        # segment carries its own class colour.
        self._obstacle_segs: list[pg.PlotCurveItem] = []
        for _ in range(12):
            seg = pg.PlotCurveItem()
            self._pw.addItem(seg)
            self._obstacle_segs.append(seg)

        # SA-correction ring scatter: white hollow rings drawn over SA-corrected channels
        self._sa_ring_scatter = pg.ScatterPlotItem(size=22, pxMode=True)
        self._pw.addItem(self._sa_ring_scatter)

        # Trajectory curve
        self._traj_curve = self._pw.plot(
            pen=pg.mkPen(QColor(150, 150, 150, 120), width=1, style=Qt.PenStyle.DashLine)
        )

        # Advantage banner label
        self._banner = pg.TextItem("", color="#FF9800", anchor=(0.5, 0))
        self._banner.setPos(0, self._range_front - 0.3)
        self._pw.addItem(self._banner)
        self._banner_countdown = 0

        # Suspension height labels — one slot per channel, visible only for Overhead (class 6)
        self._height_labels: list[pg.TextItem] = []
        _ht_font = QFont("Arial", 8)
        for _ in range(12):
            t = pg.TextItem(text="", color=(255, 230, 80), anchor=(0.5, 1.0))
            t.setFont(_ht_font)
            self._pw.addItem(t)
            self._height_labels.append(t)

        # Object width labels — one slot per channel, visible when object_width_m is available
        self._width_labels: list[pg.TextItem] = []
        for _ in range(12):
            t = pg.TextItem(text="", color=(120, 220, 255), anchor=(0.5, 0.0))
            t.setFont(_ht_font)
            self._pw.addItem(t)
            self._width_labels.append(t)

    def _build_fov_geometry(self):
        """Compute static FOV sector polygon vertices from sensor configs (called once)."""
        max_dist = 5.0
        for i, sc in enumerate(self._sensor_configs):
            if not sc:
                self._fov_poly_items[i].setPolygon(QPolygonF())
                continue
            sx = sc.get("x_m", 0.0)
            sy = sc.get("y_m", 0.0)
            yaw = math.radians(sc.get("yaw_deg", 0.0))
            fov = math.radians(sc.get("fov_deg", 75.0))

            # Generate sector vertices — negate lateral to match right+ screen x
            angles = np.linspace(yaw - fov / 2, yaw + fov / 2, 20)
            pts = [QPointF(-sy, sx)]
            pts += [QPointF(-(sy + max_dist * math.sin(a)), sx + max_dist * math.cos(a))
                    for a in angles]
            poly = QPolygonF()
            for pt in pts:
                poly.append(pt)
            self._fov_poly_items[i].setPolygon(poly)

    def _on_fov_toggled(self, show: bool) -> None:
        """Handle Show FOV checkbox toggle."""
        self._show_fov = show
        for item in self._fov_poly_items:
            item.setVisible(show)

    def _on_accum_toggled(self, on: bool) -> None:
        """Handle multi-frame accumulation checkbox toggle."""
        self._accum_on = on
        if not on:
            self._accum_buf.clear()
            for cv in self._accum_curves:
                cv.setData([], [])

    def _integrate_ego(self, frame: DataFrame) -> None:
        """Advance ego pose (world frame) from CAN speed/steering via a single-track
        (单轨 / kinematic bicycle) model."""
        ts = float(getattr(frame, "timestamp_ms", 0.0))
        if self._last_ts is None:
            self._last_ts = ts
            return
        dt = (ts - self._last_ts) / 1000.0
        self._last_ts = ts
        # Skip first frame / session reset / out-of-order timestamps (no motion update)
        if dt <= 0.0 or dt > 0.5:
            return
        v = float(getattr(frame, "vehicle_speed", 0.0)) / 3.6   # km/h → m/s
        if str(getattr(frame, "gear", "D")).upper() == "R":
            v = -v
        delta = math.radians(float(getattr(frame, "steering_angle", 0.0)) / self._steer_ratio)
        th = self._ego_th
        self._ego_x += v * math.cos(th) * dt
        self._ego_y += v * math.sin(th) * dt
        self._ego_th += (v / self._wheelbase) * math.tan(delta) * dt

    def _veh_to_world(self, xv: float, yv: float) -> tuple[float, float]:
        """Vehicle frame (x fwd, y left) → world frame at current ego pose."""
        c, s = math.cos(self._ego_th), math.sin(self._ego_th)
        return (self._ego_x + xv * c - yv * s, self._ego_y + xv * s + yv * c)

    def _world_to_veh(self, wx: float, wy: float) -> tuple[float, float]:
        """World frame → vehicle frame at current ego pose."""
        c, s = math.cos(self._ego_th), math.sin(self._ego_th)
        dx, dy = wx - self._ego_x, wy - self._ego_y
        return (dx * c + dy * s, -dx * s + dy * c)

    def _render_accum(self) -> None:
        """Re-project accumulated world-frame segments into the current ego frame."""
        if not self._accum_on:
            for cv in self._accum_curves:
                cv.setData([], [])
            return
        xs: list[list[float]] = [[] for _ in self._accum_curves]
        ys: list[list[float]] = [[] for _ in self._accum_curves]
        for group in self._accum_buf:
            for (wx0, wy0, wx1, wy1, cid) in group:
                if cid >= len(self._accum_curves):
                    continue
                xv0, yv0 = self._world_to_veh(wx0, wy0)
                xv1, yv1 = self._world_to_veh(wx1, wy1)
                # vehicle frame → screen (x_screen = -y_v, y_screen = x_v); NaN breaks segments
                xs[cid] += [-yv0, -yv1, math.nan]
                ys[cid] += [xv0, xv1, math.nan]
        for cid, cv in enumerate(self._accum_curves):
            if xs[cid]:
                r, g, b = (self._class_colors[cid]
                           if cid < len(self._class_colors) else (180, 180, 180))
                cv.setData(xs[cid], ys[cid], connect="finite",
                           pen=pg.mkPen(r, g, b, 70, width=3))
            else:
                cv.setData([], [])

    def _update_fov_colors(self, valid_flags: np.ndarray) -> None:
        """Update each FOV sector fill: orange if obstacle detected, transparent otherwise."""
        for i, item in enumerate(self._fov_poly_items):
            if not self._show_fov:
                item.setVisible(False)
                continue
            item.setVisible(True)
            if valid_flags[i] >= self._m1_threshold:
                item.setBrush(pg.mkBrush(QColor(255, 140, 0, 50)))   # orange — obstacle
                item.setPen(pg.mkPen(QColor(255, 140, 0, 160), width=1))
            else:
                item.setBrush(pg.mkBrush(QColor(0, 0, 0, 0)))        # transparent
                item.setPen(pg.mkPen(QColor(255, 255, 255, 30), width=1))

    def on_result_updated(self, result: AlgoResult) -> None:
        """Observer callback (used together with update_ogm)."""
        pass

    def reset(self) -> None:
        """Clear OGM scatter and trajectory when switching sessions."""
        for seg in self._obstacle_segs:
            seg.setData([], [])
        for cv in self._accum_curves:
            cv.setData([], [])
        self._accum_buf.clear()
        self._ego_x = self._ego_y = self._ego_th = 0.0
        self._last_ts = None
        self._sa_ring_scatter.setData([])
        self._trajectory.clear()
        self._traj_curve.setData([], [])
        self._banner.setText("")
        self._banner_countdown = 0
        for t in self._height_labels:
            t.setText("")
        for t in self._width_labels:
            t.setText("")
        self._ogm_heatmap.clear()
        self._ogm_free_heatmap.clear()

    def update_ogm(
        self,
        result: AlgoResult,
        frame: DataFrame,
        advantage_text: str = ""
    ) -> None:
        """Refresh OGM display."""
        # Clear all per-channel height labels before re-populating
        for t in self._height_labels:
            t.setText("")
        for t in self._width_labels:
            t.setText("")
        # Advance ego pose for odometry-registered accumulation, then collect this
        # frame's segments (in world coords) for the rolling history buffer.
        self._integrate_ego(frame)
        cur_group: list[tuple[float, float, float, float, int]] = []
        # Per-channel obstacle line segments at the measured distance.
        active_chs: list[int] = []
        obstacle_pts: dict[int, tuple[float, float]] = {}
        for ch in range(12):
            if result.valid_flags[ch] < self._m1_threshold:
                continue
            sc = self._sensor_configs[ch]
            if not sc:
                continue
            cid = int(result.class_ids[ch])
            # Skip "Open/Clear" class (ID 4) — no obstacle present, nothing to plot
            if cid == 4:
                continue
            dist = float(frame.edi_distance[ch])
            # If EDI has no distance but AI says obstacle, fall back to sensor max range
            # and render with low opacity to signal uncertain position
            uncertain = dist <= 0
            if uncertain:
                dist = sc.get("max_range_m", 5.5) * 0.5
            yaw = math.radians(sc.get("yaw_deg", 0.0))
            sx = sc.get("x_m", 0.0)
            sy = sc.get("y_m", 0.0)
            # Sensor mount position (arc centre) — OGM x-axis: right+, y-axis: fwd+.
            # config uses left+ for y_m, so negate to get right+ for screen x.
            cx = -sy
            cy = sx
            # Obstacle midpoint on the arc (used for labels / SA rings)
            obs_x = -(sy + dist * math.sin(yaw))
            obs_y = sx + dist * math.cos(yaw)
            r, g, b = self._class_colors[cid] if cid < len(self._class_colors) else (180, 180, 180)
            # Segment length = object width; orientation perpendicular to the sensor
            # line-of-sight, centred on the obstacle midpoint.
            w_m = float(result.object_width_m[ch])
            if w_m > 0.0:
                width_m = float(np.clip(w_m, 0.15, 2.5))
            else:
                width_m = 0.20 if uncertain else 0.40
            bearing = math.atan2(obs_y - cy, obs_x - cx)
            perp_x = -math.sin(bearing)
            perp_y = math.cos(bearing)
            half = width_m / 2.0
            # Clamp the drawn half-length to the sensor's FOV footprint at this range,
            # so a single-sensor segment never extends beyond its beam (object_width_m is
            # the object's full physical width, which can exceed one beam's coverage).
            fov = math.radians(sc.get("fov_deg", 75.0))
            fov_half = dist * math.tan(fov / 2.0)
            half = min(half, fov_half)
            seg_x = [obs_x - half * perp_x, obs_x + half * perp_x]
            seg_y = [obs_y - half * perp_y, obs_y + half * perp_y]
            pen = (pg.mkPen(r, g, b, 110, width=3) if uncertain
                   else pg.mkPen(r, g, b, 235, width=6))
            self._obstacle_segs[ch].setData(seg_x, seg_y, pen=pen)
            active_chs.append(ch)
            # Record this segment's endpoints in the world frame for accumulation.
            wx0, wy0 = self._veh_to_world(seg_y[0], -seg_x[0])
            wx1, wy1 = self._veh_to_world(seg_y[1], -seg_x[1])
            cur_group.append((wx0, wy0, wx1, wy1, cid))
            obstacle_pts[ch] = (obs_x, obs_y)
            # Head-3: suspension height label for Overhead obstacles (class 6)
            if cid == 6:
                s_m = float(result.suspension_height_m[ch])
                if s_m >= 0.0:
                    self._height_labels[ch].setText(f"\u21d5{s_m:.2f}m")
                    self._height_labels[ch].setPos(obs_x, obs_y + 0.35)
            # Object width annotation (shown once object_width_m is filled by an engine)
            if w_m > 0.0:
                self._width_labels[ch].setText(f"\u2190{w_m:.2f}m\u2192")
                self._width_labels[ch].setPos(obs_x, obs_y - 0.35)
        # Clear segments for channels with no obstacle this frame
        for ch in range(12):
            if ch not in obstacle_pts:
                self._obstacle_segs[ch].setData([], [])

        # Push this frame into the rolling history and re-render the accumulated boundary.
        self._accum_buf.append(cur_group)
        self._render_accum()

        # SA-correction rings: white hollow circle over channels where SA changed SC's class.
        # Only meaningful for AI engine results; Traditional engine leaves sc_class_ids at
        # default zeros, so skip rings entirely for non-AI results to avoid false indicators.
        ring_spots = []
        if result.engine_type == "AI":
            sc_ids = getattr(result, "sc_class_ids", None)
            for ch in active_chs:
                if sc_ids is not None and int(result.class_ids[ch]) != int(sc_ids[ch]):
                    ox, oy = obstacle_pts[ch]
                    ring_spots.append({
                        "pos":   (ox, oy),
                        "brush": pg.mkBrush(0, 0, 0, 0),           # transparent fill
                        "pen":   pg.mkPen("w", width=2),            # white ring
                        "size":  22,
                    })
        self._sa_ring_scatter.setData(ring_spots)

        # Trajectory (simplified: fixed at origin; real project should accumulate odometry)
        self._trajectory.append((0.0, 0.0))
        if len(self._trajectory) > 30:
            self._trajectory.pop(0)
        if len(self._trajectory) > 1:
            txs = [p[0] for p in self._trajectory]
            tys = [p[1] for p in self._trajectory]
            self._traj_curve.setData(txs, tys)

        # Update FOV sector fill colors based on M1 validity
        self._update_fov_colors(result.valid_flags)

        # Advantage banner
        if advantage_text:
            self._banner.setText(advantage_text)
            self._banner_countdown = 3
        elif self._banner_countdown > 0:
            self._banner_countdown -= 1
            if self._banner_countdown == 0:
                self._banner.setText("")

        # M6 OGM heatmap overlay
        ogm = getattr(result, "ogm_grid", None)
        if ogm is not None:
            # Mask out ego-vehicle footprint: the CNN has no knowledge of the vehicle body,
            # so it may produce spurious occupancy inside the chassis area.
            ogm = ogm.copy()
            _res = self._resolution
            _rf  = self._range_front
            _rs  = self._range_side
            _hl  = self._veh_len / 2
            _hw  = self._veh_wid / 2
            r0 = int((_rf - _hl) / _res)
            r1 = int((_rf + _hl) / _res) + 1
            c0 = int((-_hw + _rs) / _res)
            c1 = int(( _hw + _rs) / _res) + 1
            ogm[r0:r1, c0:c1] = 0.0

        # Suppress OGM in zones where the corresponding sensor group has no valid echo.
        # ch0-5 = front row, ch6-11 = rear row.
        # Gate maps mean_valid ≤ 0.05 → 0.0 (fully suppressed),
        #                  mean_valid ≥ 0.30 → 1.0 (fully shown), linear in between.
        if ogm is not None and result.valid_flags is not None:
            H = ogm.shape[0]
            half = H // 2  # row 0 = front; row half = longitudinal midpoint
            front_gate = float(np.clip((np.mean(result.valid_flags[0:6])  - 0.05) / 0.25, 0.0, 1.0))
            rear_gate  = float(np.clip((np.mean(result.valid_flags[6:12]) - 0.05) / 0.25, 0.0, 1.0))
            ogm[:half, :] *= front_gate
            ogm[half:, :]  *= rear_gate
        # Global noise floor removal: zero out residual low-amplitude model artifacts
        if ogm is not None:
            ogm[ogm < 0.18] = 0.0
        if ogm is not None and float(ogm.max()) > 0.01:
            # Smooth to eliminate block artifacts from upscaled grid cells
            ogm_smooth = gaussian_filter(ogm, sigma=0.6)
            # ogm: [H, W] row=0=front(+y), col=0=left(-x)
            # ImageItem axis convention: first axis → x, second → y, y=0 at bottom.
            # Flip rows so row0(-range_front) is at y-min, then transpose.
            disp = (ogm_smooth[::-1, :].T * 255).clip(0, 255).astype(np.uint8)  # [W, H]
            self._ogm_heatmap.setImage(disp, autoLevels=False, levels=(0, 255))
            # setImage() resets the ImageItem transform; re-apply bounding rect each frame
            self._ogm_heatmap.setRect(QRectF(
                -self._range_side, -self._range_front,
                2 * self._range_side, 2 * self._range_front,
            ))
        else:
            self._ogm_heatmap.clear()

        # M6 free-space (drivable) heatmap overlay — channel 1 of the 2-ch model.
        # Green wedges swept from each sensor to the nearest obstacle: drivable space
        # that a per-channel point classifier (M2) fundamentally cannot output.
        free = getattr(result, "ogm_free", None)
        if free is not None and float(free.max()) > 0.01:
            free = free.copy()
            _res = self._resolution
            _rf  = self._range_front
            _rs  = self._range_side
            _hl  = self._veh_len / 2
            _hw  = self._veh_wid / 2
            r0 = int((_rf - _hl) / _res)
            r1 = int((_rf + _hl) / _res) + 1
            c0 = int((-_hw + _rs) / _res)
            c1 = int(( _hw + _rs) / _res) + 1
            free[r0:r1, c0:c1] = 0.0      # blank ego footprint
            free[free < 0.40] = 0.0        # noise floor (free is dense; keep confident only)
            if float(free.max()) > 0.01:
                free_smooth = gaussian_filter(free, sigma=0.8)
                disp_f = (free_smooth[::-1, :].T * 255).clip(0, 255).astype(np.uint8)
                self._ogm_free_heatmap.setImage(disp_f, autoLevels=False, levels=(0, 255))
                self._ogm_free_heatmap.setRect(QRectF(
                    -self._range_side, -self._range_front,
                    2 * self._range_side, 2 * self._range_front,
                ))
            else:
                self._ogm_free_heatmap.clear()
        else:
            self._ogm_free_heatmap.clear()
