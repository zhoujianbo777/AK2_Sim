"""
window_control.py  —  W1 Main Control Panel
See spec section 5.2.1.

Features:
  - Session management (scan/load TestData directory)
  - Playback control bar (play/pause, step, seek, speed)
  - Current frame metadata display
  - Noise injection sub-panel (robustness test)
  - Batch evaluate button
  - Launch compare mode button
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QTreeWidget, QTreeWidgetItem, QGroupBox,
    QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QSizePolicy,
    QFileDialog, QMessageBox, QLineEdit, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QColor

import logging

from modules.config_loader import ConfigLoader
from modules.obstacle_classes import OBSTACLE_TABLE
from modules.data_manager import DataManager, SessionMeta
from modules.result_fusion import AlgoResult


# ─────────────────────────────────────────────────────────────
# Thread-safe GUI log handler
# ─────────────────────────────────────────────────────────────

class _GuiLogHandler(QObject, logging.Handler):
    """logging.Handler that safely appends records to a QTextEdit from any thread."""

    _sig = pyqtSignal(str)   # carries pre-formatted HTML

    # Level → foreground color
    _COLORS = {
        logging.DEBUG:    "#888888",
        logging.INFO:     "#cccccc",
        logging.WARNING:  "#ffd740",
        logging.ERROR:    "#ff5252",
        logging.CRITICAL: "#ff1744",
    }

    def __init__(self, text_widget: QTextEdit):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self._widget = text_widget
        # Queued connection ensures delivery on the GUI thread regardless of caller
        self._sig.connect(self._append, Qt.ConnectionType.QueuedConnection)
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname).1s] %(name)s: %(message)s",
                                datefmt="%H:%M:%S")
        self.setFormatter(fmt)

    def emit(self, record: logging.LogRecord) -> None:  # called from any thread
        try:
            msg = self.format(record)
            color = self._COLORS.get(record.levelno, "#cccccc")
            # Escape HTML special chars
            msg = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html = f'<span style="color:{color};">{msg}</span>'
            self._sig.emit(html)
        except Exception:
            self.handleError(record)

    def _append(self, html: str) -> None:              # always on GUI thread
        self._widget.insertHtml(html + "<br>")
        sb = self._widget.verticalScrollBar()
        sb.setValue(sb.maximum())


class WindowControl(QWidget):
    """W1 Main Control Panel."""

    # Signals: notify main application of user actions
    sig_session_loaded = pyqtSignal(str)          # session loaded
    sig_play = pyqtSignal()
    sig_pause = pyqtSignal()
    sig_seek = pyqtSignal(int)                    # seek to frame
    sig_step_forward = pyqtSignal()
    sig_step_backward = pyqtSignal()
    sig_speed_changed = pyqtSignal(float)
    sig_loop_changed = pyqtSignal(bool)
    sig_launch_compare = pyqtSignal()             # launch compare mode (second process)
    sig_batch_evaluate = pyqtSignal(list)         # batch evaluate: list of session IDs

    def __init__(self, cfg: ConfigLoader, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.dm = data_manager
        self._total_frames = 0
        self._engine_type = "unknown"

        self.setWindowTitle("W1 - Main Control Panel")
        self.setMinimumWidth(400)
        self.setMinimumHeight(700)

        self._build_ui()
        self._refresh_session_list()

        # Attach GUI log handler to root logger (captures all modules)
        self._log_handler = _GuiLogHandler(self._log_view)
        self._log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._log_handler)

    def closeEvent(self, event):
        """Closing the control panel exits the entire application."""
        QApplication.quit()
        event.accept()

    # ─────────────────────────────────────────
    # UI build
    # ─────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Title ──
        title = QLabel("AK2 Ultrasonic Perception Simulator")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(); font.setPointSize(12); font.setBold(True)
        title.setFont(font)
        root.addWidget(title)

        # ── Session manager ──
        root.addWidget(self._build_session_group())

        # ── Playback control ──
        root.addWidget(self._build_playback_group())

        # ── Current frame metadata ──
        root.addWidget(self._build_meta_group())

        # ── Robustness test ──
        root.addWidget(self._build_noise_group())

        # ── Bottom buttons ──
        root.addWidget(self._build_bottom_buttons())

        # ── Obstacle class color legend ──
        root.addWidget(self._build_legend_group())

        # ── Runtime log view ──
        root.addWidget(self._build_log_group())

    def _build_session_group(self) -> QGroupBox:
        grp = QGroupBox("Dataset Manager")
        lay = QVBoxLayout(grp)

        # Session tree
        self._session_tree = QTreeWidget()
        self._session_tree.setHeaderLabels(["Session ID", "Date", "Scene", "Frames"])
        self._session_tree.setMaximumHeight(200)
        self._session_tree.itemDoubleClicked.connect(self._on_session_double_click)
        lay.addWidget(self._session_tree)

        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh List")
        btn_refresh.clicked.connect(self._refresh_session_list)
        btn_load = QPushButton("Load Selected Session")
        btn_load.clicked.connect(self._on_load_session)
        btn_row.addWidget(btn_refresh)
        btn_row.addWidget(btn_load)
        lay.addLayout(btn_row)

        return grp

    def _build_playback_group(self) -> QGroupBox:
        grp = QGroupBox("Playback Control")
        lay = QVBoxLayout(grp)

        # Progress bar
        prog_row = QHBoxLayout()
        self._lbl_frame = QLabel("Frame: 0 / 0")
        self._progress = QSlider(Qt.Orientation.Horizontal)
        self._progress.setMinimum(0)
        self._progress.setMaximum(0)
        self._progress.valueChanged.connect(self._on_progress_changed)
        self._lbl_time = QLabel("00:00:00.000")
        prog_row.addWidget(self._lbl_frame)
        prog_row.addWidget(self._progress, stretch=1)
        prog_row.addWidget(self._lbl_time)
        lay.addLayout(prog_row)

        # Control button row
        btn_row = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Prev Frame")
        self._btn_prev.clicked.connect(self.sig_step_backward)
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self._on_play_pause)
        self._btn_next = QPushButton("Next Frame ▶")
        self._btn_next.clicked.connect(self.sig_step_forward)
        btn_row.addWidget(self._btn_prev)
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(self._btn_next)
        lay.addLayout(btn_row)

        # Speed + loop
        opt_row = QHBoxLayout()
        opt_row.addWidget(QLabel("Speed:"))
        self._cbox_speed = QComboBox()
        for spd in ["0.25×", "0.5×", "1×", "2×", "5×"]:
            self._cbox_speed.addItem(spd)
        self._cbox_speed.setCurrentIndex(2)  # default 1×
        self._cbox_speed.currentIndexChanged.connect(self._on_speed_changed)
        opt_row.addWidget(self._cbox_speed)
        self._chk_loop = QCheckBox("Loop Playback")
        self._chk_loop.toggled.connect(self.sig_loop_changed)
        opt_row.addWidget(self._chk_loop)
        opt_row.addStretch()
        lay.addLayout(opt_row)

        # Jump/seek
        jump_row = QHBoxLayout()
        jump_row.addWidget(QLabel("Jump to Frame:"))
        self._spin_jump = QSpinBox()
        self._spin_jump.setMinimum(0)
        self._spin_jump.setMaximum(0)
        btn_jump = QPushButton("Go")
        btn_jump.clicked.connect(lambda: self.sig_seek.emit(self._spin_jump.value()))
        jump_row.addWidget(self._spin_jump)
        jump_row.addWidget(btn_jump)
        jump_row.addStretch()
        lay.addLayout(jump_row)

        return grp

    def _build_meta_group(self) -> QGroupBox:
        grp = QGroupBox("Current Frame Info")
        lay = QVBoxLayout(grp)
        self._meta_labels = {}
        fields = [
            ("frame_id", "Frame ID"),
            ("timestamp", "Timestamp"),
            ("speed", "Speed (km/h)"),
            ("steering", "Steering Angle (°)"),
            ("gear", "Gear"),
        ]
        for key, name in fields:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            lbl = QLabel("-")
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._meta_labels[key] = lbl
            row.addWidget(lbl)
            lay.addLayout(row)
        return grp

    def _build_noise_group(self) -> QGroupBox:
        grp = QGroupBox("Robustness Test (Noise Injection)")
        grp.setCheckable(True)
        grp.setChecked(False)
        grp.toggled.connect(self._on_noise_toggle)
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Gaussian Noise SNR (dB):"))
        self._spin_snr = QDoubleSpinBox()
        self._spin_snr.setRange(20.0, 50.0)
        self._spin_snr.setValue(40.0)
        self._spin_snr.valueChanged.connect(self._on_noise_param_changed)
        row1.addWidget(self._spin_snr)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Frame Drop Probability:"))
        self._spin_dropout = QDoubleSpinBox()
        self._spin_dropout.setRange(0.0, 0.5)
        self._spin_dropout.setSingleStep(0.05)
        self._spin_dropout.valueChanged.connect(self._on_noise_param_changed)
        row2.addWidget(self._spin_dropout)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Timing Shift (samples):"))
        self._spin_shift = QSpinBox()
        self._spin_shift.setRange(-10, 10)
        self._spin_shift.valueChanged.connect(self._on_noise_param_changed)
        row3.addWidget(self._spin_shift)
        lay.addLayout(row3)

        self._noise_group = grp
        return grp

    def _build_legend_group(self) -> QGroupBox:
        """Obstacle class color legend — generated from obstacle_classes.OBSTACLE_TABLE."""
        LEGEND = [
            (row[3], str(row[0]), row[1])
            for row in OBSTACLE_TABLE
        ]

        grp = QGroupBox("Obstacle Class Legend (9-class unified)")
        outer = QVBoxLayout(grp)
        outer.setSpacing(3)
        outer.setContentsMargins(6, 6, 6, 6)

        # Header row
        hdr = QHBoxLayout()
        for text, stretch in [("Color", 1), ("ID", 0), ("Class Name", 3)]:
            lbl = QLabel(f"<b>{text}</b>")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hdr.addWidget(lbl, stretch=stretch)
        outer.addLayout(hdr)

        # One row per class
        for color, cid, name in LEGEND:
            row = QHBoxLayout()
            row.setSpacing(4)

            # Color swatch
            swatch = QLabel()
            swatch.setFixedSize(32, 18)
            swatch.setStyleSheet(
                f"background-color: {color}; border: 1px solid #888; border-radius: 3px;"
            )

            # Class ID
            lbl_id = QLabel(cid)
            lbl_id.setFixedWidth(20)
            lbl_id.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Class name
            lbl_name = QLabel(name)
            lbl_name.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

            row.addWidget(swatch, stretch=0)
            row.addSpacing(4)
            row.addWidget(lbl_id, stretch=0)
            row.addSpacing(4)
            row.addWidget(lbl_name, stretch=1)
            outer.addLayout(row)

        return grp

    def _build_log_group(self) -> QGroupBox:
        """Build the runtime log view at the bottom of the control panel."""
        grp = QGroupBox("Runtime Log")
        lay = QVBoxLayout(grp)
        lay.setContentsMargins(4, 4, 4, 4)

        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        font = QFont("Courier New", 8)
        self._log_view.setFont(font)
        self._log_view.setStyleSheet(
            "QTextEdit { background:#1e1e1e; color:#cccccc; border:none; }"
        )
        # Keep at most 500 lines to avoid memory growth during long sessions
        self._log_view.document().setMaximumBlockCount(500)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedHeight(20)
        btn_clear.setFixedWidth(60)
        btn_clear.clicked.connect(self._log_view.clear)

        top = QHBoxLayout()
        top.addStretch()
        top.addWidget(btn_clear)
        lay.addLayout(top)
        lay.addWidget(self._log_view)
        return grp

    def _build_bottom_buttons(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        btn_compare = QPushButton("⚖ Compare Mode")
        btn_compare.setToolTip("Launch a second instance with the opposite algorithm for side-by-side comparison")
        btn_compare.clicked.connect(self.sig_launch_compare.emit)
        lay.addWidget(btn_compare)

        btn_batch = QPushButton("📊 Batch Evaluate")
        btn_batch.clicked.connect(self._on_batch_evaluate)
        lay.addWidget(btn_batch)
        return w

    # ─────────────────────────────────────────
    # Data binding & updates
    # ─────────────────────────────────────────

    def set_engine_type(self, engine_type: str) -> None:
        self._engine_type = engine_type

    def _refresh_session_list(self) -> None:
        self._session_tree.clear()
        for meta in self.dm.get_session_list():
            item = QTreeWidgetItem([
                meta.session_id,
                meta.date,
                meta.road_type,
                str(meta.total_frames),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, meta.session_id)
            if meta.has_ground_truth:
                item.setText(3, f"{meta.total_frames} ✓GT")
            self._session_tree.addTopLevelItem(item)
        self._session_tree.resizeColumnToContents(0)

    def _on_session_double_click(self, item: QTreeWidgetItem, col: int) -> None:
        session_id = item.data(0, Qt.ItemDataRole.UserRole)
        if session_id:
            self._load_session(session_id)

    def _on_load_session(self) -> None:
        items = self._session_tree.selectedItems()
        if not items:
            return
        session_id = items[0].data(0, Qt.ItemDataRole.UserRole)
        if session_id:
            self._load_session(session_id)

    def _load_session(self, session_id: str) -> None:
        ok = self.dm.load_session(session_id)
        if not ok:
            QMessageBox.warning(self, "Load Failed", f"Session {session_id} failed to load. Please check data files.")
            return
        n = self.dm.get_frame_count()
        self._total_frames = n
        self._progress.setMaximum(max(0, n - 1))
        self._spin_jump.setMaximum(max(0, n - 1))
        self.update_frame_display(0, 0.0, 0.0, 0.0, "P")
        self.sig_session_loaded.emit(session_id)

    def update_frame_display(
        self, frame_id: int, timestamp_ms: float,
        speed: float, steering: float, gear: str
    ) -> None:
        """Called by main loop to refresh progress bar and metadata display."""
        self._progress.blockSignals(True)
        self._progress.setValue(frame_id)
        self._progress.blockSignals(False)

        self._lbl_frame.setText(f"Frame: {frame_id} / {self._total_frames - 1}")
        self._spin_jump.blockSignals(True)
        self._spin_jump.setValue(frame_id)
        self._spin_jump.blockSignals(False)

        total_s = timestamp_ms / 1000.0
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = total_s % 60
        self._lbl_time.setText(f"{h:02d}:{m:02d}:{s:06.3f}")

        self._meta_labels["frame_id"].setText(str(frame_id))
        self._meta_labels["timestamp"].setText(f"{timestamp_ms:.1f} ms")
        self._meta_labels["speed"].setText(f"{speed:.1f}")
        self._meta_labels["steering"].setText(f"{steering:.1f}")
        self._meta_labels["gear"].setText(gear)

    # ─────────────────────────────────────────
    # Slot functions
    # ─────────────────────────────────────────

    def _on_play_pause(self, checked: bool) -> None:
        if checked:
            self._btn_play.setText("⏸ Pause")
            self.sig_play.emit()
        else:
            self._btn_play.setText("▶ Play")
            self.sig_pause.emit()

    def on_playback_ended(self) -> None:
        """Called externally to reset play button state when playback ends."""
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶ Play")

    def _on_progress_changed(self, value: int) -> None:
        self.sig_seek.emit(value)

    def _on_speed_changed(self, idx: int) -> None:
        speeds = [0.25, 0.5, 1.0, 2.0, 5.0]
        self.sig_speed_changed.emit(speeds[idx])

    def _on_noise_toggle(self, enabled: bool) -> None:
        self.dm.noise_injector.enabled = enabled
        self._on_noise_param_changed()

    def _on_noise_param_changed(self) -> None:
        ni = self.dm.noise_injector
        ni.gaussian_snr_db = self._spin_snr.value()
        ni.dropout_prob = self._spin_dropout.value()
        ni.temp_shift_samples = self._spin_shift.value()

    def _on_batch_evaluate(self) -> None:
        items = self._session_tree.selectedItems()
        ids = [item.data(0, Qt.ItemDataRole.UserRole) for item in items if item.data(0, Qt.ItemDataRole.UserRole)]
        if not ids:
            ids = [self._session_tree.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
                   for i in range(self._session_tree.topLevelItemCount())]
        if ids:
            self.sig_batch_evaluate.emit(ids)
