"""
window_statistics.py  —  W6 Performance Statistics Panel
Accumulates statistics in real time; only active when session includes ground_truth.json.
See spec section 5.2.6.

Features:
  - Processed frames / total frames
  - Per-class precision, recall (table)
  - Overall accuracy progress bar
  - Confusion matrix heatmap
  - Export Markdown report
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QTableWidget, QTableWidgetItem, QPushButton, QGroupBox,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

from modules.result_fusion import AlgoResult
from modules.config_loader import ConfigLoader
from modules.obstacle_classes import N_CLASSES, CLASS_NAMES


class WindowStatistics(QWidget):
    """W6 Performance Statistics Panel."""

    def __init__(self, cfg: ConfigLoader, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._n_classes = N_CLASSES
        self._class_names = list(CLASS_NAMES)
        self._confusion    = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # SA (final)
        self._sc_confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # SC backbone
        self._total_frames = 0
        self._processed_frames = 0
        self._engine_type = "AI"
        self._session_id = ""
        self._has_gt = True
        self._sa_enabled = cfg.get("ai.m2_sa_enabled", False)
        self._reports_dir = cfg.get("data.reports_output", "./reports")

        self.setWindowTitle("W6 - Performance Statistics")
        self.resize(720 if self._sa_enabled else 560, 520)
        self._build_ui()

        # Confusion matrix redraws are expensive (matplotlib sync render).
        # Throttle to at most one redraw per 500 ms via a QTimer so the
        # main thread is never blocked during session load or playback.
        self._cm_dirty = False
        self._cm_timer = QTimer(self)
        self._cm_timer.setSingleShot(True)
        self._cm_timer.setInterval(500)
        self._cm_timer.timeout.connect(self._on_cm_timer)

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Progress area ──
        prog_grp = QGroupBox("Evaluation Progress")
        prog_lay = QVBoxLayout(prog_grp)
        self._lbl_progress = QLabel("No session loaded")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Overall Accuracy: 0%")
        self._progress_bar.setFixedHeight(22)
        prog_lay.addWidget(self._lbl_progress)
        prog_lay.addWidget(self._progress_bar)
        root.addWidget(prog_grp)

        # ── Left/right layout ──
        split = QHBoxLayout()

        # Left: metrics table
        left = QVBoxLayout()
        if self._sa_enabled:
            left.addWidget(QLabel("Per-Class Metrics  SC / SA (Precision | Recall | F1)"))
            self._table = QTableWidget(N_CLASSES, 7)
            self._table.setHorizontalHeaderLabels(
                ["Class", "SC-P", "SC-R", "SC-F1", "SA-P", "SA-R", "SA-F1"])
            self._table.setMinimumWidth(300)
            self._table.setMaximumWidth(440)
        else:
            left.addWidget(QLabel("Per-Class Metrics (Precision / Recall / F1)"))
            self._table = QTableWidget(N_CLASSES, 4)
            self._table.setHorizontalHeaderLabels(["Class", "Precision", "Recall", "F1"])
            self._table.setMinimumWidth(240)
            self._table.setMaximumWidth(320)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._init_table()
        left.addWidget(self._table)
        split.addLayout(left)

        # Right: confusion matrix heatmap(s)
        right = QVBoxLayout()
        if self._sa_enabled:
            right.addWidget(QLabel("Confusion Matrix  ·  left: SC backbone   right: SA refined"))
            self._fig_cm = Figure(figsize=(6.0, 3.0), tight_layout=True)
            self._ax_sc = self._fig_cm.add_subplot(121)
            self._ax_cm = self._fig_cm.add_subplot(122)   # ax_cm = SA axis (existing name)
        else:
            right.addWidget(QLabel("Confusion Matrix"))
            self._fig_cm = Figure(figsize=(3.2, 3.0), tight_layout=True)
            self._ax_cm = self._fig_cm.add_subplot(111)
            self._ax_sc = None
        self._canvas_cm = FigureCanvas(self._fig_cm)
        # Allow the canvas to shrink so it never forces the left column wide
        self._canvas_cm.setMinimumWidth(260)
        right.addWidget(self._canvas_cm)
        split.addLayout(right)

        root.addLayout(split)

        # Export buttons
        btn_row = QHBoxLayout()
        btn_export = QPushButton("📄 Export Markdown Report")
        btn_export.clicked.connect(self._on_export)
        btn_reset = QPushButton("🔄 Reset Statistics")
        btn_reset.clicked.connect(self.reset)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _init_table(self):
        n_cols = 7 if self._sa_enabled else 4
        for row in range(N_CLASSES):
            name = CLASS_NAMES[row] if row < len(CLASS_NAMES) else f"Class{row}"
            self._table.setItem(row, 0, QTableWidgetItem(name))
            for col in range(1, n_cols):
                self._table.setItem(row, col, QTableWidgetItem("-"))
        self._table.resizeColumnsToContents()

    # ── Public interface ──────────────────────────────

    def set_session(self, session_id: str, total_frames: int, n_classes: int,
                    engine_type: str, has_gt: bool = True) -> None:
        self._session_id = session_id
        self._total_frames = total_frames
        self._n_classes = N_CLASSES   # always 9-class, unified
        self._class_names = list(CLASS_NAMES)
        self._engine_type = engine_type
        self._has_gt = has_gt
        self.reset()
        self._table.setRowCount(N_CLASSES)
        for row in range(N_CLASSES):
            name = CLASS_NAMES[row]
            self._table.setItem(row, 0, QTableWidgetItem(name))
        self._confusion    = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        self._sc_confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        self._lbl_progress.setText(f"Session: {session_id}  |  Engine: {engine_type}  |  Frames: {total_frames}")
        if not has_gt:
            self._progress_bar.setValue(0)
            self._progress_bar.setFormat("无标注数据 — 本 session 不可评估")
            self._init_table()  # restore "—" placeholders

    def update_with_gt(self, result: AlgoResult, gt_class_ids: np.ndarray) -> None:
        """
        Update confusion matrix with GT labels.
        gt_class_ids: [12] uint8, true class ID per channel
        """
        self._processed_frames += 1
        n = self._n_classes
        for ch in range(12):
            true = int(gt_class_ids[ch])
            if not (0 <= true < n):
                continue
            pred = int(result.class_ids[ch])
            if 0 <= pred < n:
                self._confusion[true, pred] += 1
            # SC backbone confusion (mirrors main when SA disabled)
            sc_ids = getattr(result, "sc_class_ids", result.class_ids)
            sc_pred = int(sc_ids[ch])
            if 0 <= sc_pred < n:
                self._sc_confusion[true, sc_pred] += 1
        self._refresh_metrics()

    def reset(self) -> None:
        self._confusion    = np.zeros((self._n_classes, self._n_classes), dtype=int)
        self._sc_confusion = np.zeros((self._n_classes, self._n_classes), dtype=int)
        self._processed_frames = 0
        self._refresh_metrics()

    # ── Internal refresh ──────────────────────────────

    def _refresh_metrics(self):
        if not self._has_gt:
            return
        n = self._n_classes
        cm    = self._confusion       # SA / final
        sc_cm = self._sc_confusion    # SC backbone
        total_correct = int(np.trace(cm))
        total_samples = int(cm.sum())
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        pct = int(accuracy * 100)
        self._progress_bar.setValue(pct)
        self._progress_bar.setFormat(f"Accuracy: {pct}%  ({self._processed_frames}/{self._total_frames} frames)")

        for i in range(n):
            if self._sa_enabled:
                # SC columns (1-3) and SA columns (4-6)
                for cm_use, col_start in [(sc_cm, 1), (cm, 4)]:
                    tp = cm_use[i, i]
                    fp = cm_use[:, i].sum() - tp
                    fn = cm_use[i, :].sum() - tp
                    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    for off, val in enumerate([p, r, f1]):
                        item = self._table.item(i, col_start + off)
                        if item:
                            item.setText(f"{val:.3f}")
            else:
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                for col, val in enumerate([p, r, f1], start=1):
                    item = self._table.item(i, col)
                    if item:
                        item.setText(f"{val:.3f}")

        # Schedule a confusion matrix redraw (at most once per 500 ms)
        self._cm_dirty = True
        if not self._cm_timer.isActive():
            self._cm_timer.start()

    def _on_cm_timer(self):
        """Timer callback: redraw confusion matrix if data changed since last draw."""
        if self._cm_dirty:
            self._redraw_confusion()
            self._cm_dirty = False

    def _redraw_confusion(self):
        n     = self._n_classes
        names = self._class_names[:n]

        def _draw_cm(ax, cm_data, title):
            ax.clear()
            if cm_data.sum() == 0:
                msg = "\u65e0\u6807\u6ce8\u6570\u636e" if not self._has_gt else "No Data"
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
                return
            norm_cm = np.full((n, n), np.nan)
            row_sums = cm_data.astype(float).sum(axis=1, keepdims=True)
            mask = row_sums.squeeze() > 0
            norm_cm[mask] = (cm_data.astype(float)[mask] / row_sums[mask]) * 100
            cmap = plt.cm.RdYlGn.copy()
            cmap.set_bad(color="#D0D0D0")
            ax.imshow(np.ma.masked_invalid(norm_cm),
                      cmap=cmap, vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
            ax.set_title(title, fontsize=9)
            for i in range(n):
                for j in range(n):
                    val = int(cm_data[i, j])
                    if val > 0:
                        txt_color = "white" if norm_cm[i, j] < 50 else "black"
                        ax.text(j, i, f"{norm_cm[i, j]:.0f}", ha="center", va="center",
                                fontsize=6, color=txt_color)

        if self._sa_enabled and self._ax_sc is not None:
            _draw_cm(self._ax_sc, self._sc_confusion, "SC \u9aa8\u5e72 (%)")
            _draw_cm(self._ax_cm, self._confusion,    "SA \u7cbe\u5316 (%)")
        else:
            _draw_cm(self._ax_cm, self._confusion, "Confusion Matrix (%)")

        self._canvas_cm.draw()

    # ── Report export ──────────────────────────────

    def _on_export(self):
        os.makedirs(self._reports_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = os.path.join(self._reports_dir, f"report_{timestamp}.md")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_name, "Markdown Files (*.md)"
        )
        if not path:
            return
        try:
            self._write_report(path)
            QMessageBox.information(self, "Export Successful", f"Report saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _write_report(self, path: str):
        n = self._n_classes
        cm = self._confusion
        total = int(cm.sum())
        accuracy = int(np.trace(cm)) / total if total > 0 else 0.0
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# AK2 Simulator Performance Evaluation Report",
            f"",
            f"## 1. Summary",
            f"| Item | Value |",
            f"|------|-------|",
            f"| Evaluation Time | {now} |",
            f"| Session ID | {self._session_id} |",
            f"| Algorithm | {self._engine_type} |",
            f"| Frames Processed | {self._processed_frames} / {self._total_frames} |",
            f"| Overall Accuracy | {accuracy:.4f} |",
            f"",
            f"## 2. Per-Class Metrics",
            f"| Class | Precision | Recall | F1 |",
            f"|-------|-----------|--------|-----|",
        ]
        for i in range(n):
            tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            name = self._class_names[i] if i < len(self._class_names) else f"Class{i}"
            lines.append(f"| {name} | {p:.4f} | {r:.4f} | {f1:.4f} |")

        lines += ["", "## 3. Confusion Matrix (sample counts)",
                  "| GT\\Pred | " + " | ".join(self._class_names[:n]) + " |",
                  "|" + "---|" * (n + 1)]
        for i in range(n):
            row_name = self._class_names[i] if i < len(self._class_names) else f"Class{i}"
            row_vals = " | ".join(str(cm[i, j]) for j in range(n))
            lines.append(f"| {row_name} | {row_vals} |")

        lines += ["", "---", "*Auto-generated by AK2 PC Simulator*"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
