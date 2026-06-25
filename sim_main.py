"""
sim_main.py  —  AK2 PC-side simulator main entry point
See spec sections 7.1 (startup flow) and 7.2 (single-session playback flow).

Usage:
  python sim_main.py                              # default mode (traditional)
  python sim_main.py --mode ai                    # AI algorithm mode
  python sim_main.py --mode traditional           # traditional algorithm mode
  python sim_main.py --mode ai --ipc-role slave   # compare mode slave node
  python sim_main.py --mode traditional --ipc-role master  # compare mode master node
"""

import sys
import os
import argparse
import logging
import datetime

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

# ── Path setup (ensure imports from project root) ──
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from modules.config_loader import ConfigLoader
from modules.data_manager import DataManager
from modules.result_fusion import ResultFusion, AlgoResult
from modules.engine_traditional import TraditionalEngine
from modules.engine_ai import AIEngine
from modules.ipc_sync import IpcSyncManager
from modules.data_manager import DataFrame

from views.window_control import WindowControl
from views.window_display import DisplayWindow


# ─────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"sim_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ─────────────────────────────────────────────────────────────
# Application main controller class
# ─────────────────────────────────────────────────────────────

class AK2SimApp:
    """
    Simulator main application class.
    Coordinates data management, algorithm engine, result fusion, and display windows.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = logging.getLogger("AK2SimApp")

        # Load config
        self.logger.info("Loading config.yaml ...")
        self.cfg = ConfigLoader(os.path.join(ROOT_DIR, "config.yaml"))

        # Init logging
        setup_logging(self.cfg.get("data.logs_output", "./logs"))
        os.makedirs(self.cfg.get("data.reports_output", "./reports"), exist_ok=True)

        # Determine run mode
        self.mode = args.mode or self.cfg.get("engine.default_mode", "traditional")
        self.ipc_role = args.ipc_role  # "master" / "slave" / None
        self.logger.info("Run mode: %s | IPC role: %s", self.mode.upper(), self.ipc_role or "none")

        # Init core modules
        self.logger.info("Initializing DataManager ...")
        self.data_manager = DataManager(self.cfg)
        self.logger.info("Initializing ResultFusion ...")
        self.result_fusion = ResultFusion(self.cfg)

        # Init algorithm engine
        self.logger.info("Initializing %s engine ...",
                         "AIEngine" if self.mode == "ai" else "TraditionalEngine")
        if self.mode == "ai":
            self.engine = AIEngine(self.cfg)
        else:
            self.engine = TraditionalEngine(self.cfg)

        # IPC sync (compare mode)
        self.ipc: IpcSyncManager | None = None
        if self.ipc_role in ("master", "slave"):
            self.logger.info("Initializing IPC sync manager (role: %s) ...", self.ipc_role)
            self.ipc = IpcSyncManager(self.ipc_role)

        # Playback timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)

        # Init GUI windows
        self.logger.info("Creating GUI windows ...")
        self._init_windows()
        self._connect_signals()
        self._subscribe_results()

        # Start IPC
        if self.ipc:
            self._start_ipc()

        # Show all windows
        self.logger.info("Showing windows ...")
        self._show_windows()

        self.logger.info("AK2 simulator started. Mode: %s, IPC: %s",
                         self.mode.upper(), self.ipc_role or "none")
        if self.mode == "ai" and hasattr(self.engine, "is_sim_mode") and self.engine.is_sim_mode:
            self.logger.warning("AI engine running in simulation mode (DLL not found). Output is random data.")

    # ── Window initialization ──────────────────────────────

    def _init_windows(self):
        # W1: main control panel (standalone small window)
        self.w1 = WindowControl(self.cfg, self.data_manager)

        # Integrated display window (W2~W6 embedded)
        self.display = DisplayWindow(self.cfg, self.mode)

        engine_label = "AI Inference Engine"
        if self.mode == "ai" and hasattr(self.engine, "is_sim_mode") and self.engine.is_sim_mode:
            engine_label = "AI Inference Engine (Simulation Mode)"
        elif self.mode == "traditional":
            engine_label = "Traditional Threshold Algorithm"
        self.w1.set_engine_type(engine_label)

    def _connect_signals(self):
        """Connect W1 control signals to main application logic."""
        self.w1.sig_session_loaded.connect(self._on_session_loaded)
        self.w1.sig_play.connect(self._on_play)
        self.w1.sig_pause.connect(self._on_pause)
        self.w1.sig_seek.connect(self._on_seek)
        self.w1.sig_step_forward.connect(self._on_step_forward)
        self.w1.sig_step_backward.connect(self._on_step_backward)
        self.w1.sig_speed_changed.connect(self._on_speed_changed)
        self.w1.sig_loop_changed.connect(self.data_manager.set_loop)
        self.w1.sig_launch_compare.connect(self._on_launch_compare)
        self.w1.sig_batch_evaluate.connect(self._on_batch_evaluate)

    def _subscribe_results(self):
        """Subscribe callback to result fusion module to refresh display window."""
        self.result_fusion.subscribe(self.display.on_result_updated)

    def _show_windows(self):
        """Layout and show all windows."""
        from PyQt6.QtWidgets import QApplication

        screen = QApplication.primaryScreen().availableGeometry()
        mode_label = "Traditional" if self.mode == "traditional" else "AI"

        # W1 control panel: explicit position and size, top-left.
        # Stretch the height to the full available desktop so the bottom edge
        # sits flush with the screen (W5/W6 bottoms then align with it too).
        self.w1.setWindowTitle(f"W1 - Main Control Panel [{mode_label}]")
        ctrl_w = self.cfg.get("windows.control_panel.width", 420)
        ctrl_h = screen.height()
        gap    = self.cfg.get("windows.display_panel.gap", 10)
        self.w1.move(screen.left(), screen.top())
        self.w1.resize(ctrl_w, ctrl_h)
        self.w1.show()
        self._fit_height_to_screen(self.w1, screen)
        self.logger.info("W1 Control Panel shown: pos=(%d,%d) size=%dx%d",
                         screen.left(), screen.top(), ctrl_w, ctrl_h)

        # Integrated display window: starts right of control panel, same top edge
        self.display.setWindowTitle(f"AK2 Simulator — {mode_label} Display Panel")
        disp_x = screen.left() + ctrl_w + gap
        max_w  = screen.right() - disp_x + 1          # maximum usable width
        disp_w = self.cfg.get("windows.display_panel.width", max_w)
        disp_w = min(disp_w, max_w)                    # clamp to screen boundary
        disp_h = screen.height()                        # fill the desktop height
        self.display.move(disp_x, screen.top())
        self.display.resize(disp_w, disp_h)
        self.display.show()
        self._fit_height_to_screen(self.display, screen)
        self.logger.info("Display Panel shown: pos=(%d,%d) size=%dx%d",
                         disp_x, screen.top(), disp_w, disp_h)

    @staticmethod
    def _fit_height_to_screen(win, screen):
        """Shrink a top-level window so its outer frame (title bar + borders)
        fits within the available screen area, i.e. the bottom edge sits just
        above the Windows taskbar instead of disappearing behind it."""
        # resize() sets the client area; window decorations add extra height on
        # top, so subtract that overhead to keep the frame inside the desktop.
        overhead = win.frameGeometry().height() - win.height()
        if overhead > 0:
            win.resize(win.width(), max(200, screen.height() - overhead))
        win.move(win.x(), screen.top())

    # ── Session control ──────────────────────────────

    def _on_session_loaded(self, session_id: str):
        self.logger.info("Session loaded: %s", session_id)
        if hasattr(self.engine, "reset"):
            self.engine.reset()
            self.logger.info("Engine reset for new session.")

        # Clear inference cache — new session may reuse the same frame_id sequence
        self.result_fusion.reset()

        n = self.data_manager.get_frame_count()
        is_ai = self.mode == "ai"
        self.display.set_session(session_id, n, 9 if is_ai else 5, self.mode.upper())

        # Render frame 0
        self._render_current_frame()

        # IPC: notify peer to sync session
        if self.ipc and self.ipc_role == "master":
            session_path = ""
            for meta in self.data_manager.get_session_list():
                if meta.session_id == session_id:
                    session_path = meta.session_path
                    break
            self.ipc.send_session(session_path)

    # ── Playback control ──────────────────────────────

    def _on_play(self):
        self.data_manager.play()
        interval = self.data_manager.get_timer_interval_ms()
        self._timer.start(interval)
        self.logger.info("Play started. Timer interval=%d ms (speed=%.2fx)",
                         interval, self.data_manager._speed)
        if self.ipc and self.ipc_role == "master":
            self.ipc.send_play_state("PLAY")

    def _on_pause(self):
        self.data_manager.pause()
        self._timer.stop()
        self.logger.info("Playback paused at frame %d.", self.data_manager.get_current_index())
        if self.ipc and self.ipc_role == "master":
            self.ipc.send_play_state("PAUSE")

    def _on_seek(self, frame_idx: int):
        self.data_manager.seek(frame_idx)
        self._render_current_frame()

    def _on_step_forward(self):
        ended = self.data_manager.step_forward()
        self._render_current_frame()
        if ended:
            self.w1.on_playback_ended()

    def _on_step_backward(self):
        self.data_manager.step_backward()
        self._render_current_frame()

    def _on_speed_changed(self, speed: float):
        self.data_manager.set_speed(speed)
        if self._timer.isActive():
            self._timer.setInterval(self.data_manager.get_timer_interval_ms())
        self.logger.info("Playback speed changed to %.2fx (interval=%d ms).",
                         speed, self.data_manager.get_timer_interval_ms())
        if self.ipc and self.ipc_role == "master":
            self.ipc.send_speed(speed)

    def _on_timer_tick(self):
        """Timer tick: advance frame and render."""
        ended = self.data_manager.step_forward()
        self._render_current_frame()
        if ended and not self.data_manager._loop:
            self._timer.stop()
            self.data_manager.pause()
            self.w1.on_playback_ended()

        # IPC master: send frame advance signal
        if self.ipc and self.ipc_role == "master":
            self.ipc.send_frame_advance(self.data_manager.get_current_index())

    # ── Frame rendering ──────────────────────────────

    def _render_current_frame(self):
        """Get current frame → inference → refresh display window."""
        frame = self.data_manager.get_current_frame()
        if frame is None:
            return

        # Check inference cache
        cached = self.result_fusion.get_by_frame_id(frame.frame_id)
        if cached:
            result = cached
        else:
            result = self.engine.process(frame)
            self.result_fusion.update(result)

        # Refresh W1 frame metadata
        self.w1.update_frame_display(
            frame.frame_id, frame.timestamp_ms,
            frame.vehicle_speed, frame.steering_angle, frame.gear
        )

        # Refresh integrated display window (W2~W5)
        self.display.update_frame(frame, result)

        # Refresh W6 (GT evaluation)
        if self.data_manager.ground_truth:
            gt_frame = self.data_manager.ground_truth.get(str(frame.frame_id))
            if gt_frame:
                import numpy as np
                gt_ids = np.array(gt_frame.get("class_ids", [0] * 12), dtype="uint8")
                self.display.update_with_gt(result, gt_ids)

    # ── IPC ──────────────────────────────

    def _start_ipc(self):
        ok = self.ipc.start()
        if not ok:
            self.logger.error("IPC sync manager failed to start (role: %s).", self.ipc_role)
            return
        self.logger.info("IPC sync manager started (role: %s).", self.ipc_role)
        if self.ipc_role == "slave":
            # Slave: respond to frame advance
            self.ipc.on("FRAME_ADVANCE", lambda payload: self._on_ipc_frame_advance(payload))
            self.ipc.on("PLAY_STATE", lambda payload: self._on_ipc_play_state(payload))
            self.ipc.on("SPEED", lambda payload: self.data_manager.set_speed(float(payload)))
            self.ipc.on("SESSION", lambda payload: self.data_manager.load_session(
                os.path.basename(payload.strip())))

    def _on_ipc_frame_advance(self, payload: str):
        """Slave: received frame advance signal, seek and render, then ACK."""
        try:
            frame_id = int(payload.strip())
            self.data_manager.seek(frame_id)
            self._render_current_frame()
            self.ipc.send_ack(frame_id)
        except ValueError:
            pass

    def _on_ipc_play_state(self, payload: str):
        state = payload.strip()
        if state == "PLAY":
            self._on_play()
        elif state == "PAUSE":
            self._on_pause()

    # ── Misc ──────────────────────────────

    def _on_launch_compare(self):
        """Launch compare mode (run the other algorithm type in a new process)."""
        import subprocess
        other_mode = "ai" if self.mode == "traditional" else "traditional"
        cmd = [sys.executable, os.path.join(ROOT_DIR, "sim_main.py"),
               "--mode", other_mode, "--ipc-role", "slave"]
        subprocess.Popen(cmd)
        self.logger.info("Compare instance launched: mode=%s, role=slave", other_mode)

    def _on_batch_evaluate(self, session_ids: list[str]):
        """Batch evaluate multiple sessions."""
        from PyQt6.QtWidgets import QProgressDialog
        dlg = QProgressDialog("Batch evaluating...", "Cancel", 0, len(session_ids))
        dlg.setWindowTitle("Batch Evaluate")
        dlg.show()

        for idx, sid in enumerate(session_ids):
            if dlg.wasCanceled():
                break
            dlg.setLabelText(f"Evaluating: {sid} ({idx+1}/{len(session_ids)})")
            dlg.setValue(idx)
            QApplication.processEvents()

            if not self.data_manager.load_session(sid):
                continue
            if hasattr(self.engine, "reset"):
                self.engine.reset()

            for _ in range(self.data_manager.get_frame_count()):
                frame = self.data_manager.get_current_frame()
                if frame:
                    result = self.engine.process(frame)
                    self.result_fusion.update(result)
                self.data_manager.step_forward()

        dlg.setValue(len(session_ids))
        self.logger.info("Batch evaluation completed.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AK2 PC-Side Ultrasonic Perception Simulator")
    parser.add_argument("--mode", "-m", choices=["ai", "traditional"], default=None,
                        help="Algorithm mode: ai or traditional (default reads config.yaml)")
    # Convenience short flags: -ai / -trad  (mutually exclusive with --mode)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-ai", dest="_mode_ai", action="store_true",
                            help="Shorthand for --mode ai")
    mode_group.add_argument("-trad", dest="_mode_trad", action="store_true",
                            help="Shorthand for --mode traditional")
    parser.add_argument("--ipc-role", choices=["master", "slave"], default=None,
                        dest="ipc_role", help="Compare mode IPC role: master (Process A) or slave (Process B)")
    parser.add_argument("--config", default="./config.yaml", help="Config file path")
    args = parser.parse_args()
    # Resolve short flags into args.mode
    if args._mode_ai:
        args.mode = "ai"
    elif args._mode_trad:
        args.mode = "traditional"
    return args


def main():
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("AK2 Ultrasonic Perception Simulator")
    app.setOrganizationName("Bowei Yuanjing Technology")

    try:
        sim = AK2SimApp(args)
    except FileNotFoundError as e:
        QMessageBox.critical(None, "Startup Failed", f"Config file error: {e}\nPlease ensure config.yaml exists in the program root directory.")
        sys.exit(1)
    except Exception as e:
        QMessageBox.critical(None, "Startup Failed", f"Initialization error: {e}")
        raise

    ret = app.exec()
    # Deinit: shut down engine DLL if loaded
    logging.getLogger("AK2SimApp").info("Application closing. Shutting down engine ...")
    if hasattr(sim.engine, "shutdown"):
        sim.engine.shutdown()
    logging.getLogger("AK2SimApp").info("Shutdown complete.")
    sys.exit(ret)


if __name__ == "__main__":
    main()
