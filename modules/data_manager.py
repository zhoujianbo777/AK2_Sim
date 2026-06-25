"""
data_manager.py  —  Test data management module
Responsible for TestData/ directory scanning, session loading, frame sequence management,
playback control, and noise injection.

DataFrame field specification (see design spec section 8.1):
    frame_id           int        Frame index (0-based)
    timestamp_ms       float      Frame timestamp (ms)
    edi_distance       [12]f32    12-channel EDI distance (m)
    edi_amplitude      [12]f32    12-channel EDI amplitude (normalized)
    edi_confidence     [12]u8     12-channel EDI confidence (0~100)
    edi_echo_type      [12]u8     12-channel echo type (0~3)
    envelopes          [12,256]f32 12-channel normalized envelope sequence
    elastic_features   [12,20]f32  12-channel elastic wave feature matrix
    vehicle_speed      float      Vehicle speed (km/h)
    steering_angle     float      Steering angle (deg)
    gear               str        Gear position (P/R/N/D)
"""

import os
import json
import struct
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from modules.config_loader import ConfigLoader


# ─────────────────────────────────────────────────────────────
# DataFrame dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class DataFrame:
    frame_id: int = 0
    timestamp_ms: float = 0.0
    edi_distance: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    edi_amplitude: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    edi_confidence: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.uint8))
    edi_echo_type: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.uint8))
    envelopes: np.ndarray = field(default_factory=lambda: np.zeros((12, 256), dtype=np.float32))
    elastic_features: np.ndarray = field(default_factory=lambda: np.zeros((12, 20), dtype=np.float32))
    vehicle_speed: float = 0.0
    steering_angle: float = 0.0
    gear: str = "P"


@dataclass
class SessionMeta:
    session_id: str = ""
    date: str = ""
    weather: str = ""
    temperature_c: float = 25.0
    vehicle_model: str = ""
    road_type: str = ""
    total_frames: int = 0
    has_ground_truth: bool = False
    session_path: str = ""


# ─────────────────────────────────────────────────────────────
# Noise injector
# ─────────────────────────────────────────────────────────────

class NoiseInjector:
    """Applies configurable noise to data frames in memory without modifying source files."""

    def __init__(self):
        self.enabled: bool = False
        self.gaussian_snr_db: float = 40.0       # Gaussian white noise SNR (dB)
        self.fault_channels: list[int] = []      # Completely failed channel indices (0-based)
        self.dropout_prob: float = 0.0            # Random frame dropout probability (0~0.5)
        self.temp_shift_samples: int = 0          # Envelope time-delay shift (±10 samples)
        self.elastic_noise_multiplier: float = 1.0  # Elastic wave noise energy multiplier

    def apply(self, frame: DataFrame) -> DataFrame:
        """Apply noise injection to a frame; returns a deep copy (original frame unchanged)."""
        if not self.enabled:
            return frame

        import copy
        f = copy.deepcopy(frame)

        # Add Gaussian white noise to envelopes
        if self.gaussian_snr_db < 50.0:
            signal_power = np.mean(f.envelopes ** 2)
            noise_power = signal_power / (10 ** (self.gaussian_snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), f.envelopes.shape).astype(np.float32)
            f.envelopes = np.clip(f.envelopes + noise, 0.0, 1.0)

        # Single-channel sensor complete fault (zero out)
        for ch in self.fault_channels:
            if 0 <= ch < 12:
                f.envelopes[ch] = np.zeros(256, dtype=np.float32)
                f.edi_distance[ch] = 0.0
                f.edi_amplitude[ch] = 0.0
                f.edi_confidence[ch] = 0

        # Random dropout (zero confidence to simulate lost frames)
        if self.dropout_prob > 0:
            mask = np.random.random(12) < self.dropout_prob
            f.edi_confidence[mask] = 0

        # Envelope time-delay shift (simulate temperature drift)
        if self.temp_shift_samples != 0:
            f.envelopes = np.roll(f.envelopes, self.temp_shift_samples, axis=1)

        # Raise elastic wave background noise floor
        if self.elastic_noise_multiplier > 1.0:
            baseline_rms = np.sqrt(np.mean(f.elastic_features ** 2))
            noise = np.random.normal(
                0,
                baseline_rms * (self.elastic_noise_multiplier - 1.0),
                f.elastic_features.shape
            ).astype(np.float32)
            f.elastic_features += noise

        return f


# ─────────────────────────────────────────────────────────────
# DataManager main class
# ─────────────────────────────────────────────────────────────

class DataManager:
    """
    Manages loading and frame-sequence playback for all sessions under TestData/.
    Provides playback controls (play/pause/step/seek/speed) and noise injection.
    """

    # Playback speed factor → timer interval (ms)
    SPEED_FACTORS = {0.25: 4000, 0.5: 2000, 1.0: 1000, 2.0: 500, 5.0: 200}

    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg
        self.logger = logging.getLogger("DataManager")
        self.testdata_root: str = cfg.get("data.testdata_root", "./TestData")

        self._sessions: list[SessionMeta] = []
        self._frames: list[DataFrame] = []
        self._current_frame_idx: int = 0
        self._is_playing: bool = False
        self._speed: float = 1.0
        self._loop: bool = False
        self._range_start: int = 0
        self._range_end: int = 0

        self.ground_truth: Optional[dict] = None
        self.noise_injector = NoiseInjector()

        self._scan_sessions()

    # ── Directory scan ──────────────────────────────

    def _scan_sessions(self) -> None:
        """Scan the TestData/ directory and build the session index."""
        self._sessions.clear()
        index_path = os.path.join(self.testdata_root, "index.json")

        if not os.path.isdir(self.testdata_root):
            self.logger.warning("TestData root not found: %s", self.testdata_root)
            return

        for entry in sorted(os.listdir(self.testdata_root)):
            session_path = os.path.join(self.testdata_root, entry)
            if not os.path.isdir(session_path):
                continue
            meta_path = os.path.join(session_path, "session_meta.json")
            if not os.path.exists(meta_path):
                continue
            meta = self._load_session_meta(session_path, entry)
            self._sessions.append(meta)

        self.logger.info(
            "Scan complete: %d sessions found in %s",
            len(self._sessions), self.testdata_root
        )
        for s in self._sessions:
            gt_tag = " [GT]" if s.has_ground_truth else ""
            self.logger.info(
                "  Session: %s | %d frames | %s | %s%s",
                s.session_id, s.total_frames, s.date, s.weather, gt_tag
            )

        # Update index.json
        index_data = {
            "version": "1.0",
            "generated": __import__("datetime").date.today().isoformat(),
            "sessions": [s.session_id for s in self._sessions]
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def _load_session_meta(self, session_path: str, session_id: str) -> SessionMeta:
        """Read session metadata from session_meta.json."""
        meta_path = os.path.join(session_path, "session_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        gt_path = os.path.join(session_path, "ground_truth.json")
        return SessionMeta(
            session_id=session_id,
            date=raw.get("date", ""),
            weather=raw.get("weather", ""),
            temperature_c=raw.get("temperature_c", 25.0),
            vehicle_model=raw.get("vehicle_model", ""),
            road_type=raw.get("road_type", ""),
            total_frames=raw.get("total_frames", 0),
            has_ground_truth=os.path.exists(gt_path),
            session_path=session_path,
        )

    # ── Session loading ──────────────────────────────

    def get_session_list(self) -> list[SessionMeta]:
        """Return the list of all scanned session metadata."""
        return self._sessions

    def load_session(self, session_id: str) -> bool:
        """
        Load all frame data for the specified session.
        Returns True on success, False on failure.
        """
        meta = next((s for s in self._sessions if s.session_id == session_id), None)
        if meta is None:
            return False

        frames_path = os.path.join(meta.session_path, "esi_frames.bin")
        can_path = os.path.join(meta.session_path, "can_signals.csv")
        gt_path = os.path.join(meta.session_path, "ground_truth.json")

        self._frames = self._parse_esi_frames(frames_path, can_path)
        self._current_frame_idx = 0
        self._range_start = 0
        self._range_end = max(0, len(self._frames) - 1)
        self._is_playing = False

        if os.path.exists(gt_path):
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_raw = json.load(f)
            # Build frame_id (str) → channel_labels lookup for fast per-frame access
            self.ground_truth = {
                str(entry["frame_id"]): entry["channel_labels"]
                for entry in gt_raw.get("frame_labels", [])
            }
        else:
            self.ground_truth = None

        ok = len(self._frames) > 0
        if ok:
            self.logger.info(
                "Session '%s' loaded: %d frames | GT=%s | ESI=%s | CAN=%s",
                session_id, len(self._frames),
                "yes" if self.ground_truth else "no",
                os.path.basename(frames_path),
                os.path.basename(can_path) if os.path.exists(can_path) else "n/a"
            )
        else:
            self.logger.error("Session '%s' loaded 0 frames (ESI file missing or empty).", session_id)
        return ok

    def _parse_esi_frames(self, frames_path: str, can_path: str) -> list[DataFrame]:
        """
        Parse esi_frames.bin binary file into a list of DataFrame objects.

        Binary frame format (fixed length per frame):
          4B  frame_id (uint32)
          4B  timestamp_ms (float32)
          12×4B  edi_distance (float32×12)
          12×4B  edi_amplitude (float32×12)
          12×1B  edi_confidence (uint8×12)
          12×1B  edi_echo_type (uint8×12)
          12×256×4B  envelopes (float32×12×256)
          12×20×4B   elastic_features (float32×12×20)
        Total: 4+4+48+48+12+12+12288+960 = 13376 bytes/frame
        """
        FRAME_SIZE = 13376
        frames = []

        if not os.path.exists(frames_path):
            return frames

        # Load CAN signals (timestamp-aligned)
        can_data = self._load_can_signals(can_path)
        can_ts_sorted = sorted(can_data.keys())   # ascending, for bisect nearest-neighbour

        with open(frames_path, "rb") as f:
            raw = f.read()

        offset = 0
        while offset + FRAME_SIZE <= len(raw):
            df = DataFrame()
            df.frame_id = struct.unpack_from("<I", raw, offset)[0]; offset += 4
            df.timestamp_ms = struct.unpack_from("<f", raw, offset)[0]; offset += 4
            df.edi_distance = np.frombuffer(raw, dtype=np.float32, count=12, offset=offset).copy(); offset += 48
            df.edi_amplitude = np.frombuffer(raw, dtype=np.float32, count=12, offset=offset).copy(); offset += 48
            df.edi_confidence = np.frombuffer(raw, dtype=np.uint8, count=12, offset=offset).copy(); offset += 12
            df.edi_echo_type = np.frombuffer(raw, dtype=np.uint8, count=12, offset=offset).copy(); offset += 12
            df.envelopes = np.frombuffer(raw, dtype=np.float32, count=12*256, offset=offset).reshape(12, 256).copy(); offset += 12*256*4
            df.elastic_features = np.frombuffer(raw, dtype=np.float32, count=12*20, offset=offset).reshape(12, 20).copy(); offset += 12*20*4

            # Sync CAN signals (nearest-neighbor timestamp match)
            can = can_data.get(df.timestamp_ms) or self._nearest_can(
                can_data, can_ts_sorted, df.timestamp_ms
            )
            if can:
                df.vehicle_speed = can.get("speed", 0.0)
                df.steering_angle = can.get("steering_angle", 0.0)
                df.gear = can.get("gear", "P")

            frames.append(df)

        return frames

    def _load_can_signals(self, can_path: str) -> dict:
        """Load can_signals.csv; return {timestamp_ms: {speed, steering_angle, gear}} dict."""
        can_data = {}
        if not os.path.exists(can_path):
            return can_data
        import csv
        with open(can_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = float(row["timestamp_ms"])
                    can_data[ts] = {
                        "speed": float(row.get("speed_kmh", 0.0)),
                        "steering_angle": float(row.get("steering_angle_deg", 0.0)),
                        "gear": row.get("gear", "P"),
                    }
                except (ValueError, KeyError):
                    continue
        return can_data

    def _nearest_can(self, can_data: dict, can_ts_sorted: list,
                     timestamp_ms: float) -> Optional[dict]:
        """Find the nearest-neighbor timestamp entry via bisect on a sorted key list.

        O(log N) per lookup instead of the previous O(N) linear scan; CAN and
        frame timestamps rarely match exactly (float32 vs float64), so this path
        is taken almost every frame.
        """
        if not can_ts_sorted:
            return None
        import bisect
        pos = bisect.bisect_left(can_ts_sorted, timestamp_ms)
        if pos == 0:
            closest_ts = can_ts_sorted[0]
        elif pos >= len(can_ts_sorted):
            closest_ts = can_ts_sorted[-1]
        else:
            before, after = can_ts_sorted[pos - 1], can_ts_sorted[pos]
            closest_ts = before if (timestamp_ms - before) <= (after - timestamp_ms) else after
        return can_data[closest_ts]

    # ── Playback control ──────────────────────────────

    def get_current_frame(self) -> Optional[DataFrame]:
        """Get the current frame (with noise injection applied)."""
        if not self._frames or self._current_frame_idx >= len(self._frames):
            return None
        frame = self._frames[self._current_frame_idx]
        return self.noise_injector.apply(frame)

    def get_frame_count(self) -> int:
        return len(self._frames)

    def get_current_index(self) -> int:
        return self._current_frame_idx

    def step_forward(self) -> bool:
        """Step forward one frame; returns True if end of range reached."""
        if self._current_frame_idx < self._range_end:
            self._current_frame_idx += 1
            return False
        if self._loop:
            self._current_frame_idx = self._range_start
        return True

    def step_backward(self) -> bool:
        """Step backward one frame; returns True if start of range reached."""
        if self._current_frame_idx > self._range_start:
            self._current_frame_idx -= 1
            return False
        return True

    def seek(self, frame_idx: int) -> None:
        """Seek to the specified frame index."""
        self._current_frame_idx = max(self._range_start, min(frame_idx, self._range_end))

    def set_play_range(self, start: int, end: int) -> None:
        """Set the start and end frame indices for range playback."""
        self._range_start = max(0, start)
        self._range_end = min(len(self._frames) - 1, end)

    def set_speed(self, speed: float) -> None:
        """Set playback speed (0.25/0.5/1.0/2.0/5.0)."""
        if speed in self.SPEED_FACTORS:
            self._speed = speed

    def get_timer_interval_ms(self) -> int:
        """Return the timer trigger interval (ms) for the current playback speed."""
        return self.SPEED_FACTORS.get(self._speed, 1000)

    def set_loop(self, loop: bool) -> None:
        self._loop = loop

    def play(self) -> None:
        self._is_playing = True

    def pause(self) -> None:
        self._is_playing = False

    def is_playing(self) -> bool:
        return self._is_playing
