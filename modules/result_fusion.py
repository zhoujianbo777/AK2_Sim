"""
result_fusion.py  —  Result fusion module + observer-pattern notification hub
Holds the current frame's inference result, maintains a history buffer,
and notifies all subscribed display windows to refresh.

Algorithm engine output spec (see design spec section 8.2):
    frame_id              int        Frame index
    valid_flags           [12]f32    M1 validity probability
    class_probs           [12,9]f32  M2 9-class probabilities (AI) or [12,5] (traditional)
    class_ids             [12]u8     Primary class ID (argmax)
    material_hardness     [12]f32    Material hardness score (AI only)
    suspension_height_m   [12]f32    Overhead clearance estimate (AI only; -1 if not overhead)
    collision_probs       [4]f32     M3 4-class collision probabilities (AI only)
    collision_type        uint8      Collision type ID (0~3)
    anomaly_score         float32    M4 VAE anomaly score (AI only)
    inference_time_ms     float      Inference latency (ms)
    engine_type           str        "AI" or "Traditional"
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from modules.config_loader import ConfigLoader


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class AlgoResult:
    frame_id: int = 0
    valid_flags: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    class_probs: np.ndarray = field(default_factory=lambda: np.zeros((12, 9), dtype=np.float32))
    class_ids: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.uint8))
    # M2-SC backbone output (before SA refinement); mirrors class_probs/class_ids when SA disabled
    sc_class_probs: np.ndarray = field(default_factory=lambda: np.zeros((12, 9), dtype=np.float32))
    sc_class_ids: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.uint8))
    material_hardness: np.ndarray = field(default_factory=lambda: np.full(12, -1.0, dtype=np.float32))
    suspension_height_m: np.ndarray = field(default_factory=lambda: np.full(12, -1.0, dtype=np.float32))
    # Estimated lateral width of each detected obstacle (m); -1.0 = not estimated.
    # Derived geometrically from envelope half-power width + sensor FOV + distance.
    object_width_m: np.ndarray = field(default_factory=lambda: np.full(12, -1.0, dtype=np.float32))
    collision_probs: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    collision_type: int = 0
    anomaly_score: float = 0.0
    inference_time_ms: float = 0.0
    engine_type: str = "Traditional"
    # M6 OGM probability grid [H, W] = [160, 130] @ 0.1 m/cell; zeros when M6 disabled
    ogm_grid: np.ndarray = field(default_factory=lambda: np.zeros((160, 130), dtype=np.float32))
    # M6 free-space (drivable) probability grid [H, W]; zeros when M6 disabled / 1-ch model
    ogm_free: np.ndarray = field(default_factory=lambda: np.zeros((160, 130), dtype=np.float32))


# ─────────────────────────────────────────────────────────────
# Observer interface
# ─────────────────────────────────────────────────────────────

# Subscriber callback type: receives AlgoResult
ResultCallback = Callable[["AlgoResult"], None]


# ─────────────────────────────────────────────────────────────
# ResultFusion main class
# ─────────────────────────────────────────────────────────────

class ResultFusion:
    """
    Holds the latest inference result and a history buffer.
    Display windows register via subscribe() and are notified automatically on each update.
    """

    def __init__(self, cfg: ConfigLoader):
        cache_size = cfg.get("engine.result_cache_frames", 200)
        self._cache: deque[AlgoResult] = deque(maxlen=cache_size)
        self._latest: Optional[AlgoResult] = None
        self._subscribers: list[ResultCallback] = []

        # AI advantage comparison buffer (used in comparison mode)
        self._compare_result: Optional[AlgoResult] = None

    def subscribe(self, callback: ResultCallback) -> None:
        """Register a result-update callback (observer pattern)."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: ResultCallback) -> None:
        self._subscribers.remove(callback)

    def update(self, result: AlgoResult) -> None:
        """
        Receive the latest inference result from the algorithm engine,
        store it in the cache, and notify all subscribers.
        """
        self._latest = result
        self._cache.append(result)
        self._notify(result)

    def _notify(self, result: AlgoResult) -> None:
        for cb in self._subscribers:
            try:
                cb(result)
            except Exception as e:
                print(f"[ResultFusion] Subscriber callback error: {e}")

    def get_latest(self) -> Optional[AlgoResult]:
        return self._latest

    def get_history(self, n: int = 5) -> list[AlgoResult]:
        """Return the most recent n results (newest first)."""
        history = list(self._cache)
        return list(reversed(history[-n:]))

    def get_by_frame_id(self, frame_id: int) -> Optional[AlgoResult]:
        """Look up a cached result by frame ID (avoids redundant inference)."""
        for result in reversed(self._cache):
            if result.frame_id == frame_id:
                return result
        return None

    def reset(self) -> None:
        """Clear cache when switching sessions to prevent stale frame_id hits."""
        self._cache.clear()
        self._latest = None
        self._compare_result = None

    def set_compare_result(self, result: AlgoResult) -> None:
        """Store the comparison instance's latest result (used for AI advantage detection)."""
        self._compare_result = result

    def check_ai_advantage(self, ai_result: AlgoResult, trad_result: AlgoResult) -> Optional[str]:
        """
        Compare AI and traditional results. If the AI detects a target missed or
        misclassified by the traditional algorithm, return a description string;
        otherwise return None.
        """
        # Overhead obstacle detection (class 6, AI only)
        for ch in range(12):
            ai_cls = ai_result.class_ids[ch]
            trad_cls = trad_result.class_ids[ch]
            if ai_cls == 6 and trad_cls != 6:
                h = ai_result.suspension_height_m[ch]
                return f"AI detected overhead obstacle (sensor S{ch+1:02d}, clearance≈{h:.2f}m); traditional algo misclassified as class {trad_cls}"
            if ai_cls == 8 and trad_cls not in (4, 8):
                return f"AI detected wet ground (sensor S{ch+1:02d}); traditional algo false alarm as class {trad_cls}"
            if ai_cls == 7 and trad_cls not in (7,):
                return f"AI detected curb (sensor S{ch+1:02d}); traditional algo missed"
        return None
