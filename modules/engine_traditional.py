"""
engine_traditional.py  —  Traditional threshold-based classification engine
Implemented entirely in Python without ML model dependencies.
Corresponds to design spec section 4.2.

Outputs class_ids and class_probs using the unified 9-class obstacle taxonomy
defined in modules/obstacle_classes.py.
"""

import time
import logging
import numpy as np
from collections import deque
from modules.data_manager import DataFrame
from modules.result_fusion import AlgoResult
from modules.config_loader import ConfigLoader
from modules.obstacle_classes import (
    N_CLASSES, TRAD_TO_UNIFIED,
    _TRAD_HARD, _TRAD_SOFT, _TRAD_OPEN, _TRAD_CLUTTER, _TRAD_UNKNOWN,
    ID_OPEN,
)

# Internal 5-rule label constants (private to this module)
CLS_HARD    = _TRAD_HARD
CLS_SOFT    = _TRAD_SOFT
CLS_OPEN    = _TRAD_OPEN
CLS_GROUND  = _TRAD_CLUTTER
CLS_UNKNOWN = _TRAD_UNKNOWN


class TraditionalEngine:
    """
    Traditional threshold-based classification engine.
    Uses rule-based hierarchical decisions without ML models.
    """

    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg
        self.logger = logging.getLogger("TraditionalEngine")

        # Read thresholds from configuration
        self.confidence_threshold: int = cfg.get("traditional.confidence_threshold", 50)
        self.amplitude_sigma: float = cfg.get("traditional.amplitude_sigma", 3.0)
        self.debounce_frames: int = cfg.get("traditional.debounce_frames", 3)
        self.elastic_rms_threshold: float = cfg.get("traditional.elastic_rms_threshold", 0.5)

        # Debounce state (per-channel consecutive valid frame count)
        self._debounce_counters = np.zeros(12, dtype=int)
        # Amplitude sliding window (for dynamic noise baseline estimation)
        self._amp_history: list[deque] = [deque(maxlen=30) for _ in range(12)]

        self.logger.info(
            "TraditionalEngine init: conf_threshold=%d | amp_sigma=%.1f | "
            "debounce=%d frames | elastic_rms_thr=%.2f",
            self.confidence_threshold, self.amplitude_sigma,
            self.debounce_frames, self.elastic_rms_threshold
        )

    def reset(self) -> None:
        """Reset engine state (called when switching sessions)."""
        self._debounce_counters = np.zeros(12, dtype=int)
        self._amp_history = [deque(maxlen=30) for _ in range(12)]
        self.logger.info("Engine state reset (debounce counters and amplitude history cleared).")

    def process(self, frame: DataFrame) -> AlgoResult:
        """
        Process one frame and return the traditional algorithm result.
        """
        t_start = time.perf_counter()

        result = AlgoResult()
        result.frame_id = frame.frame_id
        result.engine_type = "Traditional"

        # Output arrays — class_ids use unified 9-class IDs
        class_probs = np.zeros((12, N_CLASSES), dtype=np.float32)
        class_ids = np.zeros(12, dtype=np.uint8)
        valid_flags = np.zeros(12, dtype=np.float32)

        for ch in range(12):
            conf = frame.edi_confidence[ch]
            amp = frame.edi_amplitude[ch]
            dist = frame.edi_distance[ch]
            envelope = frame.envelopes[ch]

            # ── 4.2.1 Validity decision ──
            # Update noise baseline only from low-amplitude frames (avoid self-contamination)
            NOISE_UPPER = 0.15  # only frames below this are considered noise
            if amp < NOISE_UPPER:
                self._amp_history[ch].append(amp)
            noise_baseline = float(np.mean(self._amp_history[ch])) if self._amp_history[ch] else 0.01
            amp_threshold = max(noise_baseline * self.amplitude_sigma, 0.10)

            level1_valid = (conf >= self.confidence_threshold) and (amp > amp_threshold)

            if level1_valid:
                self._debounce_counters[ch] += 1
            else:
                self._debounce_counters[ch] = 0

            is_valid = self._debounce_counters[ch] >= self.debounce_frames
            valid_flags[ch] = 1.0 if is_valid else float(self._debounce_counters[ch]) / self.debounce_frames

            # ── 4.2.2 Obstacle classification (internal 5-rule labels) ──
            internal_cls = CLS_UNKNOWN
            if not is_valid or amp <= noise_baseline:
                internal_cls = CLS_OPEN
            else:
                envelope_tail = float(np.sum(envelope[20:] > 0.1))  # envelope tail length
                peak_idx = int(np.argmax(envelope))

                # Ground clutter: long tail and short distance
                if envelope_tail > 20 and dist < 0.3:
                    internal_cls = CLS_GROUND
                # Hard obstacle: high amplitude, sharp single peak
                elif amp > 0.7 and peak_idx < 200:
                    internal_cls = CLS_HARD
                # Soft/small obstacle: medium amplitude
                elif 0.3 < amp <= 0.7:
                    internal_cls = CLS_SOFT
                else:
                    internal_cls = CLS_UNKNOWN

            # Map internal 5-rule label → unified 9-class ID
            cls_id = TRAD_TO_UNIFIED[internal_cls]
            class_ids[ch] = cls_id

            # ── Confidence score: reflects actual signal quality ──
            # NOTE: branch on the INTERNAL 5-rule label, not the unified cls_id.
            # TRAD_TO_UNIFIED is many-to-one (both HARD and UNKNOWN → Wall/0),
            # so using cls_id here would misclassify HARD obstacles as "unknown"
            # and cap their confidence at 0.45 instead of the SNR-based value.
            if internal_cls == CLS_OPEN:
                # Confidence in “nothing here”: 1.0 - (amp / amp_threshold), clamped
                primary_conf = float(np.clip(1.0 - amp / max(amp_threshold, 1e-6), 0.30, 0.98))
            elif internal_cls == CLS_UNKNOWN:
                primary_conf = 0.45  # low certainty for unknown
            else:
                # SNR above threshold: how many times amp exceeds the noise floor
                snr_ratio = amp / max(amp_threshold, 1e-6)   # e.g. 0.85 / 0.10 = 8.5
                # Map [1.0 … 10.0+] → [0.50 … 0.97] with a log curve
                primary_conf = float(np.clip(0.50 + 0.47 * (1.0 - 1.0 / snr_ratio), 0.50, 0.97))
                # Also weight by debounce stability (partial credit during ramp-up)
                debounce_ratio = min(self._debounce_counters[ch] / max(self.debounce_frames, 1), 1.0)
                primary_conf *= (0.6 + 0.4 * debounce_ratio)

            # Build full 9-class probability vector
            probs = np.full(N_CLASSES, (1.0 - primary_conf) / (N_CLASSES - 1), dtype=np.float32)
            probs[cls_id] = primary_conf
            probs = np.clip(probs, 0, 1)
            probs /= probs.sum()
            class_probs[ch] = probs

        # ── 4.2.3 Elastic-wave collision decision ──
        elastic_rms = np.sqrt(np.mean(frame.elastic_features ** 2, axis=1))  # [12]
        max_rms = float(np.max(elastic_rms))
        if max_rms >= self.elastic_rms_threshold:
            collision_type = 3 if max_rms > 0.8 else (2 if max_rms > 0.6 else 1)
        else:
            collision_type = 0

        # Collision probabilities (traditional algorithm: one-hot)
        collision_probs = np.zeros(4, dtype=np.float32)
        collision_probs[collision_type] = 1.0

        result.valid_flags = valid_flags
        result.class_probs = class_probs    # (12, 9) unified 9-class probs
        result.class_ids = class_ids        # unified 9-class IDs
        result.collision_probs = collision_probs
        result.collision_type = collision_type
        result.inference_time_ms = (time.perf_counter() - t_start) * 1000.0

        return result
