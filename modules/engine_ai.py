"""
engine_ai.py  —  AI inference engine
Loads inference_core_x64.dll via ctypes, calling the same inference pipeline as the ECU side.
Corresponds to design spec section 4.3.

DLL C interface (matching ECU SDK interface provided by embedded team):
  int  ak2_infer_init(const char* model_dir);
  void ak2_infer_reset();
  int  ak2_infer_process(
           const AK2InputFrame* input,
           AK2OutputResult*     output
       );
  void ak2_infer_deinit();

When the DLL is unavailable, the engine enters "simulation mode",
producing random output data for UI development and debugging.
"""

import os
import time
import logging
import ctypes
import numpy as np
from collections import deque
from modules.data_manager import DataFrame
from modules.result_fusion import AlgoResult
from modules.config_loader import ConfigLoader
from modules.feature_extractor import extract_all_channels
from modules.obstacle_classes import N_CLASSES, WIDTH_CLASSES

try:
    import os as _os
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # suppress OpenMP dup-lib warning
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# M1 obstacle detection MLP  (must match train/train_M1.py)
# ─────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    class _ObstacleDetectionMLP(nn.Module):
        """20 → 32(BN,ReLU,Drop0.1) → 16(BN,ReLU,Drop0.1) → 1(Sigmoid)"""
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(20, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(16,  1), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.layers(x).squeeze(-1)   # (B,)

    class _ObstacleClassifier1DCNN(nn.Module):
        """M2: 1D-CNN 障碍物分类器（§6.3.1）— 必须与 train/train_M2.py 保持一致"""
        def __init__(self, num_classes: int = N_CLASSES):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1,  16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            )
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            self.classifier       = nn.Linear(32, num_classes)
            self.hardness_head    = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
            self.height_head      = nn.Sequential(nn.Linear(32, 1), nn.ReLU())

        def forward(self, x: torch.Tensor, return_features: bool = False):
            """x: [B, 1, 256] → dict, or [B, 32] feat if return_features=True"""
            x    = self.conv_layers(x)
            x    = self.global_avg_pool(x).squeeze(-1)   # [B, 64]
            feat = self.fc1(x)                           # [B, 32]
            if return_features:
                return feat
            return {
                "class_logits":      self.classifier(feat),
                "hardness":          self.hardness_head(feat),
                "suspension_height": self.height_head(feat),
            }

    class _M2GroupAttention(nn.Module):
        """M2-SA: 组内空间注意力精化模块（§6.4.2）"""

        def __init__(self, feat_dim: int = 32, num_classes: int = N_CLASSES,
                     num_positions: int = 6):
            super().__init__()
            self.pos_emb        = nn.Embedding(num_positions, feat_dim)
            self.spatial_attn   = nn.MultiheadAttention(
                embed_dim=feat_dim, num_heads=4, batch_first=True, dropout=0.1
            )
            self.spatial_norm   = nn.LayerNorm(feat_dim)
            self.temporal_score = nn.Linear(feat_dim, 1)
            self.cls_head       = nn.Linear(feat_dim, num_classes)
            self.hardness_head  = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
            self.height_head    = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())
            self.width_head     = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())

        def _temporal_fusion(self, feat_seq: torch.Tensor,
                             seq_padding_mask=None) -> torch.Tensor:
            """feat_seq: [B, T, N, 32] → [B, N, 32]

            seq_padding_mask: optional [B, T, N] bool, True = that channel was a
            padding (invalid) channel in that frame. Masked frames are excluded
            from the per-channel temporal softmax so stale features from frames
            where the sensor had no valid echo do not pollute the fusion.
            Default None preserves the original (trained) behaviour.
            """
            B, T, N, D = feat_seq.shape
            scores    = self.temporal_score(feat_seq).squeeze(-1)      # [B, T, N]
            time_bias = torch.linspace(
                0.0, 1.0, T, device=feat_seq.device
            ).view(1, T, 1)
            scores = scores + time_bias
            if seq_padding_mask is not None:
                # Large finite negative (not -inf): a channel padded in ALL T
                # frames yields a uniform softmax instead of NaN; such channels
                # are forced to Open downstream anyway.
                scores = scores.masked_fill(seq_padding_mask, -1e9)
            weights   = torch.softmax(scores, dim=1)                   # [B, T, N]
            return (feat_seq * weights.unsqueeze(-1)).sum(dim=1)        # [B, N, 32]

        def forward(self, feat: torch.Tensor, positions: torch.Tensor,
                    padding_mask=None, feat_seq=None, seq_padding_mask=None) -> dict:
            """
            feat:         [B, N, 32]
            positions:    [B, N] int  (0~5)
            padding_mask: [B, N] bool  True = padding channel (ignored in attention)
            feat_seq:     [B, T, N, 32]  multi-frame (optional)
            """
            if feat_seq is not None:
                feat = self._temporal_fusion(feat_seq, seq_padding_mask)
            x = feat + self.pos_emb(positions)
            attn_out, _ = self.spatial_attn(x, x, x, key_padding_mask=padding_mask)
            x = self.spatial_norm(x + attn_out)
            return {
                "class_logits":      self.cls_head(x),       # [B, N, 9]
                "hardness":          self.hardness_head(x),  # [B, N, 1]
                "suspension_height": self.height_head(x),    # [B, N, 1]
                "object_width":      self.width_head(x),      # [B, N, 1]
            }

    class _SceneUnderstandingCNN(nn.Module):
        """M6 v3: encoder → FC geometric remap → learned-upsampling decoder (§6.6).

        v2 mapped the [12,256] sensor×range feature map to the Cartesian OGM by
        bilinear interpolation, which could not align the two coordinate systems
        (blobs landed >1 m off). v3 inserts an FC bottleneck that learns the
        sensor-range → Cartesian remap, then upsamples with transposed convolutions
        for sharp, correctly-located occupancy.

        Input:  [B, 1, 12, 256]  (normalised 12-channel envelope matrix)
        Output: [B, 2, OGM_H, OGM_W]  per-cell probability in [0,1] (Sigmoid):
                ch0 = occupancy, ch1 = free-space.
        Must stay identical to SceneUnderstandingCNN in train/train_M6.py.
        """

        _C0, _H0, _W0 = 16, 10, 8

        def __init__(self, ogm_h: int = 160, ogm_w: int = 130):
            super().__init__()
            self.ogm_h = ogm_h
            self.ogm_w = ogm_w
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((3, 8)),
            )
            self.fc = nn.Sequential(
                nn.Linear(64 * 3 * 8, 256), nn.ReLU(inplace=True),
                nn.Linear(256, self._C0 * self._H0 * self._W0),
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(self._C0, 32, 4, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),
                nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(8, 2, 3, padding=1)   # ch0=occupancy, ch1=free

        def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
            import torch.nn.functional as F
            b = x.size(0)
            f = self.enc(x).flatten(1)
            seed = self.fc(f).view(b, self._C0, self._H0, self._W0)
            d = self.dec(seed)
            logits = self.head(d)
            logits = F.interpolate(logits, size=(self.ogm_h, self.ogm_w),
                                   mode='bilinear', align_corners=False)
            return logits if return_logits else torch.sigmoid(logits)





# ─────────────────────────────────────────────────────────────
# ctypes structures (one-to-one with ECU SDK C header structs)
# ─────────────────────────────────────────────────────────────

class AK2InputFrame(ctypes.Structure):
    """Corresponds to AK2InputFrame_t in the ECU SDK."""
    _fields_ = [
        ("frame_id",         ctypes.c_uint32),
        ("timestamp_ms",     ctypes.c_float),
        ("edi_distance",     ctypes.c_float * 12),
        ("edi_amplitude",    ctypes.c_float * 12),
        ("edi_confidence",   ctypes.c_uint8 * 12),
        ("edi_echo_type",    ctypes.c_uint8 * 12),
        ("envelopes",        (ctypes.c_float * 256) * 12),
        ("elastic_features", (ctypes.c_float * 20) * 12),
        ("vehicle_speed",    ctypes.c_float),
        ("steering_angle",   ctypes.c_float),
        ("gear",             ctypes.c_uint8),  # 0=P 1=R 2=N 3=D
    ]


class AK2OutputResult(ctypes.Structure):
    """Corresponds to AK2OutputResult_t in the ECU SDK."""
    _fields_ = [
        ("valid_flags",         ctypes.c_float * 12),
        ("class_probs",         (ctypes.c_float * N_CLASSES) * 12),  # N_CLASSES = 9, matches obstacle_classes
        ("class_ids",           ctypes.c_uint8 * 12),
        ("material_hardness",   ctypes.c_float * 12),
        ("suspension_height_m", ctypes.c_float * 12),
        ("collision_probs",     ctypes.c_float * 4),
        ("collision_type",      ctypes.c_uint8),
        ("anomaly_score",       ctypes.c_float),
    ]


# Gear string → index mapping
GEAR_MAP = {"P": 0, "R": 1, "N": 2, "D": 3}


# ─────────────────────────────────────────────────────────────
# AI engine main class
# ─────────────────────────────────────────────────────────────

class AIEngine:
    """
    AI inference engine.
    Prefers loading inference_core_x64.dll; falls back to simulation mode (random output) when DLL is unavailable.
    """

    def __init__(self, cfg: ConfigLoader):
        self.cfg = cfg
        self.logger = logging.getLogger("AIEngine")
        self._dll = None
        self._sim_mode = False  # True = simulation mode (no DLL)
        self._dll_path = cfg.get("engine.dll_path", "./lib/inference_core_x64.dll")
        self._expected_version = cfg.get("engine.dll_version", "1.2.0")

        self._pt_model = None   # M1: binary obstacle detector (fallback when DLL absent)
        self._pt_m2_model = None  # M2: 9-class obstacle type classifier (optional)
        self._pt_threshold = cfg.get("ai.m1_validity_threshold", 0.5)
        self._m1_spatial_suppress: bool = cfg.get("ai.m1_spatial_suppress", True)
        self._m1_suppress_alpha: float = cfg.get("ai.m1_suppress_alpha", 0.4)
        self._pt_model_path = cfg.get(
            "ai.m1_model_path", "./models/M1/M1_obstacle_detection_v1.0.0.pt"
        )
        self._pt_m2_model_path = cfg.get(
            "ai.m2_model_path", "./models/M2/M2_obstacle_classifier_v1.0.0.pt"
        )
        self._pt_sa_model = None   # M2-SA: spatial attention refinement (optional)
        self._pt_sa_model_path = cfg.get(
            "ai.m2_sa_model_path", "./models/M2/M2_spatial_attention_v1.0.0.pt"
        )
        self._temporal_frames: int = cfg.get("ai.temporal_frames", 1)
        self._feat_buf_front: deque = deque(maxlen=8)  # front-group temporal feat buffer
        self._feat_buf_rear:  deque = deque(maxlen=8)  # rear-group  temporal feat buffer
        self._pad_buf_front:  deque = deque(maxlen=8)  # front-group per-frame padding mask
        self._pad_buf_rear:   deque = deque(maxlen=8)  # rear-group  per-frame padding mask
        self._pt_m6_model = None   # M6: scene understanding 2D-CNN OGM (optional)
        self._pt_m6_model_path = cfg.get(
            "ai.m6_model_path", "./models/M6/M6_scene_understanding_v3.0.0.pt"
        )
        self._m6_enabled: bool = cfg.get("ai.m6_enabled", True)
        # OGM grid dimensions from display config (must match training)
        _res = float(cfg.get("display.ogm_resolution_m", 0.1))
        self._ogm_h = int(round(2 * float(cfg.get("display.ogm_range_front_m", 8.0)) / _res))
        self._ogm_w = int(round(2 * float(cfg.get("display.ogm_range_side_m",  6.5)) / _res))

        # Per-sensor mount height (z_m) and pitch angle (pitch_deg), used to
        # convert sensor-referenced overhead clearance into ground clearance.
        _mount_z, _mount_pitch = [], []
        for ch in range(1, 13):
            sc = cfg.get_sensor_config(f"S{ch:02d}")
            _mount_z.append(float(sc.get("z_m", 0.5)))
            _mount_pitch.append(float(sc.get("pitch_deg", 0.0)))
        self._mount_z = np.asarray(_mount_z, dtype=np.float32)            # (12,)
        self._mount_pitch_rad = np.radians(_mount_pitch).astype(np.float32)  # (12,)
        self.logger.info("AIEngine init: DLL=%s | M1=%s | M2=%s | M2-SA=%s | M6=%s | temporalFrames=%d",
                          self._dll_path, self._pt_model_path, self._pt_m2_model_path,
                          self._pt_sa_model_path, self._pt_m6_model_path, self._temporal_frames)
        self._load_dll()

    # ── DLL loading ──────────────────────────────

    def _load_dll(self) -> None:
        """Try DLL → PyTorch M1 fallback → simulation mode (random)."""
        dll_abs = os.path.abspath(self._dll_path)
        if not os.path.exists(dll_abs):
            self.logger.warning("DLL not found: %s", dll_abs)
            if self._load_pytorch_model():
                return  # PyTorch fallback active
            self.logger.warning("No model found. Switched to simulation mode (random output).")
            self._sim_mode = True
            return

        try:
            self._dll = ctypes.CDLL(dll_abs)
            self._bind_functions()
            ret = self._dll.ak2_infer_init(b"./lib")
            if ret != 0:
                raise RuntimeError(f"ak2_infer_init returned error code: {ret}")
            self.logger.info("DLL loaded and initialized: %s", dll_abs)
            self._verify_version()
            self._load_pytorch_m6()  # M6 is pure-PyTorch, always load alongside DLL
        except Exception as e:
            self.logger.error("DLL load failed: %s", e)
            if self._load_pytorch_model():
                return  # PyTorch fallback active
            self.logger.warning("Switched to simulation mode.")
            self._dll = None
            self._sim_mode = True

    def _load_pytorch_model(self) -> bool:
        """Load M1 PyTorch weights as fallback. Returns True if successful."""
        if not _TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available — cannot use M1 model fallback.")
            return False
        pt_abs = os.path.abspath(self._pt_model_path)
        if not os.path.exists(pt_abs):
            self.logger.warning("M1 model file not found: %s", pt_abs)
            return False
        try:
            model = _ObstacleDetectionMLP()
            state = torch.load(pt_abs, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._pt_model = model
            self.logger.info("M1 PyTorch model loaded: %s (threshold=%.2f)", pt_abs, self._pt_threshold)
            # Try loading M2 alongside M1
            self._load_pytorch_m2()
            return True
        except Exception as e:
            self.logger.error("M1 model load failed: %s", e)
            return False

    def _load_pytorch_m2(self) -> None:
        """Load M2 9-class classifier. Optional — silently skipped if model file absent."""
        if not _TORCH_AVAILABLE:
            return
        m2_abs = os.path.abspath(self._pt_m2_model_path)
        if not os.path.exists(m2_abs):
            self.logger.info("M2 model not found, skipping: %s", m2_abs)
            return   # M2 not trained yet — stay with M1 binary output
        try:
            model = _ObstacleClassifier1DCNN(num_classes=N_CLASSES)
            state = torch.load(m2_abs, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._pt_m2_model = model
            self.logger.info("M2 1D-CNN model loaded: %s", m2_abs)
            self._load_pytorch_m2_sa()
        except Exception as e:
            self.logger.error("M2 model load failed (continuing with M1 only): %s", e)

    def _load_pytorch_m2_sa(self) -> None:
        """Load M2-SA spatial attention weights. Optional — silently skipped if file absent."""
        if not _TORCH_AVAILABLE:
            return
        sa_abs = os.path.abspath(self._pt_sa_model_path)
        if not os.path.exists(sa_abs):
            self.logger.info("M2-SA model not found, skipping: %s", sa_abs)
            return  # M2-SA not yet trained
        try:
            model = _M2GroupAttention(feat_dim=32, num_classes=N_CLASSES, num_positions=6)
            state = torch.load(sa_abs, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._pt_sa_model = model
            self.logger.info("M2-SA spatial attention model loaded: %s", sa_abs)
        except Exception as e:
            self.logger.error("M2-SA model load failed (SA disabled): %s", e)
        self._load_pytorch_m6()

    def _load_pytorch_m6(self) -> None:
        """Load M6 scene understanding 2D-CNN. Optional — silently skipped if file absent."""
        if not _TORCH_AVAILABLE or not self._m6_enabled:
            if not self._m6_enabled:
                self.logger.info("M6 disabled by config.")
            return
        m6_abs = os.path.abspath(self._pt_m6_model_path)
        if not os.path.exists(m6_abs):
            self.logger.info("M6 model not found, skipping: %s", m6_abs)
            return  # M6 not yet trained
        try:
            model = _SceneUnderstandingCNN(ogm_h=self._ogm_h, ogm_w=self._ogm_w)
            state = torch.load(m6_abs, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            self._pt_m6_model = model
            self.logger.info("M6 loaded: %s (OGM %dx%d)", m6_abs, self._ogm_h, self._ogm_w)
        except Exception as e:
            self.logger.error(
                "M6 model load failed (OGM will remain zero): %s\n"
                "Checkpoint may be stale — re-run train/train_M6.py.", e
            )

    def _bind_functions(self) -> None:
        """Bind DLL function signatures."""
        dll = self._dll
        dll.ak2_infer_init.argtypes = [ctypes.c_char_p]
        dll.ak2_infer_init.restype = ctypes.c_int
        dll.ak2_infer_reset.argtypes = []
        dll.ak2_infer_reset.restype = None
        dll.ak2_infer_process.argtypes = [
            ctypes.POINTER(AK2InputFrame),
            ctypes.POINTER(AK2OutputResult)
        ]
        dll.ak2_infer_process.restype = ctypes.c_int
        dll.ak2_infer_deinit.argtypes = []
        dll.ak2_infer_deinit.restype = None

    def _verify_version(self) -> None:
        """Check if the DLL version file matches the expected version."""
        ver_path = os.path.join(os.path.dirname(self._dll_path), "inference_core_version.txt")
        if not os.path.exists(ver_path):
            self.logger.warning("DLL version file not found: %s", ver_path)
            return
        with open(ver_path, "r", encoding="utf-8") as f:
            actual_ver = f.readline().strip()
        if actual_ver != self._expected_version:
            self.logger.warning("DLL version mismatch! Expected %s, got %s",
                                self._expected_version, actual_ver)
        else:
            self.logger.info("DLL version verified: %s", actual_ver)

    def reset(self) -> None:
        """Reset engine internal state (called when switching sessions)."""
        if self._dll:
            self._dll.ak2_infer_reset()
            self.logger.info("DLL ak2_infer_reset() called.")
        self._feat_buf_front.clear()
        self._feat_buf_rear.clear()
        self._pad_buf_front.clear()
        self._pad_buf_rear.clear()
        self.logger.info("Engine state reset (temporal buffers cleared).")

    def shutdown(self) -> None:
        """Release DLL resources."""
        if self._dll:
            self.logger.info("Calling ak2_infer_deinit() ...")
            self._dll.ak2_infer_deinit()
            self.logger.info("DLL deinitialized.")
            self._dll = None

    # ── Frame inference ──────────────────────────────

    def process(self, frame: DataFrame) -> AlgoResult:
        """Process one frame and return the AI inference result."""
        t_start = time.perf_counter()

        if self._pt_model is not None:
            result = self._infer_pytorch(frame)
        elif self._sim_mode:
            result = self._simulate(frame)
        else:
            result = self._infer_dll(frame)

        # ── M6: Scene Understanding OGM (§6.6) — runs for all inference paths ──
        if self._pt_m6_model is not None:
            # Absolute/fixed-reference + log compression (precision-first M6, §6.5.3)
            # MUST match normalize_envelopes() in train/train_M6.py. Envelopes are in
            # absolute [0,1] units (ADC full-scale=1.0); dividing by a FIXED reference
            # (not the per-frame max) preserves absolute amplitude, and log compression
            # lifts sparse weak echoes while staying monotonic.
            _ENV_FULL_SCALE, _LOG_K = 1.0, 50.0
            env_np = frame.envelopes.astype(np.float32).copy()  # [12, 256]
            env_np = np.clip(env_np / _ENV_FULL_SCALE, 0.0, 1.0)
            env_np = (np.log1p(_LOG_K * env_np) / np.log1p(_LOG_K)).astype(np.float32)
            x_m6 = torch.from_numpy(env_np).unsqueeze(0).unsqueeze(0)  # [1,1,12,256]
            with torch.no_grad():
                ogm_t = self._pt_m6_model(x_m6)   # [1,2,H,W] (ch0=occ, ch1=free)
            ogm_np = ogm_t.squeeze(0).numpy().astype(np.float32)  # [2,H,W]
            result.ogm_grid = ogm_np[0]   # occupancy [H,W]
            result.ogm_free = ogm_np[1]   # free-space [H,W]
            if "M6" not in result.engine_type:
                result.engine_type = result.engine_type.rstrip(")") + "+M6)"

        result.inference_time_ms = (time.perf_counter() - t_start) * 1000.0
        return result

    def _infer_dll(self, frame: DataFrame) -> AlgoResult:
        """Invoke the DLL to run inference."""
        input_buf = AK2InputFrame()
        output_buf = AK2OutputResult()

        # Fill input struct
        input_buf.frame_id = frame.frame_id
        input_buf.timestamp_ms = frame.timestamp_ms
        for i in range(12):
            input_buf.edi_distance[i] = float(frame.edi_distance[i])
            input_buf.edi_amplitude[i] = float(frame.edi_amplitude[i])
            input_buf.edi_confidence[i] = int(frame.edi_confidence[i])
            input_buf.edi_echo_type[i] = int(frame.edi_echo_type[i])
            for j in range(256):
                input_buf.envelopes[i][j] = float(frame.envelopes[i, j])
            for j in range(20):
                input_buf.elastic_features[i][j] = float(frame.elastic_features[i, j])
        input_buf.vehicle_speed = frame.vehicle_speed
        input_buf.steering_angle = frame.steering_angle
        input_buf.gear = GEAR_MAP.get(frame.gear, 0)

        ret = self._dll.ak2_infer_process(
            ctypes.byref(input_buf), ctypes.byref(output_buf)
        )
        if ret != 0:
            self.logger.error("ak2_infer_process error code %d, frame_id=%d", ret, frame.frame_id)

        return self._parse_output(frame.frame_id, output_buf)

    def _parse_output(self, frame_id: int, out: AK2OutputResult) -> AlgoResult:
        """Convert DLL output structure into an AlgoResult."""
        result = AlgoResult()
        result.frame_id = frame_id
        result.engine_type = "AI"
        result.valid_flags = np.array(list(out.valid_flags), dtype=np.float32)
        result.class_probs = np.array(
            [[out.class_probs[i][j] for j in range(9)] for i in range(12)],
            dtype=np.float32
        )
        result.class_ids = np.array(list(out.class_ids), dtype=np.uint8)
        result.material_hardness = np.array(list(out.material_hardness), dtype=np.float32)
        result.suspension_height_m = np.array(list(out.suspension_height_m), dtype=np.float32)
        result.collision_probs = np.array(list(out.collision_probs), dtype=np.float32)
        result.collision_type = int(out.collision_type)
        result.anomaly_score = float(out.anomaly_score)
        return result

    def _suppress_isolated(self, valid_probs: np.ndarray, thr: float) -> np.ndarray:
        """
        Spatial-consistency post-filter (rule-based, no ML).

        A channel that fires above threshold while BOTH of its in-row neighbours
        stay below threshold is likely a single-channel false alarm (sidelobe /
        multipath). Dampen its probability by alpha to reduce isolated false
        positives. Front row = ch0-5, rear row = ch6-11 (no cross-row adjacency).
        alpha=0 disables suppression; alpha=1 fully suppresses isolated channels.
        """
        alpha = float(self._m1_suppress_alpha)
        if alpha <= 0.0:
            return valid_probs
        out = valid_probs.copy()
        for grp_start in (0, 6):
            for i in range(6):
                ch = grp_start + i
                if valid_probs[ch] < thr:
                    continue
                left_ok  = i > 0 and valid_probs[ch - 1] >= thr
                right_ok = i < 5 and valid_probs[ch + 1] >= thr
                if not left_ok and not right_ok:
                    out[ch] = valid_probs[ch] * (1.0 - alpha)
        return out

    def _infer_pytorch(self, frame: DataFrame) -> AlgoResult:
        """
        Direct PyTorch M1 inference (fallback when DLL absent).
        Runs 12 channels through the shared MLP in one batched forward pass.
        M1 is a binary obstacle detector: high prob → obstacle (Wall), low prob → free (Open).
        Class IDs follow the 9-class GT scheme: 0=Wall, 4=Open/Clear.
        """
        # Compute features on-the-fly from envelopes + EDI (same as training)
        x_np = extract_all_channels(
            frame.edi_distance, frame.edi_amplitude,
            frame.edi_confidence, frame.edi_echo_type,
            frame.envelopes,
        )  # (12, 20)
        x = torch.from_numpy(x_np)

        with torch.no_grad():
            valid_probs = self._pt_model(x).numpy()   # (12,)  range [0, 1]

        thr = self._pt_threshold
        if self._m1_spatial_suppress:
            valid_probs = self._suppress_isolated(valid_probs, thr)
        result = AlgoResult()
        result.frame_id    = frame.frame_id

        # ── M2 cascade: 1D-CNN 9-class type classification (§6.3.1) ─────────
        if self._pt_m2_model is not None:
            result.engine_type = "AI(M1+M2)"
            # M2 input: raw envelopes [12, 1, 256]
            x_env = torch.from_numpy(frame.envelopes.astype(np.float32)).unsqueeze(1)
            with torch.no_grad():
                m2_out = self._pt_m2_model(x_env)   # dict
            logits = m2_out["class_logits"].numpy()  # (12, 9)
            exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs9 = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)
            # If M1 says free (prob < thr), force class to Open(4) regardless of M2
            m2_ids    = probs9.argmax(axis=1).astype(np.uint8)
            class_ids = np.where(valid_probs >= thr, m2_ids, np.uint8(4))
            # Blend: for free channels, move all probability mass to Open slot
            for ch in range(12):
                if valid_probs[ch] < thr:
                    probs9[ch, :] = 0.0
                    probs9[ch, 4] = 1.0
            # M2 auxiliary outputs
            m2_hardness = m2_out["hardness"].numpy().squeeze(-1).astype(np.float32)    # (12,)
            m2_height   = m2_out["suspension_height"].numpy().squeeze(-1).astype(np.float32)  # (12,)
        else:
            # M1 binary fallback (no M2): Wall or Open only
            result.engine_type = "AI(M1)"
            probs9 = np.zeros((12, N_CLASSES), dtype=np.float32)
            probs9[:, 0] = valid_probs          # Wall  probability
            probs9[:, 4] = 1.0 - valid_probs    # Open  probability
            class_ids = np.where(valid_probs >= thr, 0, 4).astype(np.uint8)
        # ──────────────────────────────────────────────────────────────────

        # valid_flags: raw M1 obstacle probability
        result.valid_flags = valid_probs.astype(np.float32)

        # ── Persist SC (backbone) output before any SA modification ──────────
        result.sc_class_probs = probs9.copy()
        result.sc_class_ids   = class_ids.copy()

        # ── M2-SA: group spatial attention (§6.4.2) ──────────────────────────
        sa_enabled = self.cfg.get("ai.m2_sa_enabled", False)
        _sa_refined_aux = False  # True when SA provides hardness/height directly
        if sa_enabled and self._pt_m2_model is not None:
            # Extract M2-SC intermediate features [12, 32] (second forward pass)
            with torch.no_grad():
                feats12 = self._pt_m2_model(x_env, return_features=True)  # [12, 32]
            self._feat_buf_front.append(feats12[:6].unsqueeze(0))   # [1, 6, 32]
            self._feat_buf_rear.append(feats12[6:].unsqueeze(0))    # [1, 6, 32]
            # Per-frame padding mask (True = invalid channel) kept in lockstep
            # with the feature buffers, for temporal-fusion masking when T>1.
            pad_all = torch.tensor(valid_probs < thr, dtype=torch.bool)
            self._pad_buf_front.append(pad_all[:6].unsqueeze(0))    # [1, 6]
            self._pad_buf_rear.append(pad_all[6:].unsqueeze(0))     # [1, 6]

            sa_probs    = probs9.copy()
            sa_hardness = m2_hardness.copy()
            sa_height   = m2_height.copy()
            sa_width    = np.full(12, -1.0, dtype=np.float32)
            T = self._temporal_frames

            if self._pt_sa_model is not None:
                # Real SA model inference
                result.engine_type = "AI(M1+M2+SA)"
                positions = torch.arange(6, dtype=torch.long).unsqueeze(0)  # [1, 6]
                for buf, pad_buf, grp_start in (
                    (self._feat_buf_front, self._pad_buf_front, 0),
                    (self._feat_buf_rear,  self._pad_buf_rear,  6),
                ):
                    grp_end  = grp_start + 6
                    pad_mask = torch.tensor(
                        valid_probs[grp_start:grp_end] < thr, dtype=torch.bool
                    ).unsqueeze(0)                             # [1, 6] True=ignore
                    # When ALL channels are padding, attention softmax is undefined.
                    # Skip SA for this group and keep SC outputs as-is.
                    if pad_mask.all():
                        continue

                    if T > 1 and len(buf) >= 2:
                        t_frames = min(T, len(buf))
                        # Stack frames: [T', 6, 32] → unsqueeze → [1, T', 6, 32]
                        feat_seq = torch.cat(
                            list(buf)[-t_frames:], dim=0
                        ).unsqueeze(0)
                        # Matching per-frame padding mask: [1, T', 6]
                        seq_pad = torch.cat(
                            list(pad_buf)[-t_frames:], dim=0
                        ).unsqueeze(0)
                        cur_feat = feat_seq[:, -1, :, :]       # [1, 6, 32]
                        with torch.no_grad():
                            sa_out = self._pt_sa_model(
                                cur_feat, positions,
                                padding_mask=pad_mask, feat_seq=feat_seq,
                                seq_padding_mask=seq_pad,
                            )
                    else:
                        cur_feat = list(buf)[-1]               # [1, 6, 32]
                        with torch.no_grad():
                            sa_out = self._pt_sa_model(
                                cur_feat, positions, padding_mask=pad_mask,
                            )
                    logits_sa = sa_out["class_logits"].squeeze(0).numpy()  # [6, 9]
                    exp_l     = np.exp(logits_sa - logits_sa.max(axis=1, keepdims=True))
                    sa_probs[grp_start:grp_end] = (
                        exp_l / exp_l.sum(axis=1, keepdims=True)
                    ).astype(np.float32)
                    sa_hardness[grp_start:grp_end] = (
                        sa_out["hardness"].squeeze(0).squeeze(-1).numpy()
                    )
                    sa_height[grp_start:grp_end] = (
                        sa_out["suspension_height"].squeeze(0).squeeze(-1).numpy()
                    )
                    sa_width[grp_start:grp_end] = (
                        sa_out["object_width"].squeeze(0).squeeze(-1).numpy()
                    )
                _sa_refined_aux = True
            else:
                # SA model not yet trained — group-mean approximation
                result.engine_type = "AI(M1+M2+SA~)"
                for grp_start in (0, 6):
                    grp = sa_probs[grp_start:grp_start + 6]
                    ctx = grp.mean(axis=0)
                    sa_probs[grp_start:grp_start + 6] = 0.80 * grp + 0.20 * ctx
                row_sums = sa_probs.sum(axis=1, keepdims=True)
                sa_probs /= np.where(row_sums > 0, row_sums, 1.0)

            # Free channels always map to Open regardless of SA output
            for ch in range(12):
                if valid_probs[ch] < thr:
                    sa_probs[ch, :] = 0.0
                    sa_probs[ch, 4] = 1.0
            sa_ids = sa_probs.argmax(axis=1).astype(np.uint8)
            sa_ids = np.where(valid_probs >= thr, sa_ids, np.uint8(4))
            result.class_probs = sa_probs.astype(np.float32)
            result.class_ids   = sa_ids
            if _sa_refined_aux:
                result.material_hardness   = sa_hardness.astype(np.float32)
                result.suspension_height_m = np.where(
                    sa_ids == 6, self._ground_clearance(sa_height), np.full(12, -1.0)
                ).astype(np.float32)
        else:
            result.class_ids   = class_ids
            result.class_probs = probs9

        if not _sa_refined_aux:
            if self._pt_m2_model is not None:
                result.material_hardness   = m2_hardness
                result.suspension_height_m = np.where(
                    class_ids == 6, self._ground_clearance(m2_height),
                    np.full(12, -1.0, dtype=np.float32)
                ).astype(np.float32)
            else:
                result.material_hardness   = np.full(12, 0.0, dtype=np.float32)
                result.suspension_height_m = np.full(12, -1.0, dtype=np.float32)

        # Collision: simple heuristic from valid obstacle count
        n_valid = int((valid_probs >= thr).sum())
        if   n_valid >= 6: result.collision_type = 3
        elif n_valid >= 4: result.collision_type = 2
        elif n_valid >= 2: result.collision_type = 1
        else:              result.collision_type = 0
        cp = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        cp[result.collision_type] += 0.5
        result.collision_probs = cp / cp.sum()

        result.anomaly_score = float(np.clip(valid_probs.max(), 0, 1))

        # ── Object width ───────────────────────────────────────────────────
        # Prefer the trained M2-SA width head when available; otherwise fall
        # back to the echo-half-width heuristic. Only object classes (those in
        # WIDTH_CLASSES) receive a value; others stay at -1.
        if _sa_refined_aux:
            in_width = np.isin(result.class_ids, list(WIDTH_CLASSES))
            result.object_width_m = np.where(
                in_width, sa_width, np.full(12, -1.0, dtype=np.float32)
            ).astype(np.float32)
        else:
            result.object_width_m = self._estimate_object_width(
                x_np, valid_probs, result.class_ids, thr
            )

        return result

    def _estimate_object_width(
        self,
        feats: np.ndarray,      # (12, 20) per-channel feature matrix
        valid_probs: np.ndarray,
        class_ids: np.ndarray,
        thr: float,
    ) -> np.ndarray:
        """
        Heuristic lateral-width estimate (metres) from the echo half-width
        feature. This is an approximation, NOT a trained model output; replace
        with a dedicated width head once width-annotated data is available.
        """
        half_w_norm = feats[:, 9].astype(np.float32)            # (12,) ~0..0.3
        gain   = float(self.cfg.get("ai.width_estimate_gain_m", 6.5))
        w_min  = float(self.cfg.get("ai.width_estimate_min_m", 0.10))
        w_max  = float(self.cfg.get("ai.width_estimate_max_m", 2.50))
        w_raw  = np.clip(half_w_norm * gain, w_min, w_max).astype(np.float32)
        mask   = (valid_probs >= thr) & (class_ids != 4)
        return np.where(mask, w_raw, np.full(12, -1.0, dtype=np.float32)).astype(np.float32)

    def _ground_clearance(self, raw_height: np.ndarray) -> np.ndarray:
        """
        Convert the M2 height head's sensor-referenced overhead clearance into a
        ground-referenced clearance using each sensor's mount height (z_m) and
        pitch angle (pitch_deg).

        The height head predicts the vertical gap between the overhead object's
        lower edge and the sensor mounting plane (along the beam axis). The
        ground clearance is therefore::

            clearance = z_m + raw_height * cos(pitch_rad)

        With the default horizontal mount (pitch_deg = 0) this reduces to a pure
        z_m offset, so existing model outputs simply shift up by the mount
        height. A nose-up pitch projects the slant measurement onto the vertical
        axis via cos(theta).
        """
        raw = np.asarray(raw_height, dtype=np.float32)
        return (self._mount_z + raw * np.cos(self._mount_pitch_rad)).astype(np.float32)

    def _simulate(self, frame: DataFrame) -> AlgoResult:
        """
        Simulation mode: generate random inference results for UI debugging when DLL is absent.
        Uses simple heuristics based on frame amplitude to make outputs appear reasonable.
        """
        rng = np.random.default_rng(frame.frame_id)
        result = AlgoResult()
        result.frame_id = frame.frame_id
        result.engine_type = "AI(Sim)"

        # Validity: higher amplitude → higher validity
        amp_norm = np.clip(frame.edi_amplitude, 0, 1)
        result.valid_flags = np.clip(amp_norm + rng.normal(0, 0.1, 12), 0, 1).astype(np.float32)

        # 9-class probabilities (random softmax)
        logits = rng.normal(0, 1, (12, 9)).astype(np.float32)
        # Give high-amplitude channels more weight towards class 0 (wall)
        logits[:, 0] += (amp_norm * 2).astype(np.float32)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        result.class_probs = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)
        result.class_ids = result.class_probs.argmax(axis=1).astype(np.uint8)

        result.material_hardness = rng.uniform(0.3, 1.0, 12).astype(np.float32)
        result.suspension_height_m = np.full(12, -1.0, dtype=np.float32)

        # SC fields mirror main output in simulation mode (SA not simulated separately)
        result.sc_class_probs = result.class_probs.copy()
        result.sc_class_ids   = result.class_ids.copy()

        # Collision decision
        elastic_rms = float(np.sqrt(np.mean(frame.elastic_features ** 2)))
        if elastic_rms > 0.7:
            result.collision_type = 3
        elif elastic_rms > 0.5:
            result.collision_type = 2
        elif elastic_rms > 0.3:
            result.collision_type = 1
        else:
            result.collision_type = 0
        cp = rng.dirichlet(np.ones(4) * 0.5).astype(np.float32)
        cp[result.collision_type] = max(cp[result.collision_type], 0.6)
        cp /= cp.sum()
        result.collision_probs = cp

        result.anomaly_score = float(np.clip(elastic_rms + rng.normal(0, 0.05), 0, 1))

        # M6 not simulated — ogm_grid stays at default zeros from AlgoResult dataclass
        return result

    @property
    def is_sim_mode(self) -> bool:
        """True only when running pure random simulation (no DLL and no .pt model)."""
        return self._sim_mode
