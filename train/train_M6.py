"""
train_M6.py  —  M6 Scene Understanding 2D-CNN Training Script
==============================================================

M6 is a lightweight encoder-decoder CNN that converts the full 12-channel
ultrasonic envelope matrix [12, 256] into a probabilistic Occupancy Grid Map
(OGM) [160, 130] at 0.1 m/cell resolution (±8 m front/rear, ±6.5 m side).

Architecture v2 (~60 K trainable parameters)  —  U-Net with skip connections:
  Encoder:
    enc1: Conv2d(1→16, 3×3, p=1) + BN + ReLU + MaxPool2d(2,2)   [B,16,6,128]
    enc2: Conv2d(16→32, 3×3, p=1) + BN + ReLU + MaxPool2d(2,2)  [B,32,3,64]
    enc3: Conv2d(32→64, 3×3, p=1) + BN + ReLU                   [B,64,3,64]
          AdaptiveAvgPool2d(3,8) → bottleneck                    [B,64,3,8]
            ↑ (3 rows not 1 — preserves front/mid/rear sensor groups)
  Decoder (bilinear upsample + U-Net skip concat + conv):
    dec1: interp(bn→e2.size) cat(e2) → Conv2d(96→32)+BN+ReLU    [B,32,3,64]
    dec2: interp(d1→e1.size) cat(e1) → Conv2d(48→16)+BN+ReLU    [B,16,6,128]
    dec3: interp→[160,130]           + Conv2d(16→1)+Sigmoid  →  [B,1,160,130]

OGM Ground-Truth synthesis:
  Per channel: if obstacle class ≠ Open(4) / Clutter(5):
    dist = argmax(envelope) / 255 * 6.0  (m)
    obs_x = -(sensor_y + dist × sin(yaw))
    obs_y =   sensor_x + dist × cos(yaw)
    Paint Gaussian blob (σ = 1.5 cells) at grid cell (row, col).

Usage:
    python -m train.train_M6

Output:
    models/M6/M6_scene_understanding_v2.0.0.pt
    datasets/splits/M6/train.txt
    datasets/splits/M6/val.txt
    datasets/splits/M6/test.txt
"""

import os
import math
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))   # allow `from modules...` when run as a script
ENV_DIR = ROOT / "datasets" / "processed" / "envelopes"
SPLIT_M2_DIR = ROOT / "datasets" / "splits" / "M2"
SPLIT_M6_DIR = ROOT / "datasets" / "splits" / "M6"
MODEL_DIR = ROOT / "models" / "M6"
MODEL_PATH = MODEL_DIR / "M6_scene_understanding_v3.0.0.pt"

# Playback session data (TestData/*) — the SAME distribution the simulator/engine
# feeds to M6 at inference. Training on this avoids the train/inference domain gap
# that made M6 mislocalize obstacles on played-back sessions (gen_mock_data shapes)
# despite scoring perfectly on gen_training_data-derived envelopes.
TESTDATA_DIR = ROOT / "TestData"

# OGM parameters (must match config.yaml display section)
OGM_RANGE_FRONT = 8.0   # m
OGM_RANGE_SIDE  = 6.5   # m
OGM_RES         = 0.1   # m/cell
OGM_H = int(round(2 * OGM_RANGE_FRONT / OGM_RES))  # 160
OGM_W = int(round(2 * OGM_RANGE_SIDE  / OGM_RES))  # 130
ENVELOPE_LEN  = 256
DIST_MAX_M    = 6.0   # envelope x-axis max (256 samples → 0–6 m)
N_SENSORS     = 12
BLOB_SIGMA    = 1.0   # Gaussian blob sigma in OGM cells (tighter → sharper GT)

# ── Input normalization (precision-first M6, §6.5.3) ──────────────────────────
# The previous per-frame global-max norm destroyed the ABSOLUTE amplitude that
# distinguishes a real echo from noise (a weak frame with max≈0.05 was blown up to
# 1.0). Envelopes are already in absolute [0,1] units (ADC full-scale = 1.0), so we
# divide by a FIXED reference (not the per-frame max) to keep absolute amplitude
# consistent across frames/channels, then apply log compression to lift the sparse
# weak echoes (mean≈0.01) while staying monotonic and INT8-friendly.
# MUST stay identical to the M6 normalization in modules/engine_ai.py.
ENV_FULL_SCALE = 1.0    # ADC full-scale / calibration constant (fixed reference)
LOG_COMPRESS_K = 50.0   # log-compression strength (higher → more low-amplitude lift)
_LOG_DENOM     = float(np.log1p(LOG_COMPRESS_K))


def normalize_envelopes(env: np.ndarray) -> np.ndarray:
    """Absolute fixed-reference + log compression. MUST match engine_ai.py.

    env: [12, 256] raw envelope amplitudes (absolute units).
    Returns float32 in [0,1], preserving inter-channel/inter-frame absolute scale.
    """
    x = np.clip(env.astype(np.float32) / ENV_FULL_SCALE, 0.0, 1.0)
    return (np.log1p(LOG_COMPRESS_K * x) / _LOG_DENOM).astype(np.float32)


# ── 2-channel OGM (v3) ────────────────────────────────────────────────────────
# Channel 0 = occupancy (障碍占用),  Channel 1 = free-space (自由可通行空间).
# Free-space is the inverse-sensor-model cone swept from each sensor up to the
# detected obstacle (or to a default clear range when the channel sees nothing).
# This is information a per-channel classifier (M2) fundamentally cannot output.
OGM_CH              = 2
BEAM_HALF_ANGLE_DEG = 7.0    # USS half beam-width (occupancy lateral extent)
FREE_HALF_ANGLE_DEG = 6.0    # free cone half-width (narrower → less cross-channel bleed)
FREE_MARGIN_M       = 0.4    # stop free-space this far in front of the obstacle
FREE_RANGE_EMPTY_M  = 3.0    # clear-channel free reach (no echo → free up to here)
OCCLUSION_BUFFER_M  = 0.4    # carve a free-clear ring of this radius around occupancy
SHADOW_TOL_DEG      = 7.0    # angular width of the "unknown" shadow behind an obstacle

# Obstacle class IDs that produce a real OGM hit
VALID_OBJ_CLASSES = {0, 1, 2, 3, 6, 7, 8}  # Wall,Vehicle,Ped,Soft,Overhead,Curb,Wet

# Sensor mount config (vehicle frame: +X fwd, +Y left; yaw CCW from +X)
# Ordered ch0–ch11: S01-S12
SENSOR_CFG = [
    {"x_m":  1.20, "y_m":  0.95, "yaw_deg":  90.0},  # S01 FL-Side
    {"x_m":  2.25, "y_m":  0.85, "yaw_deg":  45.0},  # S02 FL-Corner
    {"x_m":  2.25, "y_m":  0.30, "yaw_deg":  10.0},  # S03 FL-Center
    {"x_m":  2.25, "y_m": -0.30, "yaw_deg": -10.0},  # S04 FR-Center
    {"x_m":  2.25, "y_m": -0.85, "yaw_deg": -45.0},  # S05 FR-Corner
    {"x_m":  1.20, "y_m": -0.95, "yaw_deg": -90.0},  # S06 FR-Side
    {"x_m": -0.60, "y_m":  0.95, "yaw_deg":  90.0},  # S07 RL-Side
    {"x_m": -2.25, "y_m":  0.85, "yaw_deg": 135.0},  # S08 RL-Corner
    {"x_m": -2.25, "y_m":  0.30, "yaw_deg": 170.0},  # S09 RL-Center
    {"x_m": -2.25, "y_m": -0.30, "yaw_deg":-170.0},  # S10 RR-Center
    {"x_m": -2.25, "y_m": -0.85, "yaw_deg":-135.0},  # S11 RR-Corner
    {"x_m": -0.60, "y_m": -0.95, "yaw_deg": -90.0},  # S12 RR-Side
]

# Per-cell bearing/range from the vehicle origin (precomputed once) — used to cast
# the "unknown" occlusion shadow behind each detected obstacle in the free channel.
_cc, _rr = np.meshgrid(np.arange(OGM_W), np.arange(OGM_H))
_CELL_X = ((_cc + 0.5) * OGM_RES - OGM_RANGE_SIDE).astype(np.float32)   # lateral
_CELL_Y = (OGM_RANGE_FRONT - (_rr + 0.5) * OGM_RES).astype(np.float32)  # longitudinal
_CELL_RANGE   = np.hypot(_CELL_X, _CELL_Y).astype(np.float32)
_CELL_BEARING = np.arctan2(_CELL_X, _CELL_Y).astype(np.float32)         # 0 = forward


# ─────────────────────────────────────────────────────────────────────────────
# Model definition  (identical to _SceneUnderstandingCNN in engine_ai.py)
# ─────────────────────────────────────────────────────────────────────────────

class SceneUnderstandingCNN(nn.Module):
    """M6 v3: encoder → FC geometric remap → learned-upsampling decoder.

    Why v3 (vs the v2 U-Net): v2 mapped the [12, 256] sensor×range feature map to
    the [H, W] Cartesian grid purely by bilinear interpolation. Sensor-range space
    and Cartesian space are NOT aligned, so interpolation could not place occupancy
    at the correct (x, y) — producing diffuse blobs offset >1 m from the true
    obstacle (confirmed empirically). v3 inserts a fully-connected bottleneck that
    learns the sensor-range → Cartesian remap explicitly, then upsamples with
    transposed convolutions so blobs are both correctly located and sharp.

    Input:  [B, 1, 12, 256]  (normalised 12-channel envelope matrix)
    Output: [B, 2, OGM_H, OGM_W]  per-cell probability in [0,1] (Sigmoid):
            ch0 = occupancy, ch1 = free-space,
            or raw logits when forward(..., return_logits=True).
    Must stay identical to _SceneUnderstandingCNN in modules/engine_ai.py.
    """

    # Coarse Cartesian seed grid produced by the FC bottleneck.
    _C0, _H0, _W0 = 16, 10, 8

    def __init__(self, ogm_h: int = OGM_H, ogm_w: int = OGM_W):
        super().__init__()
        self.ogm_h = ogm_h
        self.ogm_w = ogm_w

        # Encoder over sensor-range space
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # → [16, 6,128]
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # → [32, 3, 64]
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 8)),                         # → [64, 3,  8]
        )

        # FC geometric remap: sensor-range features → coarse Cartesian seed grid
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, self._C0 * self._H0 * self._W0),
        )

        # Learned upsampling decoder (10×8 → 160×128 → resized to H×W)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self._C0, 32, 4, stride=2, padding=1),  # → [32, 20, 16]
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),        # → [24, 40, 32]
            nn.BatchNorm2d(24), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 16, 4, stride=2, padding=1),        # → [16, 80, 64]
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),         # → [ 8,160,128]
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(8, OGM_CH, 3, padding=1)                 # raw logits [B,2,..]

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """x: [B, 1, 12, 256] → [B, 2, ogm_h, ogm_w]"""
        b = x.size(0)
        f = self.enc(x).flatten(1)                                # [B, 64*3*8]
        seed = self.fc(f).view(b, self._C0, self._H0, self._W0)   # [B,16,10,8]
        d = self.dec(seed)                                        # [B, 8,160,128]
        logits = self.head(d)                                     # [B, 1,160,128]
        logits = F.interpolate(logits, size=(self.ogm_h, self.ogm_w),
                               mode='bilinear', align_corners=False)
        return logits if return_logits else torch.sigmoid(logits)



# ─────────────────────────────────────────────────────────────────────────────
# OGM Ground-Truth helpers
# ─────────────────────────────────────────────────────────────────────────────

# Pre-compute a lookup table of Gaussian kernel values for efficiency
_BLOB_RADIUS = int(math.ceil(BLOB_SIGMA * 4))


def _gaussian_blob(grid: np.ndarray, row: int, col: int, sigma: float = BLOB_SIGMA) -> None:
    """Paint a 2D Gaussian blob centred at (row, col) onto grid in-place."""
    r0 = max(0, row - _BLOB_RADIUS)
    r1 = min(OGM_H, row + _BLOB_RADIUS + 1)
    c0 = max(0, col - _BLOB_RADIUS)
    c1 = min(OGM_W, col + _BLOB_RADIUS + 1)
    for r in range(r0, r1):
        for c in range(c0, c1):
            d2 = (r - row) ** 2 + (c - col) ** 2
            val = math.exp(-d2 / (2 * sigma ** 2))
            if val > grid[r, c]:
                grid[r, c] = val


def _obs_xy(ch: int, dist: float) -> tuple[float, float]:
    """Channel + range → obstacle (grid-x lateral, grid-y longitudinal) in metres."""
    sc = SENSOR_CFG[ch]
    yaw = math.radians(sc["yaw_deg"])
    ox = -(sc["y_m"] + dist * math.sin(yaw))
    oy =   sc["x_m"] + dist * math.cos(yaw)
    return ox, oy


def _xy_to_rc(ox: float, oy: float) -> tuple[int, int]:
    row = int(round((OGM_RANGE_FRONT - oy) / OGM_RES))
    col = int(round((ox + OGM_RANGE_SIDE) / OGM_RES))
    return row, col


def _paint_free_ray(free: np.ndarray, ch: int, max_dist: float) -> None:
    """Sweep the inverse-sensor-model free cone from sensor ch out to max_dist.

    The cone half-width grows with range (beam half-angle), so the swept region is
    a triangle/wedge of drivable space — exactly what a per-channel point detector
    (M2) cannot represent.
    """
    if max_dist <= 0.0:
        return
    sc  = SENSOR_CFG[ch]
    yaw = math.radians(sc["yaw_deg"])
    rx, ry = -math.sin(yaw), math.cos(yaw)   # ray direction (grid space)
    nx, ny = -ry, rx                         # perpendicular (lateral spread)
    ox0, oy0 = -sc["y_m"], sc["x_m"]         # sensor origin in grid space (dist=0)
    n = max(2, int(max_dist / OGM_RES) + 1)
    tan_half = math.tan(math.radians(FREE_HALF_ANGLE_DEG))
    for i in range(n):
        d = max_dist * i / (n - 1)
        cx, cy = ox0 + rx * d, oy0 + ry * d
        half = d * tan_half
        m = int(half / OGM_RES)
        for k in range(-m, m + 1):
            px = cx + nx * (k * OGM_RES)
            py = cy + ny * (k * OGM_RES)
            row, col = _xy_to_rc(px, py)
            if 0 <= row < OGM_H and 0 <= col < OGM_W:
                free[row, col] = 1.0


def _dilate_bool(mask: np.ndarray, r: int) -> np.ndarray:
    """Cheap iterative 4-neighbour binary dilation by r cells (pure numpy)."""
    out = mask.copy()
    for _ in range(max(0, r)):
        d = out.copy()
        d[1:, :]  |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:]  |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def build_ogm_gt(ann_list: list) -> np.ndarray:
    """Synthesize 2-channel OGM ground-truth from per-channel annotations.

    ann_list: length-12 list of dicts, each with:
        'obstacle_class' (int), 'distance_m' (float)
    Returns float32 array [2, OGM_H, OGM_W] in [0, 1]:
        ch0 = occupancy   (Gaussian blob at each detected obstacle)
        ch1 = free-space  (inverse-sensor-model cone up to the obstacle / clear range)

    Free-space is made physically consistent with occupancy:
      • each clear cone stops short of its own obstacle (FREE_MARGIN_M);
      • the wedge behind every obstacle is carved out as "unknown" (radial shadow),
        so a neighbouring clear cone cannot paint free behind a detected obstacle;
      • a buffer ring around every obstacle is cleared so green never touches yellow.
    """
    occ  = np.zeros((OGM_H, OGM_W), dtype=np.float32)
    free = np.zeros((OGM_H, OGM_W), dtype=np.float32)
    occ_dets: list[tuple[float, float]] = []   # (grid-x, grid-y) of each obstacle
    for ch, ann in enumerate(ann_list):
        if ann is None:
            # Sensor present but no annotation → treat as clear space.
            _paint_free_ray(free, ch, FREE_RANGE_EMPTY_M)
            continue
        cls  = int(ann.get("obstacle_class", 4))
        dist = float(ann.get("distance_m", 0.0))
        if cls in VALID_OBJ_CLASSES and dist > 0.0:
            ox, oy = _obs_xy(ch, dist)
            row, col = _xy_to_rc(ox, oy)
            if 0 <= row < OGM_H and 0 <= col < OGM_W:
                _gaussian_blob(occ, row, col)
                occ_dets.append((ox, oy))
            _paint_free_ray(free, ch, max(0.0, dist - FREE_MARGIN_M))
        else:
            # Open / Clutter → clear channel: free space up to the default reach.
            _paint_free_ray(free, ch, FREE_RANGE_EMPTY_M)

    # Radial occlusion: zero free in the angular shadow behind each obstacle, so a
    # clear neighbour cone cannot claim drivable space behind a detected obstacle.
    tol = math.radians(SHADOW_TOL_DEG)
    for ox, oy in occ_dets:
        r_obs = math.hypot(ox, oy)
        a_obs = math.atan2(ox, oy)
        dang  = np.abs(np.arctan2(np.sin(_CELL_BEARING - a_obs),
                                  np.cos(_CELL_BEARING - a_obs)))
        shadow = (dang < tol) & (_CELL_RANGE > r_obs + OGM_RES)
        free[shadow] = 0.0

    # Buffer carve: clear a ring around occupancy so green never overlaps yellow.
    buf = int(round(OCCLUSION_BUFFER_M / OGM_RES))
    free[_dilate_bool(occ > 0.3, buf)] = 0.0
    return np.stack([occ, free], axis=0)   # [2, OGM_H, OGM_W]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class M6Dataset(Dataset):
    """Per-frame dataset: loads 12-channel envelopes + builds OGM GT on-the-fly."""

    def __init__(self, frame_ids: list[str]):
        self._frames = frame_ids   # e.g. ["session_train_001_f0000", ...]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        fid = self._frames[idx]
        envelopes = np.zeros((N_SENSORS, ENVELOPE_LEN), dtype=np.float32)
        ann_list = [None] * N_SENSORS

        for ch in range(N_SENSORS):
            env_path = ENV_DIR / f"{fid}_c{ch:02d}.npy"
            ann_path = ENV_DIR / f"{fid}_c{ch:02d}_ann.npy"
            try:
                if env_path.exists():
                    envelopes[ch] = np.load(str(env_path)).astype(np.float32)
            except OSError:
                pass  # corrupted file — leave channel as zeros
            try:
                if ann_path.exists():
                    ann = np.load(str(ann_path), allow_pickle=True).item()
                    # Prefer the stored EDI distance (target_range_m) which matches
                    # edi_distance[] and thus the scatter-dot position in the OGM viewer.
                    # Fall back to argmax only when the field is absent or zero.
                    stored_dist = float(ann.get("target_range_m", 0.0))
                    if stored_dist > 0.0:
                        dist_m = stored_dist
                    else:
                        peak_idx = int(np.argmax(envelopes[ch]))
                        dist_m   = peak_idx / (ENVELOPE_LEN - 1) * DIST_MAX_M
                    ann["distance_m"] = dist_m
                    ann_list[ch] = ann
            except OSError:
                pass  # corrupted annotation — treat as no-obstacle

        envelopes = normalize_envelopes(envelopes)        # abs/fixed-ref + log (§6.5.3)
        x   = torch.from_numpy(envelopes).unsqueeze(0)   # [1, 12, 256]
        ogm = build_ogm_gt(ann_list)                      # [2, OGM_H, OGM_W]
        y   = torch.from_numpy(ogm)                       # [2, OGM_H, OGM_W]
        return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Playback-distribution dataset  (TestData/* — matches engine inference input)
# ─────────────────────────────────────────────────────────────────────────────

def _load_playback_sessions(session_ids: list[str]) -> list[tuple]:
    """Pre-load clean frames from TestData playback sessions into memory.

    Returns a list of (env[12,256] float32, edi[12] float32, labels[12] int64).
    Envelopes/edi_distance come straight from esi_frames.bin (the exact tensors the
    engine feeds M6); per-channel class labels come from ground_truth.json. Noise
    injection is bypassed (uses dm._frames directly) so GT is clean.
    """
    import json
    from modules.config_loader import ConfigLoader
    from modules.data_manager import DataManager

    cfg = ConfigLoader(str(ROOT / "config.yaml"))
    dm = DataManager(cfg)
    samples: list[tuple] = []
    for sid in session_ids:
        gt_path = TESTDATA_DIR / sid / "ground_truth.json"
        if not gt_path.exists():
            print(f"[M6] playback session {sid}: ground_truth.json missing — skipped.")
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        labels_by_fid = {int(fl["frame_id"]): fl["channel_labels"]
                         for fl in gt.get("frame_labels", [])}
        if not dm.load_session(sid):
            print(f"[M6] playback session {sid}: load_session failed — skipped.")
            continue
        for i in range(dm.get_frame_count()):
            fr = dm._frames[i]            # clean frame, no noise injection
            fid = int(getattr(fr, "frame_id", i))
            labels = labels_by_fid.get(fid)
            if labels is None:
                continue
            env = np.asarray(fr.envelopes,    dtype=np.float32).copy()   # [12,256]
            edi = np.asarray(fr.edi_distance,  dtype=np.float32).copy()  # [12]
            samples.append((env, edi, np.asarray(labels, dtype=np.int64)))
    return samples


def _playback_session_split() -> tuple[list[str], list[str]]:
    """Split the 34 session_train_* playback sessions into train/val by session.

    The 6 demo sessions (session_2026*) are held out entirely for visual
    verification and are NEVER used for training.
    """
    ids = [f"session_train_{i:03d}" for i in range(1, 35)]
    ids = [s for s in ids if (TESTDATA_DIR / s).exists()]
    val   = ids[-4:]    # last 4 sessions → validation
    train = ids[:-4]
    return train, val


class M6PlaybackDataset(Dataset):
    """In-memory dataset over pre-loaded playback frames. Builds the OGM GT from
    per-channel labels + edi_distance, identical geometry to window_ogm segments."""

    def __init__(self, samples: list[tuple]):
        self._s = samples

    def __len__(self) -> int:
        return len(self._s)

    def __getitem__(self, idx: int):
        env, edi, labels = self._s[idx]
        x = normalize_envelopes(env)                      # abs/fixed-ref + log (§6.5.3)
        ann_list = [{"obstacle_class": int(labels[ch]),
                     "distance_m":     float(edi[ch])} for ch in range(N_SENSORS)]
        ogm = build_ogm_gt(ann_list)                      # [2, OGM_H, OGM_W]
        return (torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(ogm))


# ─────────────────────────────────────────────────────────────────────────────
# Split file helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_m6_splits() -> None:
    """Derive M6 per-frame split lists from existing M2 per-channel split lists."""
    SPLIT_M6_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        m2_file = SPLIT_M2_DIR / f"{split}.txt"
        if not m2_file.exists():
            print(f"[M6 splits] M2 {split}.txt not found at {m2_file} — skipping.")
            continue
        with open(m2_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # Strip channel suffix _cXX to get frame IDs; deduplicate preserving order
        seen: set[str] = set()
        frame_ids: list[str] = []
        for line in lines:
            # sample_id format: "session_train_001_f0000_c00" → strip "_cXX"
            if "_c" in line:
                fid = line.rsplit("_c", 1)[0]
            else:
                fid = line
            if fid not in seen:
                seen.add(fid)
                frame_ids.append(fid)

        m6_file = SPLIT_M6_DIR / f"{split}.txt"
        with open(m6_file, "w", encoding="utf-8") as f:
            f.write("\n".join(frame_ids) + "\n")
        print(f"[M6 splits] {split}.txt: {len(frame_ids)} frames written → {m6_file}")


def _load_split(split: str) -> list[str]:
    p = SPLIT_M6_DIR / f"{split}.txt"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-3,
    seed: int = 42,
    early_stopping_patience: int = 15,
    source: str = "playback",
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"[M6] OGM grid: {OGM_H}×{OGM_W}  ({OGM_RANGE_FRONT}m front/rear, "
          f"{OGM_RANGE_SIDE}m side, {OGM_RES}m/cell)")

    # 1. Build datasets
    if source == "playback":
        tr_sess, va_sess = _playback_session_split()
        print(f"[M6] Source: playback (TestData) — "
              f"{len(tr_sess)} train sessions, {len(va_sess)} val sessions")
        train_samples = _load_playback_sessions(tr_sess)
        val_samples   = _load_playback_sessions(va_sess)
        if not train_samples:
            print("[M6] ERROR: no playback training frames found under TestData/.")
            return
        print(f"[M6] Frames — train: {len(train_samples)}, val: {len(val_samples)}")
        train_ds = M6PlaybackDataset(train_samples)
        val_ds   = M6PlaybackDataset(val_samples)
    else:
        _build_m6_splits()
        train_ids = _load_split("train")
        val_ids   = _load_split("val")
        if not train_ids:
            print("[M6] ERROR: no training samples found. Run gen_training_data.py first.")
            return
        print(f"[M6] Source: features — train: {len(train_ids)}, val: {len(val_ids)}")
        train_ds = M6Dataset(train_ids)
        val_ds   = M6Dataset(val_ids)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[M6] Device: {device}")
    model = SceneUnderstandingCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[M6] Trainable params: {n_params:,}")

    # 3. Loss + optimiser
    # OGM ch0 (occupancy) is sparse; ch1 (free-space) is dense. Combine, per channel:
    #   • weighted BCE (logits) → pixel-wise confidence, upweighting rare occupancy
    #   • soft Dice            → overlap/sharpness, computed per channel then averaged
    pos_weight = torch.tensor([15.0, 1.0], device=device).view(OGM_CH, 1, 1)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        num = 2.0 * (p * target).sum(dim=(2, 3)) + 1.0   # [B, C]
        den = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1.0
        return (1.0 - num / den).mean()

    def criterion(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return bce(logits, target) + dice_loss(logits, target)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x, return_logits=True)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= max(len(train_ds), 1)

        # ── Validate ────────────────────────────────────────────────────────
        val_loss = 0.0
        if val_dl and len(val_ds) > 0:
            model.eval()
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    logits = model(x, return_logits=True)
                    val_loss += criterion(logits, y).item() * x.size(0)
            val_loss /= max(len(val_ds), 1)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss and len(val_ds) > 0:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(MODEL_PATH))
        elif len(val_ds) > 0:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no val_loss improvement for {early_stopping_patience} epochs)")
                break

    # Final save if no validation set
    if len(val_ds) == 0:
        torch.save(model.state_dict(), str(MODEL_PATH))

    print(f"[M6] Training complete. Model saved → {MODEL_PATH}")
    print(f"[M6] Best val loss: {best_val_loss:.4f}")

    # 4. Quick sanity check
    _verify(device)


def _verify(device: torch.device) -> None:
    """Load saved weights and run one dummy forward pass to verify shape."""
    if not MODEL_PATH.exists():
        print("[M6] Verification skipped: model file not found.")
        return
    model = SceneUnderstandingCNN()
    state = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    dummy = torch.zeros(1, 1, N_SENSORS, ENVELOPE_LEN)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, OGM_CH, OGM_H, OGM_W), f"Unexpected output shape: {out.shape}"
    print(f"[M6] Verification OK — output shape: {tuple(out.shape)} "
          f"(ch0=occupancy, ch1=free), range [{out.min():.3f}, {out.max():.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train M6 Scene Understanding CNN")
    parser.add_argument("--epochs",     type=int,   default=80)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-3)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--patience",   type=int,   default=15)
    parser.add_argument("--source",     type=str,   default="playback",
                        choices=["playback", "features"],
                        help="playback = TestData/* (matches inference); "
                             "features = datasets/processed/envelopes (legacy)")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        early_stopping_patience=args.patience,
        source=args.source,
    )
