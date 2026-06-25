"""
tools/gen_training_data.py  —  AK2 模型训练仿真数据全量生成器
========================================================
依据《AK2 AI模型开发说明》§3.2.1 场景矩阵和《AK2数据采集格式定义》V2.0，
生成覆盖全部 9 类障碍物的 TestData 采集会话，并自动预处理为训练可用格式：

  TestData/session_train_NNN/   ←  L1 esi_frames.bin + ground_truth.json
  datasets/processed/edi_features_ch/{sid}_f{fid:04d}_c{ch:02d}.npy  M1 输入  (20,)
  datasets/processed/envelopes/{sid}_f{fid:04d}_c{ch:02d}.npy M2 输入  (256,)
  datasets/processed/envelopes/{sid}_f{fid:04d}_c{ch:02d}_ann.npy M2 标注
  datasets/annotations/M2_obstacle_class/{sid}_f{fid:04d}_m1.npy  M1 标注 (12,)
  datasets/splits/M1/train.txt  val.txt  test.txt
  datasets/splits/M2/train.txt  val.txt  test.txt

使用方法:
  cd F:\\CODE\\AK2_Sim
  python tools/gen_training_data.py
  python tools/gen_training_data.py --frames 200   # 快速验证，帧数减半
"""

import os, sys, json, struct, csv, hashlib, argparse, random, math
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shared feature extraction (same function used at inference time)
from modules.feature_extractor import extract_channel_features
from modules.obstacle_classes import CLASS_TYPICAL_WIDTH_M, WIDTH_CLASSES

TESTDATA_DIR   = os.path.join(ROOT, "TestData")
DATASETS_DIR   = os.path.join(ROOT, "datasets")
EDI_FEAT_DIR    = os.path.join(DATASETS_DIR, "processed", "edi_features")
EDI_FEAT_CH_DIR = os.path.join(DATASETS_DIR, "processed", "edi_features_ch")
ENV_DIR         = os.path.join(DATASETS_DIR, "processed", "envelopes")
ANN_DIR        = os.path.join(DATASETS_DIR, "annotations", "M2_obstacle_class")
SPLITS_M1_DIR   = os.path.join(DATASETS_DIR, "splits", "M1")
SPLITS_M2_DIR   = os.path.join(DATASETS_DIR, "splits", "M2")
SPLITS_M2SA_DIR = os.path.join(DATASETS_DIR, "splits", "M2SA")

FRAME_SIZE = 13376  # 固定帧字节数，见《数据采集格式定义》§6.3

CLASS_NAMES = {
    0: "Wall", 1: "Vehicle", 2: "Pedestrian", 3: "Soft",
    4: "Open", 5: "Clutter",  6: "Overhead",  7: "Curb", 8: "Wet",
}

# 12路传感器横向偏移量（m），基于 AK2 物理布局
# 索引 0-5：前组 (FL-Side, FL-Corner, FL-Center, FR-Center, FR-Corner, FR-Side)
# 索引 6-11：后组 (RL-Side, RL-Corner, RL-Center, RR-Center, RR-Corner, RR-Side)
CH_LATERAL_M = [-0.90, -0.60, -0.30, 0.30, 0.60, 0.90,
                -0.90, -0.60, -0.30, 0.30, 0.60, 0.90]

# M1 标签派生规则（文档§4.2）：有效障碍物类别
M1_VALID_CLASSES = {0, 1, 2, 3, 6, 7}

# ═══════════════════════════════════════════════════════════
# 1.  包络仿真函数（各类别典型物理特征，见§9.4）
# ═══════════════════════════════════════════════════════════

def _x():
    """0~6m 线性坐标，256点"""
    return np.linspace(0.0, 6.0, 256, dtype=np.float32)


def env_wall(dist: float, noise: float = 0.02) -> np.ndarray:
    """0-Wall: 高幅窄单峰，前沿锐利，帧间稳定"""
    x = _x(); amp = 0.88 + random.uniform(-0.05, 0.05)
    e = amp * np.exp(-((x - dist) ** 2) / (2 * 0.07 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_vehicle(dist: float, noise: float = 0.025) -> np.ndarray:
    """1-Vehicle: 高幅多峰，次级峰反映车体不同部位"""
    x = _x(); amp = 0.82 + random.uniform(-0.06, 0.06)
    e  = amp * np.exp(-((x - dist) ** 2) / (2 * 0.10 ** 2))
    e += 0.45 * np.exp(-((x - dist * 1.12) ** 2) / (2 * 0.08 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_pedestrian(dist: float, noise: float = 0.035) -> np.ndarray:
    """2-Pedestrian: 中低幅宽单峰，帧间抖动较大"""
    x = _x()
    dist_j = dist + random.uniform(-0.12, 0.12)
    amp = 0.52 + random.uniform(-0.10, 0.10)
    e = amp * np.exp(-((x - dist_j) ** 2) / (2 * 0.20 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_soft(dist: float, noise: float = 0.04) -> np.ndarray:
    """3-Soft: 低幅宽包络，不规则拖尾"""
    x = _x(); amp = 0.46 + random.uniform(-0.10, 0.10)
    e  = amp * np.exp(-((x - dist) ** 2) / (2 * 0.28 ** 2))
    e += 0.18 * np.exp(-((x - dist * 1.3) ** 2) / (2 * 0.35 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_open(noise: float = 0.015) -> np.ndarray:
    """4-Open: 全程低幅噪声基线，无明显峰"""
    e = np.random.normal(0, noise, 256)
    return np.clip(e, 0, 0.07).astype(np.float32)


def env_clutter(noise: float = 0.025) -> np.ndarray:
    """5-Clutter: 固定时延处中等幅度峰"""
    x = _x(); d = random.uniform(0.3, 0.8)
    e  = 0.42 * np.exp(-((x - d) ** 2) / (2 * 0.18 ** 2))
    e += 0.20 * np.exp(-x / 0.55)
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_overhead(dist: float, noise: float = 0.03) -> np.ndarray:
    """6-Overhead: 双峰结构（地面反射+高位反射）"""
    x = _x()
    clearance = random.uniform(1.6, 2.4)
    ground_d = dist * 0.92
    # 高位回波（悬空障碍物）
    amp_hi = 0.65 + random.uniform(-0.08, 0.08)
    e  = amp_hi * np.exp(-((x - dist) ** 2) / (2 * 0.09 ** 2))
    # 地面反射
    amp_lo = 0.30 + random.uniform(-0.06, 0.06)
    e += amp_lo * np.exp(-((x - ground_d) ** 2) / (2 * 0.12 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32), clearance


def env_curb(dist: float, noise: float = 0.03) -> np.ndarray:
    """7-Curb: 单峰，幅度低于同距离墙体"""
    x = _x(); amp = 0.65 + random.uniform(-0.08, 0.08)
    e = amp * np.exp(-((x - dist) ** 2) / (2 * 0.11 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def env_wet(dist: float, noise: float = 0.04) -> np.ndarray:
    """8-Wet: 地面反射峰异常高，可能出现二次回波"""
    x = _x(); amp = 0.72 + random.uniform(-0.10, 0.10)
    e  = amp * np.exp(-((x - dist) ** 2) / (2 * 0.12 ** 2))
    e += 0.40 * np.exp(-((x - dist * 1.85) ** 2) / (2 * 0.20 ** 2))
    return np.clip(e + np.random.normal(0, noise, 256), 0, 1).astype(np.float32)


def class_object_width(cls: int, rng: random.Random) -> float:
    """返回该类别的物体横向宽度标签（米）。

    以 CLASS_TYPICAL_WIDTH_M 为均值叠加 ±15% 抖动；非物体类（Open/Wet）返回 0.0。
    使用传入的独立 rng（按 sid+ch 确定性播种），不扰动全局随机流，
    以保证包络/硬度/悬空高度标签与原数据一致（M2-SC 骨干无需重训）。
    """
    if cls not in WIDTH_CLASSES:
        return 0.0
    base = CLASS_TYPICAL_WIDTH_M[cls]
    w = base + rng.uniform(-0.15, 0.15) * base
    return float(max(w, 0.05))


def make_envelope(cls: int, dist: float = None) -> tuple[np.ndarray, float, float]:
    """返回 (envelope(256), material_hardness, suspension_height_m)"""
    if dist is None:
        dist = random.uniform(0.3, 4.5)
    hardness = 0.0
    height_m = 0.0
    if cls == 0:
        return env_wall(dist), 0.90 + random.uniform(-0.05, 0.05), 0.0
    elif cls == 1:
        return env_vehicle(dist), 0.80 + random.uniform(-0.05, 0.05), 0.0
    elif cls == 2:
        return env_pedestrian(dist), 0.35 + random.uniform(-0.10, 0.10), 0.0
    elif cls == 3:
        return env_soft(dist), 0.25 + random.uniform(-0.10, 0.10), 0.0
    elif cls == 4:
        return env_open(), 0.0, 0.0
    elif cls == 5:
        return env_clutter(), 0.30 + random.uniform(-0.05, 0.05), 0.0
    elif cls == 6:
        env, clearance = env_overhead(dist)
        return env, 0.55 + random.uniform(-0.10, 0.10), clearance
    elif cls == 7:
        return env_curb(dist), 0.60 + random.uniform(-0.08, 0.08), 0.0
    elif cls == 8:
        return env_wet(dist), 0.10 + random.uniform(-0.05, 0.05), 0.0
    return env_open(), 0.0, 0.0


# ═══════════════════════════════════════════════════════════
# 2.  EDI 字段生成
# ═══════════════════════════════════════════════════════════

def make_edi(cls: int, dist: float) -> tuple[float, float, int, int]:
    """返回 (edi_distance, edi_amplitude, edi_confidence, edi_echo_type)"""
    if cls == 4:  # Open
        return 0.0, random.uniform(0.01, 0.04), random.randint(0, 12), 0
    elif cls == 5:  # Clutter
        d = random.uniform(0.3, 0.9)
        return d, random.uniform(0.30, 0.48), random.randint(15, 40), 3
    elif cls == 8:  # Wet
        return dist, random.uniform(0.62, 0.80), random.randint(40, 72), 1
    elif cls == 6:  # Overhead
        return dist, random.uniform(0.58, 0.72), random.randint(60, 82), 1
    elif cls == 2:  # Pedestrian
        return dist + random.uniform(-0.08, 0.08), random.uniform(0.45, 0.60), random.randint(58, 78), 1
    elif cls == 3:  # Soft
        return dist + random.uniform(-0.05, 0.05), random.uniform(0.40, 0.56), random.randint(52, 72), 1
    elif cls == 7:  # Curb
        return dist, random.uniform(0.60, 0.74), random.randint(68, 86), 1
    elif cls == 1:  # Vehicle
        return dist, random.uniform(0.78, 0.90), random.randint(85, 98), 1
    else:  # Wall (cls==0)
        return dist, random.uniform(0.85, 0.93), random.randint(88, 99), 1


# ═══════════════════════════════════════════════════════════
# 3.  240维 EDI 特征提取（M1 输入）
# ═══════════════════════════════════════════════════════════

def extract_edi_features(edi_dist, edi_amp, edi_conf, edi_echo, envelopes) -> np.ndarray:
    """
    [LEGACY — 仅用于备用 EDI_FEAT_DIR 存档]
    逐帧全路 240维特征（12 路共用 z-score 归一化。
    M1 模型训练实际使用的是逐通道 20维特征（extract_channel_features）。
    """
    feats = np.zeros((12, 20), dtype=np.float32)
    for ch in range(12):
        env = envelopes[ch]  # (256,)
        feats[ch, 0] = float(edi_dist[ch]) / 6.5           # 归一化距离
        feats[ch, 1] = float(edi_amp[ch])
        feats[ch, 2] = float(edi_conf[ch]) / 100.0
        feats[ch, 3] = float(edi_echo[ch]) / 3.0
        feats[ch, 4] = float(env.max())
        feats[ch, 5] = float(env.mean())
        feats[ch, 6] = float(env.std())
        peak_pos     = int(env.argmax())
        feats[ch, 7] = peak_pos / 255.0
        feats[ch, 8] = float(np.sqrt(np.mean(env ** 2)))    # RMS
        # 半峰宽
        peak_val = env[peak_pos]
        half = peak_val * 0.5
        above = np.where(env >= half)[0]
        feats[ch, 9] = float(len(above)) / 255.0
        # 包络偏度 & 峰度（归一化）
        feats[ch, 10] = float(np.mean((env - env.mean()) ** 3) / (env.std() ** 3 + 1e-8))
        feats[ch, 11] = float(np.mean((env - env.mean()) ** 4) / (env.std() ** 4 + 1e-8))
        # 8个均匀bin均值（256/8=32点/bin，整除）
        bins = env.reshape(8, 32).mean(axis=1)
        feats[ch, 12:20] = bins.astype(np.float32)
    # z-score 归一化（文档§2.1：M1输入范围 [-3,3]）
    mean = feats.mean(); std = feats.std() + 1e-8
    return np.clip((feats.flatten() - mean) / std, -3.0, 3.0).astype(np.float32)


def extract_channel_features(ch: int, edi_dist, edi_amp, edi_conf, edi_echo,
                              envelope: np.ndarray) -> np.ndarray:
    """单路 20维 EDI 特征提取（方案A：逐通道 M1 输入，无跨路归一化）"""
    feat = np.zeros(20, dtype=np.float32)
    env  = envelope  # (256,)
    feat[0] = float(edi_dist[ch]) / 6.5
    feat[1] = float(edi_amp[ch])
    feat[2] = float(edi_conf[ch]) / 100.0
    feat[3] = float(edi_echo[ch]) / 3.0
    feat[4] = float(env.max())
    feat[5] = float(env.mean())
    feat[6] = float(env.std())
    peak_pos = int(env.argmax())
    feat[7]  = peak_pos / 255.0
    feat[8]  = float(np.sqrt(np.mean(env ** 2)))
    above    = np.where(env >= env[peak_pos] * 0.5)[0]
    feat[9]  = float(len(above)) / 255.0
    feat[10] = float(np.mean((env - env.mean()) ** 3) / (env.std() ** 3 + 1e-8))
    feat[11] = float(np.mean((env - env.mean()) ** 4) / (env.std() ** 4 + 1e-8))
    feat[12:20] = env.reshape(8, 32).mean(axis=1).astype(np.float32)
    return feat


# ═══════════════════════════════════════════════════════════
# 4.  会话场景配置（18个训练+验证+测试会话）
# ═══════════════════════════════════════════════════════════
# 每个会话定义：(session_id, 元数据, 12路类别配置, 帧数基数, split)
# 12路布局: [FL-Side, FL-Corner, FL-Center, FR-Center, FR-Corner, FR-Side,
#             RL-Side, RL-Corner, RL-Center, RR-Center, RR-Corner, RR-Side]
# split: "train" / "val" / "test"

def _meta(date, weather, temp, road, desc, chip="YOUHANG"):
    return {
        "date": date, "weather": weather, "temperature_c": temp,
        "vehicle_model": "AK2 Sim Vehicle", "road_type": road,
        "description": desc, "chip_vendor": chip,
        "chip_model": "DJ628.30" if chip == "YOUHANG" else "G32A217",
        "bus_type": "ESI" if chip == "YOUHANG" else "DSI3",
        "bus_bitrate_kbps": 888 if chip == "YOUHANG" else 444,
        "sensor_count": 12, "frame_rate_hz": 10,
        "envelope_points": 256, "elastic_enabled": False,
        "timezone": "Asia/Shanghai",
    }


# channel_profile: list of 12 ints (obstacle class for each channel)
# dist_range: (min_dist, max_dist) for obstacle channels
TRAIN_SESSIONS = [
    # ── Class 0: Wall (≥2000 envelopes) ──────────────────────────────────────
    {
        "session_id": "session_train_001",
        "meta": _meta("2026-06-05", "Clear", 26.0, "Underground Parking",
                      "Wall approach: reverse parking into concrete wall"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,0,0,0],
        "dist_range": (0.3, 5.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-3, 3), "split": "train",
    },
    {
        "session_id": "session_train_002",
        "meta": _meta("2026-06-05", "Overcast", 22.0, "Underground Parking",
                      "Wall approach: low-temperature parking lot (concrete pillar)",
                      "GEEHY"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,0,0,0],
        "dist_range": (0.3, 4.5), "speed_range": (0.0, 4.0), "gear": "R",
        "steer_range": (-2, 2), "split": "train",
    },
    # ── Class 1: Vehicle (≥3000 envelopes) ───────────────────────────────────
    {
        "session_id": "session_train_003",
        "meta": _meta("2026-06-05", "Clear", 28.0, "Open Parking Lot",
                      "Vehicle ahead: approach parked SUV"),
        "channel_classes": [1,1,1,1,1,1, 4,4,4,4,4,4],
        "dist_range": (0.5, 5.0), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-2, 2), "split": "train",
    },
    {
        "session_id": "session_train_004",
        "meta": _meta("2026-06-05", "Clear", 30.0, "Urban Road",
                      "Vehicle ahead: slow-speed follow in parking structure"),
        "channel_classes": [4,1,1,1,1,4, 4,4,4,4,4,4],
        "dist_range": (1.0, 5.5), "speed_range": (5.0, 15.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_005",
        "meta": _meta("2026-06-06", "Cloudy", 24.0, "Underground Parking",
                      "Vehicle side: parking next to sedan, night"),
        "channel_classes": [1,1,4,4,1,1, 4,4,4,4,4,4],
        "dist_range": (0.3, 2.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-15, 15), "split": "train",
    },
    # ── Class 2: Pedestrian (≥2000 envelopes) ────────────────────────────────
    {
        "session_id": "session_train_006",
        "meta": _meta("2026-06-06", "Clear", 27.0, "Open Parking Lot",
                      "Pedestrian crossing: adult walking across front"),
        "channel_classes": [4,2,2,2,2,4, 4,4,4,4,4,4],
        "dist_range": (0.5, 3.5), "speed_range": (0.0, 5.0), "gear": "D",
        "steer_range": (-3, 3), "split": "train",
    },
    {
        "session_id": "session_train_007",
        "meta": _meta("2026-06-06", "Clear", 24.0, "Underground Parking",
                      "Pedestrian crossing: child + adult, nighttime",
                      "GEEHY"),
        "channel_classes": [4,2,2,2,2,4, 4,4,4,4,4,4],
        "dist_range": (0.4, 3.0), "speed_range": (0.0, 4.0), "gear": "R",
        "steer_range": (-5, 5), "split": "train",
    },
    # ── Class 3: Soft (≥1500 envelopes) ──────────────────────────────────────
    {
        "session_id": "session_train_008",
        "meta": _meta("2026-06-07", "Clear", 29.0, "Parking Lot",
                      "Soft obstacle: rubber cone + bush approach"),
        "channel_classes": [3,3,3,4,4,4, 4,4,4,4,4,4],
        "dist_range": (0.3, 4.0), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-8, 8), "split": "train",
    },
    {
        "session_id": "session_train_009",
        "meta": _meta("2026-06-07", "Rainy", 20.0, "Parking Lot",
                      "Soft obstacle: foam block + vegetation, rainy"),
        "channel_classes": [4,4,4,3,3,3, 4,4,4,4,4,4],
        "dist_range": (0.4, 3.5), "speed_range": (0.0, 6.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    # ── Class 4: Open (≥2000 envelopes) ──────────────────────────────────────
    {
        "session_id": "session_train_010",
        "meta": _meta("2026-06-07", "Clear", 32.0, "Open Area",
                      "Open area: no obstacles, all 12 channels"),
        "channel_classes": [4,4,4,4,4,4, 4,4,4,4,4,4],
        "dist_range": (0.0, 0.0), "speed_range": (0.0, 20.0), "gear": "D",
        "steer_range": (-15, 15), "split": "train",
    },
    # ── Class 5: Clutter (≥1000 envelopes) ───────────────────────────────────
    {
        "session_id": "session_train_011",
        "meta": _meta("2026-06-08", "Clear", 33.0, "Parking Lot",
                      "Ground clutter: road markings, slopes, asphalt reflections",
                      "GEEHY"),
        "channel_classes": [5,4,4,4,4,5, 5,4,4,4,4,5],
        "dist_range": (0.3, 0.9), "speed_range": (5.0, 15.0), "gear": "D",
        "steer_range": (-10, 10), "split": "train",
    },
    # ── Class 6: Overhead (≥1500 envelopes) ──────────────────────────────────
    {
        "session_id": "session_train_012",
        "meta": _meta("2026-06-08", "Clear", 26.0, "Parking Structure Entrance",
                      "Overhead: height-limit beam 1.8m, approach at 10 km/h"),
        "channel_classes": [4,6,6,6,6,4, 4,4,4,4,4,4],
        "dist_range": (0.5, 5.0), "speed_range": (0.0, 10.0), "gear": "D",
        "steer_range": (-2, 2), "split": "train",
    },
    {
        "session_id": "session_train_013",
        "meta": _meta("2026-06-08", "Rainy", 19.0, "Parking Structure",
                      "Overhead: rolling door beam 2.0m clearance, rainy",
                      "GEEHY"),
        "channel_classes": [4,6,6,6,6,4, 4,4,4,4,4,4],
        "dist_range": (0.5, 4.5), "speed_range": (0.0, 10.0), "gear": "D",
        "steer_range": (-3, 3), "split": "val",
    },
    # ── Class 7: Curb (≥1500 envelopes) ──────────────────────────────────────
    {
        "session_id": "session_train_014",
        "meta": _meta("2026-06-09", "Clear", 25.0, "Roadside",
                      "Curb: 10cm roadside curb, parallel parking right side"),
        "channel_classes": [4,4,4,4,4,7, 4,4,4,4,4,7],
        "dist_range": (0.2, 1.5), "speed_range": (0.0, 10.0), "gear": "D",
        "steer_range": (-20, 20), "split": "train",
    },
    {
        "session_id": "session_train_015",
        "meta": _meta("2026-06-09", "Clear", 27.0, "Roadside",
                      "Curb: 15cm curb approach perpendicular",
                      "GEEHY"),
        "channel_classes": [7,7,4,4,7,7, 4,4,4,4,4,4],
        "dist_range": (0.2, 2.0), "speed_range": (0.0, 8.0), "gear": "R",
        "steer_range": (-10, 10), "split": "val",
    },
    # ── Class 8: Wet (≥1000 envelopes) ───────────────────────────────────────
    {
        "session_id": "session_train_016",
        "meta": _meta("2026-06-09", "Rainy", 17.0, "Open Parking Lot",
                      "Wet ground: heavy rain, mirror reflection, all channels"),
        "channel_classes": [8,8,8,8,8,8, 8,8,8,8,8,8],
        "dist_range": (0.8, 2.5), "speed_range": (0.0, 15.0), "gear": "D",
        "steer_range": (-10, 10), "split": "train",
    },
    # ── 环境多样性补充（文档§3.3）────────────────────────────────────────────
    {
        "session_id": "session_train_017",
        "meta": _meta("2026-06-10", "Clear", 43.0, "Underground Parking",
                      "High temp (43°C): Wall + Open, sensor thermal drift",
                      "GEEHY"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,0,0,0],
        "dist_range": (0.4, 4.5), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-3, 3), "split": "val",
    },
    {
        "session_id": "session_train_018",
        "meta": _meta("2026-06-10", "Clear", -4.0, "Open Parking Lot",
                      "Low temp (-4°C): Wall + Vehicle, ice formation risk"),
        "channel_classes": [4,1,1,1,1,4, 0,0,0,0,0,0],
        "dist_range": (0.4, 4.5), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-5, 5), "split": "test",
    },
    {
        "session_id": "session_train_019",
        "meta": _meta("2026-06-11", "Clear", 24.0, "Underground Parking",
                      "Test set: Mixed multi-class scene (Wall+Pedestrian+Clutter)"),
        "channel_classes": [5,2,2,4,4,5, 0,0,0,0,0,0],
        "dist_range": (0.3, 4.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-8, 8), "split": "test",
    },
    # ── M2-SA 专属：组内混合类别场景（空间注意力判别训练信号）────────────────
    {
        "session_id": "session_train_020",
        "meta": _meta("2026-06-12", "Clear", 25.0, "Parking Structure",
                      "SA boundary: Wall left-half / Vehicle right-half front group"),
        "channel_classes": [0,0,0,1,1,1, 4,4,4,4,4,4],
        "dist_range": (0.4, 3.5), "speed_range": (0.0, 6.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_021",
        "meta": _meta("2026-06-12", "Clear", 27.0, "Underground Parking",
                      "SA boundary: Open left-half / Wall right-half (oblique approach)"),
        "channel_classes": [4,4,4,0,0,0, 0,0,0,0,0,0],
        "dist_range": (0.3, 4.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-10, 10), "split": "train",
    },
    {
        "session_id": "session_train_022",
        "meta": _meta("2026-06-13", "Cloudy", 23.0, "Open Parking Lot",
                      "SA surrounded: Vehicle center, Open sides (narrow passage)"),
        "channel_classes": [4,4,1,1,4,4, 4,4,4,4,4,4],
        "dist_range": (0.5, 3.0), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_023",
        "meta": _meta("2026-06-13", "Clear", 26.0, "Underground Parking",
                      "SA mixed: Wall sides / Pedestrian center (pedestrian between pillars)",
                      "GEEHY"),
        "channel_classes": [0,0,2,2,0,0, 4,4,4,4,4,4],
        "dist_range": (0.4, 3.5), "speed_range": (0.0, 4.0), "gear": "D",
        "steer_range": (-3, 3), "split": "val",
    },
    {
        "session_id": "session_train_024",
        "meta": _meta("2026-06-14", "Clear", 28.0, "Roadside",
                      "SA rear boundary: Wall left-half rear / Open right-half (reverse parking)"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,4,4,4],
        "dist_range": (0.3, 3.5), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-12, 12), "split": "train",
    },
    {
        "session_id": "session_train_025",
        "meta": _meta("2026-06-14", "Clear", 24.0, "Roadside",
                      "SA tri-class rear: Curb left / Wall center / Open right (kerb + pillar)"),
        "channel_classes": [4,4,4,4,4,4, 7,7,0,0,4,4],
        "dist_range": (0.2, 2.0), "speed_range": (0.0, 6.0), "gear": "R",
        "steer_range": (-15, 15), "split": "test",
    },
    # ── Side sensor obstacle coverage（补充 M6 侧面热力训练信号）────────────
    {
        "session_id": "session_train_026",
        "meta": _meta("2026-06-15", "Clear", 25.0, "Underground Parking",
                      "Side Wall: narrow corridor, FL-Side + FR-Side detect concrete wall"),
        "channel_classes": [0,4,4,4,4,0, 4,4,4,4,4,4],
        "dist_range": (0.4, 3.0), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-3, 3), "split": "train",
    },
    {
        "session_id": "session_train_027",
        "meta": _meta("2026-06-15", "Clear", 26.0, "Parking Structure",
                      "Side Wall: all front channels detect wall (tight passage surrounded)"),
        "channel_classes": [0,0,0,0,0,0, 4,4,4,4,4,4],
        "dist_range": (0.3, 3.5), "speed_range": (0.0, 6.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_028",
        "meta": _meta("2026-06-15", "Overcast", 22.0, "Underground Parking",
                      "Side Wall: RL-Side + RR-Side detect wall when reversing into narrow bay",
                      "GEEHY"),
        "channel_classes": [4,4,4,4,4,4, 0,4,4,4,4,0],
        "dist_range": (0.3, 3.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-5, 5), "split": "val",
    },
    {
        "session_id": "session_train_029",
        "meta": _meta("2026-06-16", "Clear", 27.0, "Roadside",
                      "Side Curb: all 4 side sensors (FL/FR/RL/RR-Side) detect road curb"),
        "channel_classes": [7,4,4,4,4,7, 7,4,4,4,4,7],
        "dist_range": (0.3, 2.5), "speed_range": (0.0, 15.0), "gear": "D",
        "steer_range": (-5, 5), "split": "val",
    },
    {
        "session_id": "session_train_030",
        "meta": _meta("2026-06-16", "Clear", 28.0, "Parking Structure",
                      "Mixed Side+Front: FL/FR-Side=Wall + inner channels=Vehicle (tight lane)"),
        "channel_classes": [0,1,1,1,1,0, 4,4,4,4,4,4],
        "dist_range": (0.5, 3.5), "speed_range": (0.0, 8.0), "gear": "D",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_031",
        "meta": _meta("2026-06-16", "Clear", 24.0, "Parking Structure",
                      "All-surround Wall: all 12 channels detect concrete walls (parking bay)",
                      "GEEHY"),
        "channel_classes": [0,0,0,0,0,0, 0,0,0,0,0,0],
        "dist_range": (0.3, 4.0), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-8, 8), "split": "val",
    },
    # ── 短距精确定位补充（改善 M6 位置精度）──────────────────────────────────
    {
        "session_id": "session_train_032",
        "meta": _meta("2026-06-17", "Clear", 24.0, "Underground Parking",
                      "Tight reverse: all 6 rear channels Wall at ultra-close range (0.2-1.5m)"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,0,0,0],
        "dist_range": (0.2, 1.5), "speed_range": (0.0, 3.0), "gear": "R",
        "steer_range": (-5, 5), "split": "train",
    },
    {
        "session_id": "session_train_033",
        "meta": _meta("2026-06-17", "Clear", 25.0, "Underground Parking",
                      "Asymmetric rear-left: RL-Side+RL-Corner+RL-Center=Wall, right=Open"),
        "channel_classes": [4,4,4,4,4,4, 0,0,0,4,4,4],
        "dist_range": (0.3, 2.5), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-8, 8), "split": "train",
    },
    {
        "session_id": "session_train_034",
        "meta": _meta("2026-06-17", "Overcast", 23.0, "Underground Parking",
                      "Asymmetric rear-right: RR-Center+RR-Corner+RR-Side=Wall, left=Open"),
        "channel_classes": [4,4,4,4,4,4, 4,4,4,0,0,0],
        "dist_range": (0.3, 2.5), "speed_range": (0.0, 5.0), "gear": "R",
        "steer_range": (-8, 8), "split": "train",
    },
]


# ═══════════════════════════════════════════════════════════
# 5.  单会话生成器
# ═══════════════════════════════════════════════════════════

def generate_session(cfg: dict, num_frames: int, preprocess: bool = True) -> list[str]:
    """
    生成一个训练会话的全部文件。
    返回：本会话的 M1 sample_id 列表（每帧一个）
    """
    session_id   = cfg["session_id"]
    session_path = os.path.join(TESTDATA_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    ch_classes  = cfg["channel_classes"]   # 长度12
    dist_min, dist_max = cfg["dist_range"]
    sp_min, sp_max     = cfg["speed_range"]
    st_min, st_max     = cfg["steer_range"]
    gear               = cfg["gear"]

    frames_path = os.path.join(session_path, "esi_frames.bin")
    can_path    = os.path.join(session_path, "can_signals.csv")
    meta_path   = os.path.join(session_path, "session_meta.json")
    gt_path     = os.path.join(session_path, "ground_truth.json")

    # 障碍物距离：按帧从远到近（模拟车辆靠近）
    def frame_dist(fid, total):
        progress = fid / max(total - 1, 1)
        if dist_max <= dist_min:
            return 0.0
        return dist_max - progress * (dist_max - dist_min)

    gt_frame_labels = []
    can_rows = []
    m1_sample_ids = []

    with open(frames_path, "wb") as f_bin:
        for fid in range(num_frames):
            ts = float(fid * 100)  # 10 Hz → 100 ms/帧
            dist = frame_dist(fid, num_frames)
            progress = fid / max(num_frames - 1, 1)
            speed  = sp_max - progress * (sp_max - sp_min)
            steer  = random.uniform(st_min, st_max)

            # 每帧为前后两组各随机采样一个障碍物入射角（模拟车辆偏斜停靠）
            # 入射角范围 ±0.30 rad（约 ±17°），独立于转向角
            front_angle = random.uniform(-0.30, 0.30)
            rear_angle  = random.uniform(-0.30, 0.30)

            edi_dist = np.zeros(12, np.float32)
            edi_amp  = np.zeros(12, np.float32)
            edi_conf = np.zeros(12, np.uint8)
            edi_echo = np.zeros(12, np.uint8)
            envelopes = np.zeros((12, 256), np.float32)

            for ch in range(12):
                cls     = ch_classes[ch]
                grp_idx = ch % 6
                angle   = front_angle if ch < 6 else rear_angle

                # 基于几何布局计算各路实际距离（同组同类通道距离连续变化）
                # Open / Clutter 无实体距离依赖，仍独立随机
                if cls in (4, 5) or dist <= 0:
                    d = 0.0
                else:
                    lat = CH_LATERAL_M[grp_idx]
                    d_geo = dist + lat * math.tan(angle)
                    d = max(d_geo + random.uniform(-0.02, 0.02), 0.05)

                env_result = make_envelope(cls, d if d > 0 else None)
                envelopes[ch] = env_result[0]
                d_edi = d if cls not in (4, 5) else 0.0
                dd, da, dc, de = make_edi(cls, d_edi)
                edi_dist[ch], edi_amp[ch], edi_conf[ch], edi_echo[ch] = dd, da, dc, de

            # 写二进制帧
            elastic = np.zeros((12, 20), np.float32)
            f_bin.write(struct.pack("<I", fid))
            f_bin.write(struct.pack("<f", ts))
            f_bin.write(edi_dist.tobytes())
            f_bin.write(edi_amp.tobytes())
            f_bin.write(edi_conf.tobytes())
            f_bin.write(edi_echo.tobytes())
            f_bin.write(envelopes.tobytes())
            f_bin.write(elastic.tobytes())

            # CAN
            can_rows.append({
                "timestamp_ms": ts, "speed_kmh": round(speed, 2),
                "steering_angle_deg": round(steer, 2), "gear": gear,
            })

            # GT
            gt_frame_labels.append({
                "frame_id": fid,
                "channel_labels": [int(c) for c in ch_classes],
                "annotator": "auto",
                "confidence": "high",
            })

            # 预处理：生成 .npy 特征文件
            if preprocess:
                sid = f"{session_id}_f{fid:04d}"
                m1_sample_ids.append(sid)
                _save_features(sid, edi_dist, edi_amp, edi_conf, edi_echo,
                               envelopes, ch_classes)

    # 写 CAN CSV
    with open(can_path, "w", newline="", encoding="utf-8") as fc:
        w = csv.DictWriter(fc, fieldnames=["timestamp_ms","speed_kmh","steering_angle_deg","gear"])
        w.writeheader(); w.writerows(can_rows)

    # 写 session_meta.json
    meta = dict(cfg["meta"])
    meta["session_id"]    = session_id
    meta["total_frames"]  = num_frames
    with open(meta_path, "w", encoding="utf-8") as fm:
        json.dump(meta, fm, ensure_ascii=False, indent=2)

    # 写 ground_truth.json
    gt_data = {
        "session_id": session_id,
        "num_classes": 9,
        "class_names": CLASS_NAMES,
        "frame_labels": gt_frame_labels,
        # 兼容旧格式 key（仿真器读取）
        "frames": {
            str(fl["frame_id"]): {
                "class_ids": fl["channel_labels"],
                "has_collision": False, "collision_type": 0
            } for fl in gt_frame_labels
        },
    }
    with open(gt_path, "w", encoding="utf-8") as fg:
        json.dump(gt_data, fg, ensure_ascii=False, indent=2)

    # 写 checksums.json
    _write_checksums(session_path, session_id)

    fsize_kb = os.path.getsize(frames_path) / 1024
    print(f"  [OK] {session_id:32s} {num_frames:4d} 帧  {fsize_kb:7.1f} KB  "
          f"split={cfg['split']:5s}  classes={set(ch_classes)}")

    return m1_sample_ids


def _save_features(sid, edi_dist, edi_amp, edi_conf, edi_echo, envelopes, ch_classes):
    """将单帧数据预处理并保存到 datasets/processed/"""
    # M1：帧级 EDI 特征向量 (240,)（保留备用）
    feat_m1 = extract_edi_features(edi_dist, edi_amp, edi_conf, edi_echo, envelopes)
    np.save(os.path.join(EDI_FEAT_DIR, f"{sid}.npy"), feat_m1)

    # M1 标签（12,）二値，从9类标签派生
    m1_label = np.array([1.0 if c in M1_VALID_CLASSES else 0.0
                         for c in ch_classes], dtype=np.float32)
    np.save(os.path.join(ANN_DIR, f"{sid}_m1.npy"), m1_label)

    # M1 逐通道特征（方案A）：帧级文件 (12,20) + (12,) 标签（避免每帧生成 24 个小文件）
    ch_feats  = np.stack([extract_channel_features(ch, edi_dist, edi_amp, edi_conf, edi_echo,
                                                   envelopes[ch]) for ch in range(12)])
    ch_labels = np.array([1.0 if ch_classes[ch] in M1_VALID_CLASSES else 0.0
                          for ch in range(12)], dtype=np.float32)
    np.save(os.path.join(EDI_FEAT_CH_DIR, f"{sid}_feat.npy"),   ch_feats)
    np.save(os.path.join(EDI_FEAT_CH_DIR, f"{sid}_m1lbl.npy"),  ch_labels)
    # M2 标签（12,）9类整型，直接存储统一障碍物分类 ID（0-8）
    m2_labels = np.array(ch_classes, dtype=np.int64)
    np.save(os.path.join(EDI_FEAT_CH_DIR, f"{sid}_m2lbl.npy"), m2_labels)

    # M2-SA 组级样本：前组 (ch 0-5) + 后组 (ch 6-11) 各保存一个 (6,256) 包络矩阵
    # 及对应的 (6,) 类别标签，供 SA 训练时整组加载
    np.save(os.path.join(ENV_DIR, f"{sid}_gF.npy"),
            envelopes[0:6].astype(np.float32))                         # (6, 256)
    np.save(os.path.join(ANN_DIR, f"{sid}_gF_lbl.npy"),
            np.array(ch_classes[0:6], dtype=np.int64))                 # (6,)
    np.save(os.path.join(ENV_DIR, f"{sid}_gR.npy"),
            envelopes[6:12].astype(np.float32))                        # (6, 256)
    np.save(os.path.join(ANN_DIR, f"{sid}_gR_lbl.npy"),
            np.array(ch_classes[6:12], dtype=np.int64))                # (6,)

    # M2：每路包络独立存储 (256,) + 标注 dict
    for ch in range(12):
        cid = f"{sid}_c{ch:02d}"
        np.save(os.path.join(ENV_DIR, f"{cid}.npy"), envelopes[ch])

        cls = ch_classes[ch]
        # 仅保存有意义的包络（非 Open、置信度高的）
        conf_val = int(edi_conf[ch])
        # 独立 RNG（按 sid+ch 确定性播种）生成宽度标签，不扰动全局随机流
        _wrng = random.Random((hash(sid) ^ (ch * 2654435761)) & 0xFFFFFFFF)
        ann = {
            "obstacle_class": int(cls),
            "confidence": "High" if conf_val >= 70 else ("Medium" if conf_val >= 40 else "Low"),
            "material_hardness": float(
                {0:0.90, 1:0.80, 2:0.35, 3:0.25, 4:0.0, 5:0.30,
                 6:0.55, 7:0.62, 8:0.10}.get(cls, 0.5)
                + random.uniform(-0.05, 0.05)),
            "is_suspended": bool(cls == 6),
            "suspension_height_m": float(random.uniform(1.6, 2.4)) if cls == 6 else 0.0,
            "object_width_m": class_object_width(cls, _wrng),
            "target_range_m": float(edi_dist[ch]),
            "envelope_quality": "Clear" if conf_val >= 70 else ("Noisy" if conf_val >= 30 else "Invalid"),
        }
        np.save(os.path.join(ENV_DIR, f"{cid}_ann.npy"),
                np.array(ann, dtype=object))


def _write_checksums(session_path: str, session_id: str) -> None:
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return "sha256:" + h.hexdigest()

    files = {
        fname: sha256(os.path.join(session_path, fname))
        for fname in ["session_meta.json", "can_signals.csv",
                      "esi_frames.bin", "ground_truth.json"]
        if os.path.exists(os.path.join(session_path, fname))
    }
    cs = {
        "session_id": session_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": files,
    }
    with open(os.path.join(session_path, "checksums.json"), "w") as f:
        json.dump(cs, f, indent=2)


# ═══════════════════════════════════════════════════════════
# 6.  数据集划分文件生成
# ═══════════════════════════════════════════════════════════

def write_splits(all_m1_ids: list[str], session_splits: dict[str, str]) -> None:
    """
    生成 datasets/splits/{M1,M2}/{train,val,test}.txt
    按会话划分（同一会话的所有帧不跨集，文档§5.2）
    """
    buckets = {"train": [], "val": [], "test": []}

    for sid_frame in all_m1_ids:
        # sid_frame 格式 "session_train_NNN_fXXXX"
        # session_id 为前三段: "session_train_NNN"
        session_id = "_".join(sid_frame.split("_")[:3])
        split = session_splits.get(session_id, "train")
        buckets[split].append(sid_frame)

    for split, ids in buckets.items():
        # M1 & M2 split: 均为通道级（方案A：M1 逐通道推理）
        ch_ids = [f"{sid}_c{ch:02d}" for sid in ids for ch in range(12)]
        with open(os.path.join(SPLITS_M1_DIR, f"{split}.txt"), "w") as f:
            f.write("\n".join(ch_ids) + ("\n" if ch_ids else ""))
        with open(os.path.join(SPLITS_M2_DIR, f"{split}.txt"), "w") as f:
            f.write("\n".join(ch_ids) + ("\n" if ch_ids else ""))
        # M2-SA split: 组级（每帧 2 个样本：gF + gR）
        sa_ids = [f"{sid}_gF" for sid in ids] + [f"{sid}_gR" for sid in ids]
        with open(os.path.join(SPLITS_M2SA_DIR, f"{split}.txt"), "w") as f:
            f.write("\n".join(sa_ids) + ("\n" if sa_ids else ""))

    # 统计输出
    for split in ["train", "val", "test"]:
        frame_n = len(buckets[split])
        ch_n    = frame_n * 12
        sa_n    = frame_n * 2
        print(f"  {split:5s}: 帧={frame_n:5d}  通道样本(M1=M2)={ch_n:6d}  SA组样本={sa_n:5d}")


# ═══════════════════════════════════════════════════════════
# 7.  更新 TestData/index.json
# ═══════════════════════════════════════════════════════════

def update_index() -> None:
    sessions = sorted(
        e for e in os.listdir(TESTDATA_DIR)
        if os.path.isdir(os.path.join(TESTDATA_DIR, e)) and e.startswith("session_")
    )
    idx = {
        "version": "1.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sessions": sessions,
    }
    with open(os.path.join(TESTDATA_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f"\n  index.json 更新完成，共 {len(sessions)} 个会话")


# ═══════════════════════════════════════════════════════════
# 8.  统计样本数量
# ═══════════════════════════════════════════════════════════

def print_class_stats(all_m1_ids: list[str], session_cfgs: list[dict]) -> None:
    """按类别统计M2通道样本数量"""
    counts = {i: 0 for i in range(9)}
    for cfg in session_cfgs:
        sid = cfg["session_id"]
        n_frames = sum(1 for s in all_m1_ids if s.startswith(sid))
        for cls in cfg["channel_classes"]:
            counts[cls] += n_frames

    min_req = {0:2000, 1:3000, 2:2000, 3:1500, 4:2000, 5:1000, 6:1500, 7:1500, 8:1000}
    print("\n  M2 通道样本量统计（文档§3.2.1最低要求）:")
    print(f"  {'类别':4s} {'名称':12s} {'实际':>8s} {'要求':>8s} {'状态':6s}")
    print(f"  {'-'*50}")
    for cls in range(9):
        actual = counts[cls]; req = min_req[cls]
        status = "OK" if actual >= req else "NG"
        print(f"  {cls:4d} {CLASS_NAMES[cls]:12s} {actual:8d} {req:8d} {status}")


# ═══════════════════════════════════════════════════════════
# 9.  主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AK2 训练数据生成器")
    parser.add_argument("--frames", type=int, default=300,
                        help="每会话帧数（默认300；用200可快速验证流水线）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="跳过 datasets/processed/ 预处理步骤")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    preprocess = not args.no_preprocess

    # 创建目录
    for d in [TESTDATA_DIR, EDI_FEAT_DIR, EDI_FEAT_CH_DIR, ENV_DIR, ANN_DIR,
              SPLITS_M1_DIR, SPLITS_M2_DIR, SPLITS_M2SA_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"\nAK2 训练数据生成器  (每会话 {args.frames} 帧)")
    print(f"  TestData      → {TESTDATA_DIR}")
    print(f"  processed/    → {os.path.join(DATASETS_DIR, 'processed')}")
    print(f"  preprocess    = {preprocess}\n")

    all_m1_ids = []
    session_splits = {}

    for cfg in TRAIN_SESSIONS:
        ids = generate_session(cfg, args.frames, preprocess=preprocess)
        all_m1_ids.extend(ids)
        session_splits[cfg["session_id"]] = cfg["split"]

    update_index()

    if preprocess:
        print("\n  生成数据集划分文件...")
        write_splits(all_m1_ids, session_splits)
        print_class_stats(all_m1_ids, TRAIN_SESSIONS)

    total_frames = len(all_m1_ids)
    print(f"\n  完成！总帧数: {total_frames}  "
          f"M1通道样本: {total_frames * 12}  M2通道样本: {total_frames * 12}")
    print(f"\n  下一步：运行训练")
    print(f"    cd F:\\CODE\\AK2_Sim")
    print(f"    python train/train_M1.py")
    print(f"    python train/train_M2.py\n")


if __name__ == "__main__":
    main()
