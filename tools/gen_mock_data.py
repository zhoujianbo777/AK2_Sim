"""
tools/gen_mock_data.py  —  Simulated test data generator
Generates simulation test data conforming to the format in spec section 3.1,
for debugging the software without real vehicle data.

Generated session scenarios:
  session_20260515_001  —  Normal parking scenario (mainly hard obstacles)
  session_20260515_002  —  Overhead obstacle scenario (height-limit beam)
  session_20260515_003  —  Wet ground scenario (rainy weather, puddles)
  session_20260515_004  —  Collision event scenario (with elastic wave trigger)

Usage:
  python tools/gen_mock_data.py
  python tools/gen_mock_data.py --output TestData --frames 200
"""

import os
import sys
import json
import math
import struct
import argparse
import numpy as np
import csv
from datetime import datetime

# Ensure project modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────────────────
# Scenario configuration
# ─────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "session_id": "session_20260515_001",
        "meta": {
            "date": "2026-05-15",
            "weather": "Clear",
            "temperature_c": 25.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Underground Parking",
            "description": "Normal parking scenario, hard wall ahead, low-speed reverse",
            "total_frames": 0  # filled automatically
        },
        "profile": "parking_hard",
        "has_gt": True,
    },
    {
        "session_id": "session_20260515_002",
        "meta": {
            "date": "2026-05-15",
            "weather": "Clear",
            "temperature_c": 23.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Parking Lot Entrance",
            "description": "Overhead obstacle scenario, height-limit beam ~1.8m clearance",
            "total_frames": 0
        },
        "profile": "suspended_obstacle",
        "has_gt": True,
    },
    {
        "session_id": "session_20260515_003",
        "meta": {
            "date": "2026-05-15",
            "weather": "Rainy",
            "temperature_c": 18.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Wet Road",
            "description": "Rainy wet ground scenario, mirror reflection, high false-alarm rate for traditional algo",
            "total_frames": 0
        },
        "profile": "wet_ground",
        "has_gt": True,
    },
    {
        "session_id": "session_20260515_004",
        "meta": {
            "date": "2026-05-15",
            "weather": "Clear",
            "temperature_c": 26.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Parking Lot",
            "description": "Collision event scenario, low-speed light bump on wheel stopper",
            "total_frames": 0
        },
        "profile": "collision_event",
        "has_gt": True,
    },
    {
        "session_id": "session_20260516_005",
        "meta": {
            "date": "2026-05-16",
            "weather": "Clear",
            "temperature_c": 24.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Urban Road",
            "description": "Mixed urban scenario: vehicle ahead, pedestrian crossing, ground clutter on sides",
            "total_frames": 0
        },
        "profile": "urban_mixed",
        "has_gt": True,
    },
    {
        "session_id": "session_20260516_006",
        "meta": {
            "date": "2026-05-16",
            "weather": "Clear",
            "temperature_c": 22.0,
            "vehicle_model": "Test Vehicle A",
            "road_type": "Roadside",
            "description": "Soft cone + curb detection scenario: soft obstacle front-left, curb on right side",
            "total_frames": 0
        },
        "profile": "curb_soft",
        "has_gt": True,
    },
]

# ─────────────────────────────────────────────────────────────
# Frame binary format parameters (matches parsing logic in data_manager.py)
# Per frame: 4+4+48+48+12+12+12288+960 = 13376 bytes
# ─────────────────────────────────────────────────────────────

FRAME_SIZE = 13376

# 9-class obstacle GT labels (AI model output classes)
GT_LABELS = {
    0: "Wall",
    1: "Vehicle",
    2: "Pedestrian",
    3: "Soft",
    4: "Open",
    5: "Clutter",
    6: "Overhead",
    7: "Curb",
    8: "Wet",
}


# ─────────────────────────────────────────────────────────────# Vehicle odometry + sensor geometry (MUST match views/window_ogm.py & config.yaml)
# Obstacle distances are derived from the integrated CAN trajectory so that a static
# world obstacle yields a frame-to-frame consistent range (W4 accumulation collapses
# to a stable contour instead of a smeared ladder).
# ────────────────────────────────────────────────
WHEELBASE_M  = 2.7    # matches display.wheelbase_m default in window_ogm
STEER_RATIO  = 15.0   # matches display.steering_ratio default in window_ogm
FRAME_DT_S   = 0.1    # 10 Hz → 100 ms/frame (matches timestamp spacing)

# Sensor mount config (vehicle frame: +X fwd, +Y left; yaw CCW from +X), ch0–ch11.
# Identical to config.yaml sensors S01–S12.
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

# Per-scenario initial obstacle distances (m) along each detecting sensor's boresight.
PARKING_REAR_D0 = 4.5
SUSPENDED_D0    = 4.0
WET_D0          = 1.5
COLLISION_D0    = 1.5
URBAN_VEH_D0    = 4.5
URBAN_PED_D0    = 2.5
CURB_CONE_D0    = 3.0
CURB_D0         = 0.6


def integrate_pose(pose: tuple, speed_kmh: float, steer_deg: float,
                   gear: str, dt: float = FRAME_DT_S) -> tuple:
    """Advance ego pose (x, y, theta) one step via a single-track (单轨 / kinematic bicycle) model.
    Identical maths to WindowOGM._integrate_ego so distances stay registered."""
    x, y, th = pose
    v = speed_kmh / 3.6
    if str(gear).upper() == "R":
        v = -v
    delta = math.radians(steer_deg / STEER_RATIO)
    x += v * math.cos(th) * dt
    y += v * math.sin(th) * dt
    th += (v / WHEELBASE_M) * math.tan(delta) * dt
    return (x, y, th)


def build_trajectory(can_list: list) -> list:
    """Integrate a list of (speed_kmh, steer_deg, gear) into ego poses.
    pose[0] = origin (first frame applies no motion, matching window_ogm)."""
    poses = [(0.0, 0.0, 0.0)]
    for f in range(1, len(can_list)):
        s, st, g = can_list[f]
        poses.append(integrate_pose(poses[-1], s, st, g))
    return poses


def sensor_world_pos(pose: tuple, ch: int) -> tuple:
    """World-frame position of sensor `ch` at the given ego pose."""
    x, y, th = pose
    sc = SENSOR_CFG[ch]
    mx, my = sc["x_m"], sc["y_m"]
    c, s = math.cos(th), math.sin(th)
    return (x + mx * c - my * s, y + mx * s + my * c)


def obstacle_point(ch: int, d0: float) -> tuple:
    """Fixed world point at distance d0 along sensor `ch`'s boresight (ego at origin)."""
    sc = SENSOR_CFG[ch]
    mx, my = sc["x_m"], sc["y_m"]
    yaw = math.radians(sc["yaw_deg"])
    return (mx + d0 * math.cos(yaw), my + d0 * math.sin(yaw))


def range_to(pose: tuple, ch: int, point: tuple) -> float:
    """Boresight-projected range from sensor `ch` (at ego pose) to a fixed world point.

    Ultrasonic localisation (and W4) assumes the echo lies along the sensor boresight,
    so the *axial* projection — not the raw Euclidean distance — is the range that lets
    W4's reprojection recover a stable world position as the vehicle moves.
    """
    x, y, th = pose
    sx, sy = sensor_world_pos(pose, ch)
    wyaw = th + math.radians(SENSOR_CFG[ch]["yaw_deg"])
    proj = (point[0] - sx) * math.cos(wyaw) + (point[1] - sy) * math.sin(wyaw)
    return max(proj, 0.1)


# ────────────────────────────────────────────────# Envelope generation helpers
# ─────────────────────────────────────────────────────────────

def gen_envelope_hard(distance_m: float, noise_level: float = 0.02) -> np.ndarray:
    """Hard obstacle envelope: single narrow peak, steep, high amplitude."""
    x = np.linspace(0, 6.0, 256)
    env = np.exp(-((x - distance_m) ** 2) / (2 * 0.08 ** 2)) * 0.92
    env += np.random.normal(0, noise_level, 256)
    return np.clip(env, 0, 1).astype(np.float32)


def gen_envelope_soft(distance_m: float, noise_level: float = 0.03) -> np.ndarray:
    """Soft obstacle envelope: wide peak, medium amplitude."""
    x = np.linspace(0, 6.0, 256)
    env = np.exp(-((x - distance_m) ** 2) / (2 * 0.25 ** 2)) * 0.55
    env += np.random.normal(0, noise_level, 256)
    return np.clip(env, 0, 1).astype(np.float32)


def gen_envelope_open(noise_level: float = 0.015) -> np.ndarray:
    """Open (no obstacle) envelope: low-amplitude noise floor."""
    env = np.random.normal(0, noise_level, 256)
    return np.clip(env, 0, 0.08).astype(np.float32)


def gen_envelope_ground_clutter(noise_level: float = 0.02) -> np.ndarray:
    """Ground clutter envelope: long near-range tail."""
    x = np.linspace(0, 6.0, 256)
    env = np.exp(-x / 0.5) * 0.45
    env += np.random.normal(0, noise_level, 256)
    return np.clip(env, 0, 1).astype(np.float32)


def gen_envelope_wet(distance_m: float, noise_level: float = 0.04) -> np.ndarray:
    """Wet ground envelope: mirror reflection, multi-peak, unstable amplitude."""
    x = np.linspace(0, 6.0, 256)
    env  = np.exp(-((x - distance_m) ** 2) / (2 * 0.12 ** 2)) * 0.70
    env += np.exp(-((x - distance_m * 1.8) ** 2) / (2 * 0.20 ** 2)) * 0.35  # multiple reflections
    env += np.random.normal(0, noise_level, 256)
    return np.clip(env, 0, 1).astype(np.float32)


def gen_elastic_features(has_collision: bool = False,
                          collision_strength: float = 0.0) -> np.ndarray:
    """Elastic wave feature matrix [12,20]."""
    base = np.random.normal(0, 0.05, (12, 20)).astype(np.float32)
    if has_collision:
        # Collision channels (rear corner/center sensors ch7~ch10)
        for ch in [7, 8, 9, 10]:
            base[ch] += np.random.normal(collision_strength, 0.1, 20)
    return base.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Frame generators (by scenario profile)
# ─────────────────────────────────────────────────────────────

def gen_frame_parking_hard(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Hard wall behind during reverse parking; rear sensors ch6~ch11.
    Each rear channel sees a fixed world wall point; its range is derived from the
    CAN odometry pose, so the wall stays put as the vehicle reverses."""
    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)

    # Front row (ch0~ch5): Open — no obstacle ahead
    for ch in range(6):
        edi_amplitude[ch]  = 0.02
        edi_confidence[ch] = np.random.randint(0, 15)
        envelopes[ch]      = gen_envelope_open()

    # Rear row (ch6~ch11): Wall — reversing into a fixed barrier
    for ch in range(6, 12):
        P = obstacle_point(ch, PARKING_REAR_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.02, 0.02), 0.1)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.88 + np.random.uniform(-0.04, 0.04)
        edi_confidence[ch] = np.random.randint(85, 99)
        envelopes[ch]      = gen_envelope_hard(d)

    gt_class_ids = np.array([4,4,4,4,4,4, 0,0,0,0,0,0], dtype=np.uint8)
    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids)


def gen_frame_suspended(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Overhead obstacle (height-limit beam): front center/corner sensors ch1~ch4.
    Side sensors ch0/ch5 point sideways → Open. Range from CAN odometry."""
    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)
    gt_class_ids   = np.full(12, 4, dtype=np.uint8)  # default Open

    for ch in [1, 2, 3, 4]:
        P = obstacle_point(ch, SUSPENDED_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.04, 0.04), 0.1)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.60 + np.random.uniform(-0.08, 0.08)
        edi_confidence[ch] = np.random.randint(65, 85)
        envelopes[ch]      = gen_envelope_hard(d, noise_level=0.04)
        gt_class_ids[ch]   = 6  # Overhead

    for ch in [0, 5, 6, 7, 8, 9, 10, 11]:
        envelopes[ch] = gen_envelope_open()

    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids)


def gen_frame_wet_ground(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Wet ground scenario: mirror reflection from puddles, all 12 channels affected.
    Puddle returns modelled as fixed world points so they stay registered."""
    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)
    gt_class_ids   = np.full(12, 8, dtype=np.uint8)  # Wet ground

    for ch in range(12):
        P = obstacle_point(ch, WET_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.05, 0.05), 0.1)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.65 + np.random.uniform(-0.15, 0.15)
        edi_confidence[ch] = np.random.randint(40, 75)
        envelopes[ch]      = gen_envelope_wet(d)

    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids)


def gen_frame_collision(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Collision event scenario: light bump on rear sensors ch7~ch10 at frames 50~70.
    Rear barrier is a fixed world point; range from CAN odometry (reverse)."""
    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)
    gt_class_ids   = np.full(12, 4, dtype=np.uint8)

    has_collision = 50 <= frame_id <= 70
    strength = 0.6 if has_collision else 0.0

    for ch in range(7, 11):
        P = obstacle_point(ch, COLLISION_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.02, 0.02), 0.05)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.90
        edi_confidence[ch] = 95
        envelopes[ch]      = gen_envelope_hard(d, noise_level=0.01)
        gt_class_ids[ch]   = 0  # Wall

    for ch in list(range(7)) + [11]:
        envelopes[ch] = gen_envelope_open()

    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids, has_collision, strength)


def gen_frame_urban_mixed(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Urban mixed scenario.
      ch2(FL-Center), ch3(FR-Center) → Vehicle ahead (fixed world point)
      ch1(FL-Corner), ch4(FR-Corner) → Pedestrian crossing (laterally moving point)
      ch0/ch5/ch6/ch11 → Clutter (side scatter)
      ch7~ch10 → Open"""
    progress = frame_id / max(total_frames - 1, 1)

    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)
    gt_class_ids   = np.full(12, 4, dtype=np.uint8)

    # ch2/ch3: Vehicle ahead — fixed world point, range from odometry
    for ch in [2, 3]:
        P = obstacle_point(ch, URBAN_VEH_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.04, 0.04), 0.2)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.88 + np.random.uniform(-0.04, 0.04)
        edi_confidence[ch] = np.random.randint(88, 98)
        envelopes[ch]      = gen_envelope_hard(d, noise_level=0.02)
        gt_class_ids[ch]   = 1  # Vehicle

    # ch1/ch4: Pedestrian — genuinely moving (world point oscillates laterally)
    ped_off = math.sin(progress * np.pi * 2) * 0.8
    for ch in [1, 4]:
        P0 = obstacle_point(ch, URBAN_PED_D0)
        P = (P0[0], P0[1] + ped_off)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.1, 0.1), 0.3)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.52 + np.random.uniform(-0.08, 0.08)
        edi_confidence[ch] = np.random.randint(60, 80)
        envelopes[ch]      = gen_envelope_soft(d, noise_level=0.04)
        gt_class_ids[ch]   = 2  # Pedestrian

    # ch0/ch5/ch6/ch11: Clutter (near-range side scatter)
    for ch in [0, 5, 6, 11]:
        edi_distance[ch]   = 0.5 + np.random.uniform(0, 0.3)
        edi_amplitude[ch]  = 0.35 + np.random.uniform(-0.05, 0.05)
        edi_confidence[ch] = np.random.randint(20, 45)
        envelopes[ch]      = gen_envelope_ground_clutter(noise_level=0.03)
        gt_class_ids[ch]   = 5  # Clutter

    for ch in [7, 8, 9, 10]:
        envelopes[ch] = gen_envelope_open()

    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids)


def gen_frame_curb_soft(frame_id: int, total_frames: int, pose: tuple) -> tuple:
    """Curb + Soft obstacle scenario.
      ch0/ch1/ch2 → Soft obstacle (cone/bush, fixed world point)
      ch5/ch11    → Curb (right roadside, fixed world point)
      others      → Open"""
    edi_distance   = np.zeros(12, np.float32)
    edi_amplitude  = np.zeros(12, np.float32)
    edi_confidence = np.zeros(12, np.uint8)
    edi_echo_type  = np.zeros(12, np.uint8)
    envelopes      = np.zeros((12, 256), np.float32)
    gt_class_ids   = np.full(12, 4, dtype=np.uint8)

    for ch in [0, 1, 2]:
        P = obstacle_point(ch, CURB_CONE_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.06, 0.06), 0.2)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.50 + np.random.uniform(-0.10, 0.10)
        edi_confidence[ch] = np.random.randint(55, 78)
        envelopes[ch]      = gen_envelope_soft(d, noise_level=0.05)
        gt_class_ids[ch]   = 3  # Soft

    for ch in [5, 11]:
        P = obstacle_point(ch, CURB_D0)
        d = max(range_to(pose, ch, P) + np.random.uniform(-0.03, 0.03), 0.1)
        edi_distance[ch]   = d
        edi_amplitude[ch]  = 0.72 + np.random.uniform(-0.06, 0.06)
        edi_confidence[ch] = np.random.randint(70, 88)
        envelopes[ch]      = gen_envelope_hard(d, noise_level=0.03)
        gt_class_ids[ch]   = 7  # Curb

    for ch in [3, 4, 6, 7, 8, 9, 10]:
        envelopes[ch] = gen_envelope_open()

    return (edi_distance, edi_amplitude, edi_confidence, edi_echo_type,
            envelopes, gt_class_ids)


# ─────────────────────────────────────────────────────────────
# Designed CAN motion profiles (source of truth for each scenario)
# Each returns (speed_kmh, steering_angle_deg, gear) for the given frame.
# Speeds decelerate linearly to ~0 so the ego stops ~0.4 m short of the
# obstacle (APA behaviour) instead of driving through it. Total travel for a
# linear V0->0 ramp over N frames at 10 Hz is ~2.75 * V0 (km/h) metres, so V0
# is tuned to (D0 - 0.4) / 2.75 for each scenario's primary obstacle.
# ─────────────────────────────────────────────────────────────

def _ramp(fid: int, total: int, v0: float) -> float:
    """Linear deceleration v0 -> ~0 over the session (km/h), floored at 0.05."""
    p = fid / max(total - 1, 1)
    return max(0.05, v0 * (1.0 - p))


def can_parking_hard(fid: int, total: int) -> tuple:
    # rear wall D0=4.5 -> travel ~4.1 m -> V0 ~= 1.5
    return (_ramp(fid, total, 1.5), 0.0, "R")


def can_suspended(fid: int, total: int) -> tuple:
    # overhead beam D0=4.0 -> travel ~3.6 m -> V0 ~= 1.3
    return (_ramp(fid, total, 1.3), 0.0, "D")


def can_wet_ground(fid: int, total: int) -> tuple:
    # near-stationary crawl through a puddle (omni clutter)
    return (_ramp(fid, total, 0.4), float(np.random.uniform(-2, 2)), "D")


def can_collision(fid: int, total: int) -> tuple:
    # slow reverse, stops ~0.4 m short of the stopper (D0=1.5 -> V0 ~= 0.4)
    return (_ramp(fid, total, 0.4), 0.0, "R")


def can_urban_mixed(fid: int, total: int) -> tuple:
    # creeping toward a stopped vehicle D0=4.5 -> travel ~4.0 m -> V0 ~= 1.5
    return (_ramp(fid, total, 1.5), 0.0, "D")


def can_curb_soft(fid: int, total: int) -> tuple:
    # approach a cone D0=3.0 -> travel ~2.5 m -> V0 ~= 1.0, gentle weave
    return (_ramp(fid, total, 1.0), float(np.random.uniform(-3, 3)), "D")



# ─────────────────────────────────────────────────────────────
# Session write function
# ─────────────────────────────────────────────────────────────

PROFILE_FUNC = {
    "parking_hard":      gen_frame_parking_hard,
    "suspended_obstacle": gen_frame_suspended,
    "wet_ground":        gen_frame_wet_ground,
    "collision_event":   gen_frame_collision,
    "urban_mixed":       gen_frame_urban_mixed,
    "curb_soft":         gen_frame_curb_soft,
}

PROFILE_CAN = {
    "parking_hard":      can_parking_hard,
    "suspended_obstacle": can_suspended,
    "wet_ground":        can_wet_ground,
    "collision_event":   can_collision,
    "urban_mixed":       can_urban_mixed,
    "curb_soft":         can_curb_soft,
}


def write_session(output_dir: str, scenario: dict, num_frames: int) -> None:
    session_id   = scenario["session_id"]
    session_path = os.path.join(output_dir, session_id)
    os.makedirs(session_path, exist_ok=True)

    profile   = scenario["profile"]
    gen_func  = PROFILE_FUNC[profile]
    can_func  = PROFILE_CAN[profile]
    has_gt    = scenario["has_gt"]

    frames_path = os.path.join(session_path, "esi_frames.bin")
    can_path    = os.path.join(session_path, "can_signals.csv")
    meta_path   = os.path.join(session_path, "session_meta.json")
    gt_path     = os.path.join(session_path, "ground_truth.json")

    ground_truth_frames = {}
    can_rows = []

    # Build the designed CAN profile first, then integrate the ego trajectory so the
    # frame generators can derive obstacle ranges from it (CAN-consistent data).
    can_list = [can_func(fid, num_frames) for fid in range(num_frames)]
    poses = build_trajectory(can_list)

    with open(frames_path, "wb") as f_bin:
        for fid in range(num_frames):
            timestamp_ms = fid * 100.0  # 10Hz -> 100ms/frame

            ret = gen_func(fid, num_frames, poses[fid])
            edi_distance, edi_amplitude, edi_confidence, edi_echo_type, \
                envelopes, gt_class_ids = ret[:6]
            has_collision = ret[6] if len(ret) > 6 else False
            strength      = ret[7] if len(ret) > 7 else 0.0
            speed, steer, gear = can_list[fid]

            elastic = gen_elastic_features(has_collision, strength)

            # Write binary frame
            f_bin.write(struct.pack("<I", fid))
            f_bin.write(struct.pack("<f", timestamp_ms))
            f_bin.write(edi_distance.tobytes())
            f_bin.write(edi_amplitude.tobytes())
            f_bin.write(edi_confidence.tobytes())
            f_bin.write(edi_echo_type.tobytes())
            f_bin.write(envelopes.tobytes())
            f_bin.write(elastic.tobytes())

            # CAN signal row (the designed source of truth for this frame)
            can_rows.append({
                "timestamp_ms": timestamp_ms,
                "speed_kmh": round(speed, 2),
                "steering_angle_deg": round(steer, 2),
                "gear": gear,
            })

            # GT labels
            if has_gt:
                ground_truth_frames[str(fid)] = {
                    "class_ids": gt_class_ids.tolist(),
                    "has_collision": bool(has_collision),
                    "collision_type": 1 if has_collision else 0,
                }

    # Write CAN CSV
    with open(can_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["timestamp_ms", "speed_kmh", "steering_angle_deg", "gear"])
        writer.writeheader()
        writer.writerows(can_rows)

    # Write session_meta.json
    meta = dict(scenario["meta"])
    meta["total_frames"] = num_frames
    meta["session_id"] = session_id
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Write ground_truth.json
    if has_gt:
        gt_data = {
            "session_id": session_id,
            "num_classes": 9,
            "class_names": GT_LABELS,
            "frames": ground_truth_frames,
        }
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)

    print(f"  [OK] {session_id}  ({num_frames} frames, {os.path.getsize(frames_path)/1024:.1f} KB)")


# ─────────────────────────────────────────────────────────────
# Update index.json
# ─────────────────────────────────────────────────────────────

def update_index(output_dir: str) -> None:
    sessions = []
    for entry in sorted(os.listdir(output_dir)):
        if os.path.isdir(os.path.join(output_dir, entry)) and entry.startswith("session_"):
            sessions.append(entry)
    index = {
        "version": "1.0",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "sessions": sessions,
    }
    with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n  index.json updated. Total sessions: {len(sessions)}.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AK2 Simulated Test Data Generator")
    parser.add_argument("--output", default="TestData", help="Output directory (default: TestData)")
    parser.add_argument("--frames", type=int, default=100, help="Frames per session (default: 100)")
    args = parser.parse_args()

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAK2 Mock Data Generator")
    print(f"Output dir : {output_dir}")
    print(f"Frames/session: {args.frames}\n")

    for scenario in SCENARIOS:
        write_session(output_dir, scenario, args.frames)

    update_index(output_dir)
    print("\nDone! Start the simulator with:")
    print("  python sim_main.py\n")


if __name__ == "__main__":
    main()
