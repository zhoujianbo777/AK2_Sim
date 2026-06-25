"""
obstacle_classes.py  —  Unified obstacle class taxonomy
Single source of truth for ALL engines and views.

Definition is authoritative; matches:
  - ground_truth.json  channel_labels  (values 0-8)
  - PC端仿真软件设计说明书.md  §5.2.3 "统一障碍物类别规范"

Traditional algorithm maps its 5 internal rules to these 9 IDs via TRAD_TO_UNIFIED.
M1 binary model uses only Wall (0) and Open (4).
Full 9-class classification requires M2 model (DLL / future work).
"""

# ─── Canonical 9-class table ─────────────────────────────────────────────────
# Each entry: (id, english_name, abbreviation, hex_color, chinese_description)
OBSTACLE_TABLE = [
    (0, "Wall",       "Wa", "#607D8B", "墙体/混凝土柱/固定硬质障碍物"),
    (1, "Vehicle",    "Ve", "#2196F3", "车辆"),
    (2, "Pedestrian", "Pe", "#F44336", "行人"),
    (3, "Soft",       "So", "#FF9800", "软质障碍物/植被/锥桶"),
    (4, "Open",       "Op", "#E0E0E0", "空旷/无障碍物"),
    (5, "Clutter",    "Cl", "#FFF176", "地面杂波/多路径干扰"),
    (6, "Overhead",   "Ov", "#9C27B0", "悬空/高位障碍物（不阻挡底盘）"),
    (7, "Curb",       "Cu", "#4CAF50", "路沿/低矮障碍物"),
    (8, "Wet",        "We", "#00BCD4", "湿滑地面/镜面反射"),
]

N_CLASSES: int = len(OBSTACLE_TABLE)   # 9

# Flat lists (index = class_id)
CLASS_IDS:    list[int] = [row[0] for row in OBSTACLE_TABLE]
CLASS_NAMES:  list[str] = [row[1] for row in OBSTACLE_TABLE]
CLASS_ABBR:   list[str] = [row[2] for row in OBSTACLE_TABLE]
CLASS_COLORS: list[str] = [row[3] for row in OBSTACLE_TABLE]
CLASS_ZH:     list[str] = [row[4] for row in OBSTACLE_TABLE]

# Convenience name → id lookup
NAME_TO_ID: dict[str, int] = {row[1]: row[0] for row in OBSTACLE_TABLE}

# Pre-converted RGB tuples for pyqtgraph / Qt painting  (r, g, b)
def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
CLASS_COLORS_RGB: list[tuple] = [_hex_to_rgb(c) for c in CLASS_COLORS]

# Commonly used IDs
ID_WALL       = 0
ID_VEHICLE    = 1
ID_PEDESTRIAN = 2
ID_SOFT       = 3
ID_OPEN       = 4
ID_CLUTTER    = 5
ID_OVERHEAD   = 6
ID_CURB       = 7
ID_WET        = 8

# ─── Object-width prior / regression target ──────────────────────────────────
# Typical physical lateral width per class (metres). Used as the width-head
# regression target during training and as a fallback prior. 0.0 = the class is
# not a discrete reflector with a meaningful width (Open / Wet road surface).
# Index = class_id; must stay aligned with OBSTACLE_TABLE.
CLASS_TYPICAL_WIDTH_M: list[float] = [
    2.50,  # 0 Wall        — wide continuous surface
    1.80,  # 1 Vehicle     — passenger-car width
    0.45,  # 2 Pedestrian  — narrow
    0.55,  # 3 Soft        — bush / cone
    0.00,  # 4 Open        — no object
    0.25,  # 5 Clutter     — small ground scatter
    2.00,  # 6 Overhead    — horizontal beam / door
    0.35,  # 7 Curb        — low linear edge
    0.00,  # 8 Wet         — road condition, not an object
]
# Classes that carry a meaningful physical width — used to mask the width loss
# during training and to gate width output / display at inference.
WIDTH_CLASSES: set[int] = {0, 1, 2, 3, 5, 6, 7}

# ─── Traditional algorithm mapping ───────────────────────────────────────────
# engine_traditional.py uses 5 internal rule labels; map them to unified IDs.
# Internal label → unified class_id
TRAD_TO_UNIFIED: dict[int, int] = {
    0: ID_WALL,     # Hard    → Wall      (强回波硬目标视为墙体类)
    1: ID_SOFT,     # Soft    → Soft      (中等幅度软目标)
    2: ID_OPEN,     # Open    → Open      (无有效回波，空旷)
    3: ID_CLUTTER,  # Clutter → Clutter   (地面短距多路径)
    4: ID_WALL,     # Unknown → Wall      (保守处理：有回波但无法分类，仍视为障碍物)
}

# Internal labels (used only inside engine_traditional.py)
_TRAD_HARD    = 0
_TRAD_SOFT    = 1
_TRAD_OPEN    = 2
_TRAD_CLUTTER = 3
_TRAD_UNKNOWN = 4
