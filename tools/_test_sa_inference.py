"""
Quick end-to-end test for M2-SA inference.
Run from workspace root: python tools/_test_sa_inference.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

from modules.config_loader import ConfigLoader
from modules.engine_ai import AIEngine
from modules.data_manager import DataManager

cfg = ConfigLoader("config.yaml")
cfg._data["ai"]["m2_sa_enabled"] = True

engine = AIEngine(cfg)
dm = DataManager(cfg)
dm.load_session("session_20260515_001")

results = []
for i in range(5):
    r = engine.process(dm.get_current_frame())
    results.append(r)
    if i < 4:
        dm.step_forward()

r = results[-1]
print(f"engine_type : {r.engine_type}")
print(f"class_ids   : {r.class_ids}")
print(f"hardness    : {np.round(r.material_hardness, 3)}")
print(f"feat_buf    : front={len(engine._feat_buf_front)}  rear={len(engine._feat_buf_rear)}")

assert r.engine_type == "AI(M1+M2+SA)", f"Wrong engine type: {r.engine_type}"
assert not np.any(np.isnan(r.material_hardness)), "NaN in material_hardness!"
assert not np.any(np.isnan(r.class_probs)), "NaN in class_probs!"
assert r.class_probs.shape == (12, 9), f"Wrong shape: {r.class_probs.shape}"
assert len(engine._feat_buf_front) == 5, f"Front buffer should have 5 frames, got {len(engine._feat_buf_front)}"

# Test reset clears buffers
engine.reset()
assert len(engine._feat_buf_front) == 0, "Buffer should be empty after reset"
print("reset() buffer cleared OK")

print("\nALL CHECKS PASSED")
