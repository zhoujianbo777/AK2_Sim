"""
Self-test: headless timing of the full session-load + render pipeline.
Run: python tools/_selftest.py
"""
import os, sys, time, json, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)

from modules.config_loader import ConfigLoader
from modules.data_manager import DataManager
from modules.engine_traditional import TraditionalEngine
from modules.result_fusion import ResultFusion
from views.window_display import DisplayWindow
from views.window_control import WindowControl
import numpy as np

cfg  = ConfigLoader("config.yaml")
dm   = DataManager(cfg)
eng  = TraditionalEngine(cfg)

def T(label, fn):
    t0 = time.perf_counter()
    try:
        ret = fn()
        ms = (time.perf_counter() - t0) * 1000
        print(f"[OK  {ms:7.1f}ms]  {label}")
        return ret
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000
        print(f"[ERR {ms:7.1f}ms]  {label}")
        traceback.print_exc()
        return None

print("=" * 60)
print("Step 1 – create windows")
w1   = T("WindowControl()", lambda: WindowControl(cfg, dm))
disp = T("DisplayWindow()", lambda: DisplayWindow(cfg, "traditional"))

print("\nStep 2 – load session")
sessions = dm.get_session_list()
print(f"  sessions found: {len(sessions)}")
ok = T(f"load_session({sessions[0].session_id})",
        lambda: dm.load_session(sessions[0].session_id))
print(f"  frames loaded: {dm.get_frame_count()}")

print("\nStep 3 – set_session (reset all sub-views)")
T("disp.set_session()", lambda: disp.set_session(
    sessions[0].session_id, dm.get_frame_count(), 5, "TRADITIONAL"))

print("\nStep 4 – engine.process frame 0")
frame  = dm.get_current_frame()
result = T("engine.process(frame0)", lambda: eng.process(frame))

print("\nStep 5 – display.update_frame")
T("disp.update_frame()", lambda: disp.update_frame(frame, result))

print("\nStep 6 – update_with_gt")
gt_raw = json.load(open(f"TestData/{sessions[0].session_id}/ground_truth.json"))
gt = {str(e["frame_id"]): e["channel_labels"] for e in gt_raw.get("frame_labels", [])}
gt_ids = np.array(gt.get("0", [0]*12), dtype="uint8")
T("disp.update_with_gt()", lambda: disp.update_with_gt(result, gt_ids))

print("\nStep 7 – simulate 5 timer ticks")
for i in range(5):
    dm.step_forward()
    frame2 = dm.get_current_frame()
    result2 = eng.process(frame2)
    T(f"tick {i+1}: update_frame (frame {dm.get_current_index()})",
      lambda f=frame2, r=result2: disp.update_frame(f, r))

print("\nStep 8 – W1 update_frame_display")
T("w1.update_frame_display()", lambda: w1.update_frame_display(
    0, 0.0, 0.0, 0.0, "D"))

print("\n" + "=" * 60)
print("Self-test complete.")
