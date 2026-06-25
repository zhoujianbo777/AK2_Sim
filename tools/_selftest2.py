"""
Fine-grained timing of each sub-view in the render pipeline.
Run: python tools/_selftest2.py
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
from views.window_envelope import WindowEnvelope
from views.window_classification import WindowClassification
from views.window_ogm import WindowOGM
from views.window_statistics import WindowStatistics
import numpy as np, json

cfg = ConfigLoader("config.yaml")
dm  = DataManager(cfg)
eng = TraditionalEngine(cfg)
dm.load_session(dm.get_session_list()[0].session_id)
frame  = dm.get_current_frame()
result = eng.process(frame)

def T(label, fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        try:
            fn()
            times.append((time.perf_counter()-t0)*1000)
        except Exception as e:
            print(f"  [ERR] {label}: {e}")
            traceback.print_exc()
            return
    avg = sum(times)/len(times)
    print(f"  {avg:7.1f}ms  {label}  (runs={reps})")

print("=" * 55)
print("W2 envelope")
w2 = WindowEnvelope(cfg, "traditional")
T("w2.update_frame (first)",  lambda: w2.update_frame(frame.envelopes, frame.edi_distance, result.valid_flags), 1)
T("w2.update_frame (steady)", lambda: w2.update_frame(frame.envelopes, frame.edi_distance, result.valid_flags), 3)

print("\nW3 classification")
w3 = WindowClassification(cfg, "traditional")
T("w3.on_result_updated", lambda: w3.on_result_updated(result), 3)

print("\nW4 OGM")
w4 = WindowOGM(cfg)
T("w4.update_ogm (first)",  lambda: w4.update_ogm(result, frame), 1)
T("w4.update_ogm (steady)", lambda: w4.update_ogm(result, frame), 3)

print("\nW6 statistics")
w6 = WindowStatistics(cfg)
w6.set_session("s001", 300, 5, "TRADITIONAL")
gt_raw = json.load(open(f"TestData/{dm.get_session_list()[0].session_id}/ground_truth.json"))
gt = {str(e["frame_id"]): e["channel_labels"] for e in gt_raw.get("frame_labels", [])}
gt_ids = np.array(gt.get("0", [0]*12), dtype="uint8")
T("w6.update_with_gt", lambda: w6.update_with_gt(result, gt_ids), 3)

print("=" * 55)
