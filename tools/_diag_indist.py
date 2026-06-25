"""Temp diag: test M6 on IN-DISTRIBUTION training frames at near range,
to separate 'model weakness' from 'TestData OOD'."""
import sys, math
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train.train_M6 import (SceneUnderstandingCNN, SENSOR_CFG, ENV_DIR,
                            OGM_RANGE_FRONT, OGM_RANGE_SIDE, OGM_RES,
                            OGM_H, OGM_W, VALID_OBJ_CLASSES, build_ogm_gt)

MODEL_PATH = ROOT/"models"/"M6"/"M6_scene_understanding_v2.0.0.pt"
SPLIT = ROOT/"datasets"/"splits"/"M6"/"val.txt"

m = SceneUnderstandingCNN(OGM_H, OGM_W)
m.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
m.eval()

frames = [l.strip() for l in open(SPLIT, encoding="utf-8") if l.strip()]

def load_frame(fid):
    env = np.zeros((12,256), np.float32); anns=[None]*12
    for ch in range(12):
        ep = ENV_DIR/f"{fid}_c{ch:02d}.npy"; ap = ENV_DIR/f"{fid}_c{ch:02d}_ann.npy"
        if ep.exists():
            e = np.load(str(ep)).astype(np.float32); mx=e.max()
            env[ch] = e/mx if mx>1e-6 else e
        if ap.exists():
            a = np.load(str(ap), allow_pickle=True).item()
            d = float(a.get("target_range_m",0.0))
            a["distance_m"]=d; anns[ch]=a
    return env, anns

def obstacles(anns):
    out=[]
    for ch,a in enumerate(anns):
        if a is None: continue
        cls=int(a.get("obstacle_class",4)); d=float(a.get("distance_m",0))
        if cls not in VALID_OBJ_CLASSES or d<=0: continue
        sc=SENSOR_CFG[ch]; yaw=math.radians(sc["yaw_deg"])
        ox=-(sc["y_m"]+d*math.sin(yaw)); oy=sc["x_m"]+d*math.cos(yaw)
        out.append((ch,cls,d,ox,oy))
    return out

def m6_cov(ogm, ox, oy):
    row=int(round((OGM_RANGE_FRONT-oy)/OGM_RES)); col=int(round((ox+OGM_RANGE_SIDE)/OGM_RES))
    if not(0<=row<OGM_H and 0<=col<OGM_W): return -1.0
    r0,r1=max(0,row-1),min(OGM_H,row+2); c0,c1=max(0,col-1),min(OGM_W,col+2)
    return float(ogm[r0:r1,c0:c1].max())

# bucket localization quality by obstacle distance
buckets = {"<1.0":[], "1.0-1.5":[], "1.5-2.0":[], "2.0-3.0":[], ">3.0":[]}
def bkey(d):
    if d<1.0: return "<1.0"
    if d<1.5: return "1.0-1.5"
    if d<2.0: return "1.5-2.0"
    if d<3.0: return "2.0-3.0"
    return ">3.0"

n_checked=0
for fid in frames:
    env, anns = load_frame(fid)
    obs = obstacles(anns)
    if not obs: continue
    x = torch.from_numpy(env).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        ogm = m(x).squeeze().numpy()
    for ch,cls,d,ox,oy in obs:
        cov = m6_cov(ogm, ox, oy)
        buckets[bkey(d)].append(cov)
    n_checked+=1
    if n_checked>=600: break

print(f"checked {n_checked} val frames (in-distribution)")
print("\nM6 coverage AT obstacle cell, bucketed by obstacle distance:")
print(f"{'bucket':10s} {'n':>5s} {'mean_cov':>9s} {'%hit(>0.5)':>11s} {'%miss(<0.18)':>13s}")
for k,v in buckets.items():
    if not v:
        print(f"{k:10s} {0:>5d}"); continue
    a=np.array(v)
    print(f"{k:10s} {len(a):>5d} {a.mean():>9.2f} {100*np.mean(a>0.5):>10.1f}% "
          f"{100*np.mean(a<0.18):>12.1f}%")
