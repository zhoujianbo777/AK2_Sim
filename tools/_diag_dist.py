"""Temp diag: M6 training distance distribution + peak<->range relationship."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ENV_DIR = ROOT/"datasets"/"processed"/"envelopes"
SPLIT = ROOT/"datasets"/"splits"/"M6"/"train.txt"
VALID = {0,1,2,3,6,7,8}
DIST_MAX_M = 6.0

frames = [l.strip() for l in open(SPLIT, encoding="utf-8") if l.strip()]
print("train frames:", len(frames))

dists = []
peak_err = []
sample = frames[::max(1, len(frames)//400)]  # subsample for speed
for fid in sample:
    for ch in range(12):
        ap = ENV_DIR/f"{fid}_c{ch:02d}_ann.npy"
        ep = ENV_DIR/f"{fid}_c{ch:02d}.npy"
        if not ap.exists() or not ep.exists():
            continue
        ann = np.load(str(ap), allow_pickle=True).item()
        cls = int(ann.get("obstacle_class", 4))
        if cls not in VALID:
            continue
        d = float(ann.get("target_range_m", 0.0))
        if d <= 0:
            continue
        dists.append(d)
        env = np.load(str(ep)).astype(np.float32)
        pk = int(np.argmax(env)) / 255.0 * DIST_MAX_M
        peak_err.append(pk - d)

dists = np.array(dists); peak_err = np.array(peak_err)
print(f"\nvalid-obstacle channel samples: {len(dists)}")
print(f"distance  min={dists.min():.2f} max={dists.max():.2f} "
      f"mean={dists.mean():.2f} median={np.median(dists):.2f}")
print("histogram (m):")
bins = [0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0]
h,_ = np.histogram(dists, bins=bins)
for i in range(len(h)):
    pct = 100*h[i]/len(dists)
    print(f"  [{bins[i]:.1f},{bins[i+1]:.1f})  {h[i]:5d}  {pct:5.1f}%  {'#'*int(pct)}")
print(f"\n<2.0m fraction: {100*np.mean(dists<2.0):.1f}%")
print(f"<1.5m fraction: {100*np.mean(dists<1.5):.1f}%")
print(f"\npeak(argmax)->range vs stored target_range_m:")
print(f"  err mean={peak_err.mean():.3f} std={peak_err.std():.3f} "
      f"abs_mean={np.abs(peak_err).mean():.3f}")
