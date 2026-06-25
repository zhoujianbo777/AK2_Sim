"""
tools/consolidate_ch_features.py
将 edi_features_ch/ 中已有的逐通道小文件合并为帧级大文件，只需运行一次。

  {sid}_c00.npy  ...  {sid}_c11.npy       → {sid}_feat.npy    shape (12, 20)
  {sid}_c00_m1.npy ... {sid}_c11_m1.npy   → {sid}_m1lbl.npy   shape (12,)

合并后 Dataset 从帧级文件加载（~5700 次 I/O 代替 ~68400 次）。
"""

import os, numpy as np

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "datasets", "processed", "edi_features_ch")

# 收集所有帧ID（去掉 _c{ch:02d} 后缀）
frame_ids = set()
for fname in os.listdir(DIR):
    if fname.endswith(".npy") and "_c" in fname and not fname.endswith("_m1.npy"):
        base = fname[:-4]                           # 去掉 .npy
        parts = base.rsplit("_c", 1)
        if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 2:
            frame_ids.add(parts[0])

frame_ids = sorted(frame_ids)
total = len(frame_ids)
print(f"发现 {total} 个帧，开始合并...", flush=True)

for i, fid in enumerate(frame_ids):
    feat_path  = os.path.join(DIR, f"{fid}_feat.npy")
    lbl_path   = os.path.join(DIR, f"{fid}_m1lbl.npy")
    if os.path.exists(feat_path):           # 已经合并过，跳过
        continue
    feat = np.stack([np.load(os.path.join(DIR, f"{fid}_c{ch:02d}.npy"))
                     for ch in range(12)])  # (12, 20)
    lbls = np.array([float(np.load(os.path.join(DIR, f"{fid}_c{ch:02d}_m1.npy")))
                     for ch in range(12)], dtype=np.float32)  # (12,)
    np.save(feat_path, feat)
    np.save(lbl_path,  lbls)
    if (i + 1) % 1000 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total} 帧已合并", flush=True)

print(f"完成！共生成 {total} 个 _feat.npy 和 {total} 个 _m1lbl.npy 文件")
