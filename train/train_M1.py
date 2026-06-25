"""
M1：快速特征融合 MLP 训练脚本
障碍物有效性检测，输入单路 EDI 特征向量（20维），输出有效性概率（1维）
方案A：逐通道推理，每路探头独立判断，12路共享同一模型权重

文档参考：AK2 AI模型开发说明 §6.2
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

# ── 固定随机种子，保证可复现（文档§6.1）──────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Feature dimension contract: must match modules/feature_extractor.extract_channel_features
N_FEATURES = 20

# ── 网络结构（文档§6.2.1）──────── 方案A：逐通道 20 → 1 ────────────────────
class ObstacleDetectionMLP(nn.Module):
    """\u5355\u8def EDI \u7279\u5f81 \u2192 \u6709\u6548\u6027\u6982\u7387  ({N_FEATURES} \u2192 32 \u2192 16 \u2192 1)"""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_FEATURES, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)   # (B,)


# ── 数据集（从 datasets/processed/edi_features_ch/ 加载 .npy 文件）──────────
class ChannelEDIDataset(Dataset):
    """
    每个样本 ID 格式: {session_id}_f{fid:04d}_c{ch:02d}
    预加载帧级文件 {sid}_feat.npy (12,20) + {sid}_m1lbl.npy (12,)
    仅需读取 N_frames 个文件（代替 N_frames×12 个小文件）
    """

    def __init__(self, split_file: str, features_dir: str):
        with open(split_file) as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]
        # 提取涉及的帧ID，预加载至内存
        frame_ids = sorted(set(sid.rsplit("_c", 1)[0] for sid in self.sample_ids))
        print(f"  Preloading {len(frame_ids)} frames ({len(self.sample_ids)} ch samples)...",
              flush=True)
        self.feat_cache  = {fid: np.load(os.path.join(features_dir, f"{fid}_feat.npy"))
                            for fid in frame_ids}   # {fid: ndarray(12,20)}
        self.label_cache = {fid: np.load(os.path.join(features_dir, f"{fid}_m1lbl.npy"))
                            for fid in frame_ids}   # {fid: ndarray(12,)}
        print(f"  Done.", flush=True)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        frame_sid, ch_str = sid.rsplit("_c", 1)
        ch    = int(ch_str)
        feat  = torch.from_numpy(self.feat_cache[frame_sid][ch].astype(np.float32))
        label = torch.tensor(float(self.label_cache[frame_sid][ch]), dtype=torch.float32)
        return feat, label


# ── 数据增强（文档§5.4.2）────────────────────────────────────────────────────
def augment_edi(x: torch.Tensor) -> torch.Tensor:
    """在线 EDI 特征增强，输入 shape=(20,)"""
    if random.random() < 0.20:
        x = x.clone(); x[:] = 0.0      # 模拟整路传感器故障
    if random.random() < 0.30:
        x = x + torch.randn_like(x) * 0.01
    if random.random() < 0.20:
        x = x * (0.7 + random.random() * 0.3)
    return x


# ── 训练主函数 ────────────────────────────────────────────────────────────────
def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 数据集与数据加载器
    train_ds = ChannelEDIDataset(
        split_file=config["train_split"],
        features_dir=config["features_dir"],
    )
    val_ds = ChannelEDIDataset(
        split_file=config["val_split"],
        features_dir=config["features_dir"],
    )
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=True)

    # 模型与优化器
    model = ObstacleDetectionMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_max_epoch"])

    pos_weight = torch.tensor([config.get("pos_weight", 1.0)]).to(device)
    criterion  = nn.BCELoss(reduction="none")  # 手动加权

    best_val_f1 = 0.0
    patience_counter = 0

    mlflow.set_experiment("AK2_M1_ObstacleDetection")
    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(1, config["max_epochs"] + 1):
            # ── Train ──
            model.train()
            train_loss = 0.0
            for feat, label in train_loader:
                feat = torch.stack([augment_edi(f) for f in feat])
                feat, label = feat.to(device), label.to(device)
                pred = model(feat)                          # (B,)
                w    = label * pos_weight[0] + (1 - label)
                loss = (criterion(pred, label) * w).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()

            # ── Validate ──
            model.eval()
            tp = fp = fn = 0
            with torch.no_grad():
                for feat, label in val_loader:
                    feat, label = feat.to(device), label.to(device)
                    pred = model(feat)
                    pred_bin = (pred >= 0.5).float()
                    tp += ((pred_bin == 1) & (label == 1)).sum().item()
                    fp += ((pred_bin == 1) & (label == 0)).sum().item()
                    fn += ((pred_bin == 0) & (label == 1)).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            miss_rate = fn / (fn + tp + 1e-8)

            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1,
                "val_miss_rate": miss_rate,
            }, step=epoch)

            print(f"[{epoch:3d}/{config['max_epochs']}] "
                  f"loss={train_loss/len(train_loader):.4f}  "
                  f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  MR={miss_rate:.4f}")

            # Early stopping（监控验证集F1，文档§6.2.3）
            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
                torch.save(model.state_dict(), config["save_path"])
                mlflow.pytorch.log_model(model, "best_model")
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_f1", best_val_f1)
        print(f"\nBest val F1: {best_val_f1:.4f}")
        print(f"Model saved to: {config['save_path']}")


# ── 默认配置（文档§6.2.3）────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "train_split":  "datasets/splits/M1/train.txt",
    "val_split":    "datasets/splits/M1/val.txt",
    "features_dir": "datasets/processed/edi_features_ch",
    "save_path":    "models/M1/M1_obstacle_detection_v1.0.0.pt",
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "batch_size":   512,
    "max_epochs":   100,
    "patience":     15,       # Early stopping 耐心值
    "t_max_epoch":  50,       # CosineAnnealingLR T_max
    "pos_weight":   1.0,      # 需根据实际训练集正负样本比例调整
    "num_workers":  0,
}

if __name__ == "__main__":
    train(DEFAULT_CONFIG)
