"""
M2: 1D-CNN 障碍物类别分类 训练脚本
====================================
严格按照《AK2 AI模型开发说明》实现：

  §6.3.1  网络结构: 1D-CNN + 3 输出头 (class / hardness / suspension_height)
  §6.3.2  损失函数: L_cls + 0.1×L_hardness + 0.5×L_height (多任务)
  §6.3.3  训练配置: AdamW lr=5e-4, OneCycleLR max_lr=5e-3, batch=128, 150 epochs
  §5.4.1  数据增强: 幅度缩放/时移/高斯噪声/截断补零/拉伸压缩

输入数据: datasets/processed/envelopes/{sample_id}.npy     shape (256,)
标注数据: datasets/processed/envelopes/{sample_id}_ann.npy  dict {obstacle_class, material_hardness, suspension_height_m}
数据划分: datasets/splits/M2/{train,val,test}.txt           行格式 {session_id}_f{fid:04d}_c{ch:02d}
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.obstacle_classes import N_CLASSES, CLASS_NAMES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR       = os.path.join(ROOT, "datasets", "processed", "envelopes")
SPLITS_M2_DIR = os.path.join(ROOT, "datasets", "splits", "M2")


# ====================================================================
# §6.3.1  网络结构：1D-CNN + 3 输出头
# ====================================================================

class ObstacleClassifier1DCNN(nn.Module):
    """
    1D-CNN 障碍物分类器（§6.3.1）

    输入: [B, 1, 256]  归一化包络波形
    输出 (dict):
      class_logits      [B, N_CLASSES]  9 类原始 logits
      hardness          [B, 1]          材质硬度 0~1 (Sigmoid)
      suspension_height [B, 1]          悬空高度 m  (ReLU, 仅 class=6 有意义)
    """

    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),    # 256 -> 128
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),    # 128 ->  64
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),    #  64 ->  32
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier       = nn.Linear(32, num_classes)
        self.hardness_head    = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.height_head      = nn.Sequential(nn.Linear(32, 1), nn.ReLU())

    def forward(self, x: torch.Tensor) -> dict:
        """x: [B, 1, 256] -> dict"""
        x    = self.conv_layers(x)                     # [B, 64, 32]
        x    = self.global_avg_pool(x).squeeze(-1)     # [B, 64]
        feat = self.fc1(x)                             # [B, 32]
        return {
            "class_logits":      self.classifier(feat),
            "hardness":          self.hardness_head(feat),
            "suspension_height": self.height_head(feat),
        }


# ====================================================================
# §5.4.1  数据增强（作用于 256 点包络，训练期在线执行）
# ====================================================================

def augment_envelope(env: torch.Tensor) -> torch.Tensor:
    """
    输入/输出均为 shape [256] 的 1D float tensor。

    增强策略（依照文档 §5.4.1）：
      1. 幅度缩放 ×(0.85~1.15), p=30%
      2. 时移 ±5 采样点,         p=20%
      3. 高斯噪声 σ=0.01~0.03,   p=40%
      4. 截断+补零 5~20 pt,      p=15%
      5. 时间轴拉伸/压缩 0.9~1.1×, p=10%
    """
    env = env.clone()
    if random.random() < 0.30:
        env = env * random.uniform(0.85, 1.15)
    if random.random() < 0.20:
        shift = random.randint(-5, 5)
        env = torch.roll(env, shift, dims=0)
        if shift > 0:
            env[:shift] = 0.0
        elif shift < 0:
            env[shift:] = 0.0
    if random.random() < 0.40:
        sigma = random.uniform(0.01, 0.03)
        env = env + torch.randn_like(env) * sigma
    if random.random() < 0.15:
        n = random.randint(5, 20)
        env[-n:] = 0.0
    if random.random() < 0.10:
        scale   = random.uniform(0.9, 1.1)
        old_len = env.shape[0]
        new_len = max(1, int(old_len * scale))
        env_np  = env.numpy()
        src_idx = np.linspace(0, old_len - 1, new_len)
        dst_idx = np.arange(old_len)
        new_env = np.interp(dst_idx,
                            np.linspace(0, old_len - 1, new_len),
                            np.interp(src_idx, np.arange(old_len), env_np))
        env = torch.from_numpy(new_env.astype(np.float32))
    return torch.clamp(env, 0.0, 1.0)


# ====================================================================
# Dataset
# ====================================================================

class EnvelopeDataset(Dataset):
    """
    加载包络数据用于 M2 1D-CNN 训练。

    split_file 行格式: {session_id}_f{fid:04d}_c{ch:02d}
    每条样本返回:
      env         Tensor[1, 256]   归一化包络（已加 channel 维）
      cls         LongTensor       障碍物类别 0~8
      hardness    FloatTensor      材质硬度 0~1
      height      FloatTensor      悬空高度(m), 非 class=6 时为 0.0
    """

    def __init__(self, split_file: str, env_dir: str, augment: bool = False):
        with open(split_file) as f:
            self.sample_ids = [ln.strip() for ln in f if ln.strip()]
        self.env_dir = env_dir
        self.augment = augment

        print(f"  Preloading {len(self.sample_ids)} envelope samples ...", flush=True)
        self.env_cache: dict = {}
        self.ann_cache: dict = {}
        for sid in self.sample_ids:
            env_path = os.path.join(env_dir, f"{sid}.npy")
            ann_path = os.path.join(env_dir, f"{sid}_ann.npy")
            self.env_cache[sid] = np.load(env_path).astype(np.float32)
            ann = np.load(ann_path, allow_pickle=True).item()
            self.ann_cache[sid] = (
                int(ann["obstacle_class"]),
                float(ann["material_hardness"]),
                float(ann["suspension_height_m"]),
            )
        print("  Done.", flush=True)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]
        env = torch.from_numpy(self.env_cache[sid])   # [256]
        if self.augment:
            env = augment_envelope(env)
        env = env.unsqueeze(0)                         # [1, 256]
        cls, hardness, height = self.ann_cache[sid]
        return (
            env,
            torch.tensor(cls,      dtype=torch.long),
            torch.tensor(hardness, dtype=torch.float32),
            torch.tensor(height,   dtype=torch.float32),
        )


# ====================================================================
# 类别权重（抑制 Open 类主导）
# ====================================================================

def compute_class_weights(split_file: str, env_dir: str) -> torch.Tensor:
    with open(split_file) as f:
        sample_ids = [ln.strip() for ln in f if ln.strip()]
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for sid in sample_ids:
        ann_path = os.path.join(env_dir, f"{sid}_ann.npy")
        ann = np.load(ann_path, allow_pickle=True).item()
        counts[int(ann["obstacle_class"])] += 1
    freq    = counts / counts.sum()
    weights = np.where(freq > 0, 1.0 / (freq + 1e-8), 0.0)
    weights = weights / weights.mean()
    print(f"  Class counts : {counts}")
    print(f"  Class weights: {np.round(weights, 2)}")
    return torch.tensor(weights, dtype=torch.float32)


# ====================================================================
# §6.3.2  多任务损失
# ====================================================================

def compute_loss(
    out:         dict,
    labels:      torch.Tensor,
    hardness_gt: torch.Tensor,
    height_gt:   torch.Tensor,
    cls_criterion,
    lambda1: float = 0.1,
    lambda2: float = 0.5,
) -> torch.Tensor:
    """
    L_total = L_cls
            + lambda1 * L_hardness  (仅高置信度样本 class != 4-Open)
            + lambda2 * L_height    (仅 class = 6 Overhead 样本)
    """
    L_cls = cls_criterion(out["class_logits"], labels)
    non_open   = labels != 4
    L_hardness = torch.tensor(0.0, device=labels.device)
    if non_open.any():
        pred_h = out["hardness"][non_open].squeeze(-1)
        gt_h   = hardness_gt[non_open]
        L_hardness = nn.functional.mse_loss(pred_h, gt_h)
    overhead = labels == 6
    L_height = torch.tensor(0.0, device=labels.device)
    if overhead.any():
        pred_ht = out["suspension_height"][overhead].squeeze(-1)
        gt_ht   = height_gt[overhead]
        L_height = nn.functional.mse_loss(pred_ht, gt_ht)
    return L_cls + lambda1 * L_hardness + lambda2 * L_height


# ====================================================================
# §6.3.3  训练主函数
# ====================================================================

def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training M2 1D-CNN on: {device}")

    train_ds = EnvelopeDataset(config["train_split"], config["env_dir"], augment=True)
    val_ds   = EnvelopeDataset(config["val_split"],   config["env_dir"], augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    model = ObstacleClassifier1DCNN(num_classes=N_CLASSES).to(device)

    print("Computing class weights from training set ...")
    class_weights = compute_class_weights(config["train_split"], config["env_dir"]).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["max_lr"],
        epochs=config["max_epochs"],
        steps_per_epoch=len(train_loader),
    )

    use_amp = torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_macro   = 0.0
    patience_counter = 0
    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)

    mlflow.set_experiment("AK2_M2_1DCNN_ObstacleClassification")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in config.items()
                           if not isinstance(v, (list, dict))})

        for epoch in range(1, config["max_epochs"] + 1):

            # -- Train --
            model.train()
            train_loss = 0.0
            for env, labels, hardness_gt, height_gt in train_loader:
                env         = env.to(device)
                labels      = labels.to(device)
                hardness_gt = hardness_gt.to(device)
                height_gt   = height_gt.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out  = model(env)
                    loss = compute_loss(out, labels, hardness_gt, height_gt, cls_criterion)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

            # -- Validate --
            model.eval()
            correct     = 0
            total       = 0
            cls_correct = np.zeros(N_CLASSES, dtype=np.int64)
            cls_total   = np.zeros(N_CLASSES, dtype=np.int64)
            with torch.no_grad():
                for env, labels, _, _ in val_loader:
                    env, labels = env.to(device), labels.to(device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(env)
                    pred     = out["class_logits"].argmax(dim=1)
                    correct += (pred == labels).sum().item()
                    total   += labels.size(0)
                    for c in range(N_CLASSES):
                        mask = labels == c
                        cls_total[c]   += mask.sum().item()
                        cls_correct[c] += (pred[mask] == c).sum().item()

            val_acc   = correct / (total + 1e-8)
            per_cls   = np.where(cls_total > 0, cls_correct / cls_total, np.nan)
            macro_acc = float(np.nanmean(per_cls))

            mlflow.log_metrics({
                "train_loss":    train_loss / len(train_loader),
                "val_acc":       val_acc,
                "val_macro_acc": macro_acc,
            }, step=epoch)

            cls_str = "  ".join(
                f"{CLASS_NAMES[c][:4]}={per_cls[c]:.2f}" if not np.isnan(per_cls[c])
                else f"{CLASS_NAMES[c][:4]}=--"
                for c in range(N_CLASSES)
            )
            print(f"[{epoch:3d}/{config['max_epochs']}] "
                  f"loss={train_loss / len(train_loader):.4f}  "
                  f"acc={val_acc:.4f}  macro={macro_acc:.4f}")
            print(f"  {cls_str}")

            if macro_acc > best_val_macro:
                best_val_macro   = macro_acc
                patience_counter = 0
                torch.save(model.state_dict(), config["save_path"])
                mlflow.pytorch.log_model(model, "best_model")
                print(f"  -> Saved best model  (macro={best_val_macro:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        mlflow.log_metric("best_val_macro_acc", best_val_macro)
        print(f"\nBest val macro accuracy: {best_val_macro:.4f}")
        print(f"Model saved to: {config['save_path']}")


DEFAULT_CONFIG = {
    "train_split":  "datasets/splits/M2/train.txt",
    "val_split":    "datasets/splits/M2/val.txt",
    "env_dir":      "datasets/processed/envelopes",
    "save_path":    "models/M2/M2_obstacle_classifier_v1.0.0.pt",
    "lr":           5e-4,
    "max_lr":       5e-3,
    "weight_decay": 1e-4,
    "batch_size":   128,
    "max_epochs":   150,
    "patience":     20,
    "num_workers":  0,
}

if __name__ == "__main__":
    train(DEFAULT_CONFIG)
