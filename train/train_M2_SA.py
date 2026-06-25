"""
train_M2_SA.py  —  M2-SA 组内空间注意力模块训练（§6.4）

依赖:
  - 已训练的 M2-SC 权重: models/M2/M2_obstacle_classifier_v1.0.0.pt
    （先运行 train/train_M2.py）
  - M2 数据集:
      datasets/splits/M2/{train,val}.txt
      datasets/processed/envelopes/{sid}.npy
      datasets/processed/envelopes/{sid}_ann.npy

两阶段训练（§6.4.4）：
  Stage 1 (20 epochs):  冻结 M2-SC 骨干，仅训练 M2-SA 全部层
  Stage 2 (80 epochs):  解冻全部，联合微调（线性 warmup 5 epoch + cosine 退火）

损失（§6.4.3）：
  Stage 1: L_cls_SA + 0.1 × L_hardness + 0.5 × L_height
  Stage 2: L_cls_SA + 0.3 × L_cls_SC + 0.1 × L_hardness + 0.5 × L_height

输出: models/M2/M2_spatial_attention_v1.0.0.pt
"""

import os
import re
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch

# ── 确保项目根目录在 sys.path 中 ─────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.obstacle_classes import N_CLASSES, CLASS_NAMES, CLASS_TYPICAL_WIDTH_M, WIDTH_CLASSES


# ====================================================================
# §6.3.1  M2-SC 骨干（必须与 modules/engine_ai.py 保持完全一致）
# ====================================================================

class ObstacleClassifier1DCNN(nn.Module):
    """M2-SC: 1D-CNN 骨干分类器 (§6.3.1)"""

    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1,  16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),    # 256 → 128
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),    # 128 →  64
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),    #  64 →  32
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier       = nn.Linear(32, num_classes)
        self.hardness_head    = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.height_head      = nn.Sequential(nn.Linear(32, 1), nn.ReLU())

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """x: [B, 1, 256] → dict, or [B, 32] feat if return_features=True"""
        x    = self.conv_layers(x)
        x    = self.global_avg_pool(x).squeeze(-1)   # [B, 64]
        feat = self.fc1(x)                           # [B, 32]
        if return_features:
            return feat
        return {
            "class_logits":      self.classifier(feat),
            "hardness":          self.hardness_head(feat),
            "suspension_height": self.height_head(feat),
        }


# ====================================================================
# §6.4.2  M2-SA 网络结构（必须与 modules/engine_ai.py 保持完全一致）
# ====================================================================

class M2GroupAttention(nn.Module):
    """
    M2-SA: 组内空间注意力精化模块（§6.4.2）

    单帧模式 (T=1):
        输入  feat:      [B, N, 32]     N = 组内传感器数（前/后各 6 路）
        输入  positions: [B, N] int     组内位置编号 0~5
        输出  dict:      class_logits [B,N,9] / hardness [B,N,1] / suspension_height [B,N,1]

    多帧模式 (T>1):
        输入  feat_seq:  [B, T, N, 32]  T 帧历史特征序列
        内部先执行时序融合 → [B, N, 32] → 再执行空间注意力
    """

    def __init__(self, feat_dim: int = 32, num_classes: int = N_CLASSES,
                 num_positions: int = 6):
        super().__init__()
        self.pos_emb        = nn.Embedding(num_positions, feat_dim)
        self.spatial_attn   = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=4, batch_first=True, dropout=0.1
        )
        self.spatial_norm   = nn.LayerNorm(feat_dim)
        self.temporal_score = nn.Linear(feat_dim, 1)
        self.cls_head       = nn.Linear(feat_dim, num_classes)
        self.hardness_head  = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid())
        self.height_head    = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())
        self.width_head     = nn.Sequential(nn.Linear(feat_dim, 1), nn.ReLU())

    def _temporal_fusion(self, feat_seq: torch.Tensor,
                         seq_padding_mask=None) -> torch.Tensor:
        """feat_seq: [B, T, N, 32] → [B, N, 32]

        seq_padding_mask: optional [B, T, N] bool, True = padding channel in that
        frame; masked frames are excluded from the per-channel temporal softmax.
        Default None preserves the original behaviour.
        """
        B, T, N, D = feat_seq.shape
        scores    = self.temporal_score(feat_seq).squeeze(-1)      # [B, T, N]
        time_bias = torch.linspace(
            0.0, 1.0, T, device=feat_seq.device
        ).view(1, T, 1)
        scores = scores + time_bias
        if seq_padding_mask is not None:
            scores = scores.masked_fill(seq_padding_mask, -1e9)
        weights   = torch.softmax(scores, dim=1)                   # [B, T, N]
        return (feat_seq * weights.unsqueeze(-1)).sum(dim=1)        # [B, N, 32]

    def forward(self, feat: torch.Tensor, positions: torch.Tensor,
                padding_mask=None, feat_seq=None, seq_padding_mask=None) -> dict:
        """
        feat:         [B, N, 32]
        positions:    [B, N] int  (0~5)
        padding_mask: [B, N] bool  True = padding channel (ignored in attention)
        feat_seq:     [B, T, N, 32]  multi-frame (optional)
        """
        if feat_seq is not None:
            feat = self._temporal_fusion(feat_seq, seq_padding_mask)
        x = feat + self.pos_emb(positions)
        attn_out, _ = self.spatial_attn(x, x, x, key_padding_mask=padding_mask)
        x = self.spatial_norm(x + attn_out)
        return {
            "class_logits":      self.cls_head(x),       # [B, N, 9]
            "hardness":          self.hardness_head(x),  # [B, N, 1]
            "suspension_height": self.height_head(x),    # [B, N, 1]
            "object_width":      self.width_head(x),      # [B, N, 1]
        }


# ====================================================================
# 数据集：按帧分组（前排 ch0-5 / 后排 ch6-11）
# ====================================================================

class GroupEnvelopeDataset(Dataset):
    """
    将 M2 的逐通道样本（split_file 行格式: {session_id}_f{fid:04d}_c{ch:02d}）
    按帧和传感器组聚合。

    每条样本 = 一个6路组（前排 OR 后排），返回:
        envs:     Tensor [6, 1, 256]   包络波形
        labels:   Tensor [6]           障碍物类别 0~8
        hardness: Tensor [6]           材质硬度 0~1
        height:   Tensor [6]           悬空高度 m（非 class=6 时为 0.0）
        width:    Tensor [6]           物体宽度 m（非物体类时为 0.0）
    """

    # 传感器位置编号在两组内均为 0~5
    _POSITIONS = list(range(6))

    def __init__(self, split_file: str, env_dir: str):
        with open(split_file) as f:
            raw_ids = [ln.strip() for ln in f if ln.strip()]

        # 按 (frame_prefix, is_rear) 分组，is_rear=True 代表后排 ch6-11
        pat = re.compile(r'^(.+_f\d{4})_c(\d{2})$')
        frame_groups: dict[tuple, dict[int, str]] = {}
        for sid in raw_ids:
            m = pat.match(sid)
            if not m:
                continue
            frame_prefix = m.group(1)
            ch           = int(m.group(2))
            is_rear      = ch >= 6
            local_pos    = ch - (6 if is_rear else 0)
            key          = (frame_prefix, is_rear)
            frame_groups.setdefault(key, {})[local_pos] = sid

        # 仅保留完整的 6 路组
        self.groups: list[list[str]] = []
        for key, pos_map in frame_groups.items():
            if len(pos_map) == 6:
                self.groups.append([pos_map[p] for p in range(6)])

        # 预加载包络与标注
        all_sids = {sid for g in self.groups for sid in g}
        print(
            f"  Preloading {len(all_sids)} samples "
            f"for {len(self.groups)} groups from {split_file} ...",
            flush=True,
        )
        self.env_cache: dict[str, np.ndarray] = {}
        self.ann_cache: dict[str, tuple]       = {}
        for sid in all_sids:
            self.env_cache[sid] = np.load(
                os.path.join(env_dir, f"{sid}.npy")
            ).astype(np.float32)
            ann = np.load(
                os.path.join(env_dir, f"{sid}_ann.npy"), allow_pickle=True
            ).item()
            _cls = int(ann["obstacle_class"])
            self.ann_cache[sid] = (
                _cls,
                float(ann["material_hardness"]),
                float(ann["suspension_height_m"]),
                # 兼容旧数据：缺 object_width_m 时回退到类别典型宽度
                float(ann.get("object_width_m", CLASS_TYPICAL_WIDTH_M[_cls])),
            )
        print("  Done.", flush=True)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        sids     = self.groups[idx]
        envs_np  = np.stack([self.env_cache[s] for s in sids])  # [6, 256]
        envs     = torch.from_numpy(envs_np).unsqueeze(1)        # [6, 1, 256]

        cls_list, h_list, ht_list, w_list = zip(*[self.ann_cache[s] for s in sids])
        labels   = torch.tensor(cls_list, dtype=torch.long)
        hardness = torch.tensor(h_list,   dtype=torch.float32)
        height   = torch.tensor(ht_list,  dtype=torch.float32)
        width    = torch.tensor(w_list,   dtype=torch.float32)
        return envs, labels, hardness, height, width


# ====================================================================
# §6.4.3  损失函数
# ====================================================================

def compute_sa_loss(
    sa_out:        dict,           # M2-SA 输出
    sc_logits:     torch.Tensor,   # M2-SC class_logits [B*N, 9]（Stage 2 使用）
    labels_flat:   torch.Tensor,   # [B*N]
    hardness_flat: torch.Tensor,   # [B*N]
    height_flat:   torch.Tensor,   # [B*N]
    width_flat:    torch.Tensor,   # [B*N]
    cls_criterion,
    stage: int = 1,
) -> tuple:
    """
    Stage 1: L_cls_SA + 0.1 × L_hardness + 0.5 × L_height + 0.3 × L_width
    Stage 2: + 0.3 × L_cls_SC
    """
    BN = labels_flat.shape[0]

    # Flatten SA outputs [B, N, ...] → [B*N, ...]
    sa_logits  = sa_out["class_logits"].reshape(BN, N_CLASSES)         # [B*N, 9]
    sa_hard    = sa_out["hardness"].squeeze(-1).reshape(BN)             # [B*N]
    sa_ht      = sa_out["suspension_height"].squeeze(-1).reshape(BN)   # [B*N]
    sa_wd      = sa_out["object_width"].squeeze(-1).reshape(BN)        # [B*N]

    L_cls_SA = cls_criterion(sa_logits, labels_flat)

    non_open   = labels_flat != 4
    L_hardness = torch.tensor(0.0, device=labels_flat.device)
    if non_open.any():
        L_hardness = nn.functional.mse_loss(sa_hard[non_open], hardness_flat[non_open])

    overhead = labels_flat == 6
    L_height = torch.tensor(0.0, device=labels_flat.device)
    if overhead.any():
        L_height = nn.functional.mse_loss(sa_ht[overhead], height_flat[overhead])

    # 宽度回归：仅对具有物理宽度的物体类（WIDTH_CLASSES）计算
    width_mask = torch.zeros_like(labels_flat, dtype=torch.bool)
    for _wc in WIDTH_CLASSES:
        width_mask |= (labels_flat == _wc)
    L_width = torch.tensor(0.0, device=labels_flat.device)
    if width_mask.any():
        L_width = nn.functional.mse_loss(sa_wd[width_mask], width_flat[width_mask])

    L_cls_SC = torch.tensor(0.0, device=labels_flat.device)
    if stage == 2:
        L_cls_SC = cls_criterion(sc_logits, labels_flat)

    total = L_cls_SA + 0.3 * L_cls_SC + 0.1 * L_hardness + 0.5 * L_height + 0.3 * L_width
    return total, {
        "L_cls_SA":   L_cls_SA.item(),
        "L_cls_SC":   L_cls_SC.item(),
        "L_hardness": L_hardness.item(),
        "L_height":   L_height.item(),
        "L_width":    L_width.item(),
    }


# ====================================================================
# 类别权重（与 train_M2.py 保持一致）
# ====================================================================

def compute_class_weights(split_file: str, env_dir: str) -> torch.Tensor:
    with open(split_file) as f:
        sample_ids = [ln.strip() for ln in f if ln.strip()]
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for sid in sample_ids:
        ann = np.load(
            os.path.join(env_dir, f"{sid}_ann.npy"), allow_pickle=True
        ).item()
        counts[int(ann["obstacle_class"])] += 1
    freq    = counts / counts.sum()
    weights = np.where(freq > 0, 1.0 / (freq + 1e-8), 0.0)
    weights = weights / weights.mean()
    print(f"  Class counts : {counts}")
    print(f"  Class weights: {np.round(weights, 2)}")
    return torch.tensor(weights, dtype=torch.float32)


# ====================================================================
# 评估（SC baseline vs SA 精化，同时计算）
# ====================================================================

def evaluate(
    sc_model: ObstacleClassifier1DCNN,
    sa_model: M2GroupAttention,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """返回 (sc_acc, sa_acc)"""
    sc_model.eval()
    sa_model.eval()
    sc_correct = sa_correct = total = 0
    positions = torch.arange(6, dtype=torch.long).unsqueeze(0).to(device)  # [1, 6]

    with torch.no_grad():
        for envs, labels, _, _, _ in val_loader:
            envs   = envs.to(device)    # [B, 6, 1, 256]
            labels = labels.to(device)  # [B, 6]
            B      = envs.shape[0]

            envs_flat = envs.view(B * 6, 1, 256)
            feats     = sc_model(envs_flat, return_features=True)   # [B*6, 32]
            sc_pred   = sc_model.classifier(feats).argmax(dim=1).view(B, 6)

            feats_grp = feats.view(B, 6, 32)
            pos_exp   = positions.expand(B, -1)
            sa_out    = sa_model(feats_grp, pos_exp)
            sa_pred   = sa_out["class_logits"].argmax(dim=-1)       # [B, 6]

            sc_correct += (sc_pred == labels).sum().item()
            sa_correct += (sa_pred == labels).sum().item()
            total      += labels.numel()

    return sc_correct / max(total, 1), sa_correct / max(total, 1)


# ====================================================================
# 训练主函数
# ====================================================================

def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training M2-SA on: {device}")

    # ── 数据集 ────────────────────────────────────────────────────
    train_ds = GroupEnvelopeDataset(config["train_split"], config["env_dir"])
    val_ds   = GroupEnvelopeDataset(config["val_split"],   config["env_dir"])
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )

    # ── 载入 M2-SC 骨干 ──────────────────────────────────────────
    sc_model = ObstacleClassifier1DCNN(num_classes=N_CLASSES).to(device)
    sc_ckpt  = config["sc_model_path"]
    if not os.path.exists(sc_ckpt):
        print(f"ERROR: M2-SC weights not found: {sc_ckpt}")
        print("Please run train/train_M2.py first.")
        return
    sc_model.load_state_dict(
        torch.load(sc_ckpt, map_location=device, weights_only=True)
    )
    print(f"M2-SC backbone loaded from: {sc_ckpt}")

    # ── M2-SA 模型 ────────────────────────────────────────────────
    sa_model = M2GroupAttention(
        feat_dim=32, num_classes=N_CLASSES, num_positions=6
    ).to(device)
    sa_param_count = sum(p.numel() for p in sa_model.parameters())
    print(f"M2-SA parameters: {sa_param_count:,}")

    # ── 类别权重 + 损失 ──────────────────────────────────────────
    print("Computing class weights ...")
    class_weights = compute_class_weights(
        config["train_split"], config["env_dir"]
    ).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    os.makedirs(os.path.dirname(config["save_path"]), exist_ok=True)
    positions = torch.arange(6, dtype=torch.long).unsqueeze(0).to(device)  # [1, 6]
    use_amp   = torch.cuda.is_available()
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_sa_acc = 0.0

    mlflow.set_experiment("AK2_M2SA_SpatialAttention")
    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in config.items()
                           if not isinstance(v, (list, dict))})

        # ─────────────────────────────────────────────────────────
        # Stage 1: 冻结 SC 骨干，仅训练 M2-SA（20 epochs）
        # ─────────────────────────────────────────────────────────
        print("\n=== Stage 1: Freeze SC backbone, train M2-SA only ===")
        for p in sc_model.parameters():
            p.requires_grad_(False)
        sc_model.eval()
        sa_model.train()

        optimizer_s1 = torch.optim.AdamW(
            sa_model.parameters(),
            lr=config["stage1_lr"], weight_decay=config["weight_decay"],
        )
        scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s1, T_max=config["stage1_epochs"]
        )

        for epoch in range(1, config["stage1_epochs"] + 1):
            sa_model.train()
            train_loss = 0.0

            for envs, labels, hardness_gt, height_gt, width_gt in train_loader:
                envs        = envs.to(device)          # [B, 6, 1, 256]
                labels      = labels.to(device)        # [B, 6]
                hardness_gt = hardness_gt.to(device)
                height_gt   = height_gt.to(device)
                width_gt    = width_gt.to(device)
                B           = envs.shape[0]

                # SC backbone frozen: use no_grad for efficiency
                with torch.no_grad():
                    envs_flat = envs.view(B * 6, 1, 256)
                    feats     = sc_model(envs_flat, return_features=True)  # [B*6, 32]
                    sc_logits = sc_model.classifier(feats)                 # [B*6, 9]

                feats_grp     = feats.view(B, 6, 32)
                pos_exp       = positions.expand(B, -1)
                labels_flat   = labels.view(B * 6)
                hardness_flat = hardness_gt.view(B * 6)
                height_flat   = height_gt.view(B * 6)
                width_flat    = width_gt.view(B * 6)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    sa_out = sa_model(feats_grp, pos_exp)
                    loss, _ = compute_sa_loss(
                        sa_out, sc_logits, labels_flat,
                        hardness_flat, height_flat, width_flat, cls_criterion, stage=1,
                    )

                optimizer_s1.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer_s1)
                scaler.update()
                train_loss += loss.item()

            scheduler_s1.step()
            sc_acc, sa_acc = evaluate(sc_model, sa_model, val_loader, device)
            avg_loss = train_loss / len(train_loader)
            print(
                f"[S1 {epoch:3d}/{config['stage1_epochs']}]"
                f"  loss={avg_loss:.4f}  SC={sc_acc:.4f}  SA={sa_acc:.4f}"
            )
            mlflow.log_metrics(
                {"s1_train_loss": avg_loss, "s1_sc_acc": sc_acc, "s1_sa_acc": sa_acc},
                step=epoch,
            )

        # ─────────────────────────────────────────────────────────
        # Stage 2: 解冻全部，联合微调（80 epochs）
        # ─────────────────────────────────────────────────────────
        print("\n=== Stage 2: Unfreeze all, joint fine-tuning ===")
        for p in sc_model.parameters():
            p.requires_grad_(True)

        optimizer_s2 = torch.optim.AdamW(
            list(sc_model.parameters()) + list(sa_model.parameters()),
            lr=config["stage2_lr"], weight_decay=config["weight_decay"],
        )

        # Linear warmup (5 ep) + cosine annealing
        warmup_epochs = 5

        def _lr_lambda(ep: int) -> float:
            if ep < warmup_epochs:
                return (ep + 1) / warmup_epochs
            prog = (ep - warmup_epochs) / max(
                config["stage2_epochs"] - warmup_epochs, 1
            )
            return 0.5 * (1.0 + np.cos(np.pi * prog))

        scheduler_s2  = torch.optim.lr_scheduler.LambdaLR(optimizer_s2, _lr_lambda)
        patience_ctr  = 0

        for epoch in range(1, config["stage2_epochs"] + 1):
            sc_model.train()
            sa_model.train()
            train_loss = 0.0

            for envs, labels, hardness_gt, height_gt, width_gt in train_loader:
                envs        = envs.to(device)
                labels      = labels.to(device)
                hardness_gt = hardness_gt.to(device)
                height_gt   = height_gt.to(device)
                width_gt    = width_gt.to(device)
                B           = envs.shape[0]

                envs_flat     = envs.view(B * 6, 1, 256)
                feats         = sc_model(envs_flat, return_features=True)  # [B*6, 32]
                sc_logits     = sc_model.classifier(feats)                 # [B*6, 9]
                feats_grp     = feats.view(B, 6, 32)
                pos_exp       = positions.expand(B, -1)
                labels_flat   = labels.view(B * 6)
                hardness_flat = hardness_gt.view(B * 6)
                height_flat   = height_gt.view(B * 6)
                width_flat    = width_gt.view(B * 6)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    sa_out = sa_model(feats_grp, pos_exp)
                    loss, loss_parts = compute_sa_loss(
                        sa_out, sc_logits, labels_flat,
                        hardness_flat, height_flat, width_flat, cls_criterion, stage=2,
                    )

                optimizer_s2.zero_grad()
                scaler.scale(loss).backward()
                # Gradient clipping prevents instability when unfreezing SC
                torch.nn.utils.clip_grad_norm_(
                    list(sc_model.parameters()) + list(sa_model.parameters()), 1.0
                )
                scaler.step(optimizer_s2)
                scaler.update()
                train_loss += loss.item()

            scheduler_s2.step()
            sc_acc, sa_acc = evaluate(sc_model, sa_model, val_loader, device)
            avg_loss = train_loss / len(train_loader)
            global_step = config["stage1_epochs"] + epoch
            print(
                f"[S2 {epoch:3d}/{config['stage2_epochs']}]"
                f"  loss={avg_loss:.4f}  SC={sc_acc:.4f}  SA={sa_acc:.4f}"
            )
            mlflow.log_metrics(
                {"s2_train_loss": avg_loss, "s2_sc_acc": sc_acc, "s2_sa_acc": sa_acc},
                step=global_step,
            )

            if sa_acc > best_sa_acc:
                best_sa_acc  = sa_acc
                patience_ctr = 0
                torch.save(sa_model.state_dict(), config["save_path"])
                mlflow.pytorch.log_model(sa_model, "best_sa_model")
                print(f"  -> Saved best SA model  (sa_acc={best_sa_acc:.4f})")
            else:
                patience_ctr += 1
                if patience_ctr >= config["patience"]:
                    print(f"Early stopping at Stage 2 epoch {epoch}")
                    break

        mlflow.log_metric("best_sa_acc", best_sa_acc)

    sc_acc_f, sa_acc_f = evaluate(sc_model, sa_model, val_loader, device)
    print(f"\nFinal val:  SC acc = {sc_acc_f:.4f}   SA acc = {sa_acc_f:.4f}")
    print(f"Best SA val acc: {best_sa_acc:.4f}")
    print(f"SA model saved to: {config['save_path']}")


# ====================================================================
# 默认配置
# ====================================================================

DEFAULT_CONFIG = {
    "train_split":   "datasets/splits/M2/train.txt",
    "val_split":     "datasets/splits/M2/val.txt",
    "env_dir":       "datasets/processed/envelopes",
    "sc_model_path": "models/M2/M2_obstacle_classifier_v1.0.0.pt",
    "save_path":     "models/M2/M2_spatial_attention_v1.0.0.pt",
    "stage1_lr":     5e-4,
    "stage2_lr":     5e-5,
    "weight_decay":  1e-4,
    "batch_size":    32,
    "stage1_epochs": 20,
    "stage2_epochs": 80,
    "patience":      20,
    "num_workers":   0,
}

if __name__ == "__main__":
    train(DEFAULT_CONFIG)
