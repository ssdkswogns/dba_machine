import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils

from sklearn.metrics import average_precision_score
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    train_loop = tqdm(data_loader, desc=f"[Train] Epoch {epoch}", leave=False)

    for sequences, targets in train_loop:
        # sequences: [B, T, F]
        sequences = sequences.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        c_outputs = None
        with torch.cuda.amp.autocast():
            outputs, cls_attn, patch_attn, fused_attn = model(sequences)
            # model returns (logits), or (logits, patch_logits), or (logits, cls_embs, patch_logits)
            outputs, c_outputs, patch_outputs = outputs

            # loss = F.multilabel_soft_margin_loss(outputs, targets)
            # outputs = outputs[-2:] # Last two layers
            loss = F.cross_entropy(outputs, targets) # L_cls
            metric_logger.update(mct_loss=loss.item())

            if c_outputs is not None:
                # 1) 임베딩 정규화
                cls_emb = F.normalize(c_outputs, dim=-1)        # [L, B, C, D]

                # 2) 유사도 행렬 (cosine similarity)
                scores = torch.matmul(cls_emb, cls_emb.transpose(-1, -2))  # [L, B, C, C]

                # 3) Cross‑Entropy용 정답 인덱스(gt_idx): diag(0…C‑1) 를 broadcast
                # Identity matrix를 만드는 대신 cross entropy 함수에는 row의 어떤 idx에 1이 있는지를 줘야함
                L, B, C, _ = scores.shape
                gt_idx = torch.arange(C, device=scores.device).view(1, 1, C).expand(L, B, C)  # [L, B, C]

                # 4) CE Loss (layer·batch·class별) → [B, C, L]
                reg_loss = F.cross_entropy(
                    scores.permute(1, 2, 3, 0),   # [B, C, C, L]
                    gt_idx.permute(1, 2, 0),      # [B, C, L]
                    reduction='none'
                )  # [B, C, L]

                # 5) 단일 라벨 마스크로 그냥 평균
                reg_loss = reg_loss.mean()

                # targets: [B, C] one‑hot 이미지 라벨
                # reg_loss = torch.mean(
                #     torch.sum(reg_loss * targets.unsqueeze(-1), dim=-2) / (targets.sum(dim=-1, keepdim=True) + 1e-8),
                #     dim=-1
                # )  # scalar

                # 6) 전체 손실에 추가
                metric_logger.update(attn_loss=reg_loss.item())
                loss = loss + args.loss_weight * reg_loss

            if patch_outputs is not None:
                # ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                ploss =F.cross_entropy(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = getattr(optimizer, 'is_second_order', False)
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        train_loop.set_postfix({
                    'loss': loss_value,
                    'mct': metric_logger.mct_loss.global_avg,
                    'attn': metric_logger.attn_loss.global_avg if 'attn_loss' in metric_logger.meters else 0,
                    'pat': metric_logger.pat_loss.global_avg if 'pat_loss' in metric_logger.meters else 0,
                    'lr': optimizer.param_groups[0]["lr"]
                })

    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Single-label 다중 클래스 분류 평가: Loss + Accuracy 반환
    """
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    val_loop = tqdm(data_loader, desc="[Val]  ", leave=False)

    correct = 0
    total = 0

    model.eval()
    for sequences, targets in val_loop:
        sequences = sequences.to(device, non_blocking=True)
        targets   = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs, cls_attn, patch_attn, fused_attn = model(sequences)

            outputs, c_outputs, patch_outputs = outputs

            loss = criterion(outputs, targets)

            # Accuracy 계산
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

            metric_logger.update(loss=loss.item())
            val_loop.set_postfix({'val_loss': metric_logger.loss.global_avg})

    metric_logger.synchronize_between_processes()
    avg_loss = metric_logger.loss.global_avg
    accuracy = correct / total if total > 0 else 0.0

    print(f'* Val ─ Loss: {avg_loss:.4f}  Acc: {accuracy:.4f}')

    return {'loss': avg_loss, 'acc': accuracy}

@torch.no_grad()
def visualize_class_attention_ts(data_loader, model, device, args):
    """
    1D 시계열용: 모델 출력에서 제공하는 Class-to-Patch attention만 시각화
    """
    args.patch_size = 10
    model.eval()
    with torch.cuda.amp.autocast():
        for i, (sequences, targets) in enumerate(data_loader):
            # if i > 0:
            #     break
            sequences = sequences.to(device)
            B, T, Fdim = sequences.shape

            # forward with attention
            raw_out, cls_attn, patch_attn, fused_attn = model(sequences)

            # model output: (logits, class2patch_attn)
            cls_logits, cls2patch, patch_logits = raw_out   # cls2patch: [B, C, P]

            class_names = ["Drowsy", "Aggressive", "Normal"]

            for b in range(B):
                # 예측 클래스
                pred = torch.argmax(cls_logits[b])
                print(f"pred: {pred}")
                mat = fused_attn[b].cpu().numpy()  # shape: [C, P]
                gt = targets[b]

                # min-max normalize
                mat_min = mat.min(axis=1, keepdims=True)  # shape: (3, 1)
                mat_max = mat.max(axis=1, keepdims=True) 
                mat = (mat - mat_min) / (mat_max - mat_min)

                seq_np = sequences[b].detach().cpu().numpy()  # [T, Fdim]
                T, Fdim = seq_np.shape
                num_plots = 1 + Fdim  # 1(attn) + feature 개수
                fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2 + 2 * num_plots), sharex=False, gridspec_kw={'height_ratios': [1] + [1]*Fdim})
                if num_plots == 2:
                    axes = [axes[0], axes[1]]  # 2개일 때도 리스트로

                # --- 1. 맨 위: Attention map ---
                ax0 = axes[0]
                print(f"mat: {mat.shape}")
                im = ax0.imshow(mat, aspect='auto', cmap='viridis')
                # fig.colorbar(im, ax=ax0, orientation='vertical', label='Attention weight')
                ax0.set_yticks(np.arange(mat.shape[0]))
                ax0.set_yticklabels([f"{class_names[i]}" for i in range(mat.shape[0])])
                ax0.set_xticks(
                    np.linspace(0, mat.shape[1]-1, min(mat.shape[1], 10), dtype=int)
                )
                # ax0.set_xlim(0, mat.shape[1])  # x축을 0~P-1로 고정
                ax0.set_xticklabels((np.linspace(0, mat.shape[1]-1, min(mat.shape[1], 10), dtype=int) * args.patch_size))
                ax0.set_xlabel("Time step (patch index × patch_size)")
                ax0.set_title(f"Sample {b}: Class→Patch Attention / GT: {class_names[gt]} / Pred: {class_names[pred]}")
                
                feature_names = ["KF_Acc_X","KF_Acc_Y","KF_Acc_Z",
                                    "Roll","Pitch","Yaw","Speed (km/h)"]

                # --- 2. 아래: 각 feature별 시계열 plot ---
                for f in range(Fdim):
                    ax = axes[1+f]
                    ax.plot(np.arange(T), seq_np[:, f], label=f'{feature_names[f]}', color=f'C{f%10}')
                    ax.set_ylabel(f'{feature_names[f]}')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, T-1)  # x축을 0~T-1로 고정
                    # patch 구간 시각적 표시 (세로선)
                    for px in range(0, T, args.patch_size):
                        ax.axvline(px, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
                    if f == Fdim-1:
                        ax.set_xlabel('Time step')
                    else:
                        ax.set_xticklabels([])

                plt.tight_layout()
                plt.show()
