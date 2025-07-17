import torch
import torch.nn as nn
from functools import partial
from .vision_transformer import TimeSeriesViT
import torch.nn.functional as F
import numpy as np

class MCTformerPlus1D(TimeSeriesViT):
    """Multi-Class Token ViT for 1D time-series, inherits from TimeSeriesViT"""
    def __init__(self, seq_len, num_features, patch_size=10, embed_dim=128,
                 depth=6, num_heads=8, mlp_ratio=4., num_classes=10,
                 drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 decay_parameter=0.996):
        super().__init__(seq_len, num_features, patch_size, embed_dim,
                         depth, num_heads, mlp_ratio, num_classes,
                         drop_rate, attn_drop_rate, drop_path_rate)
        # override head to 1D conv
        self.head = nn.Conv1d(embed_dim, num_classes, kernel_size=3, padding=1)
        self.decay_parameter = decay_parameter
        self.num_classes = num_classes

        # multi-class tokens
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, embed_dim))

        # re-init
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        nn.init.trunc_normal_(self.pos_embed_cls, std=.02)

    def forward(self, x, return_att=False, n_layers=2, attention_type='fused'):
        # x: [B, T, F]
        B = x.size(0) # [1,2000,7]
        x = self.patch_embed(x) # [1,2000/10,128] 
        # add patch pos embedding
        x = x + self.pos_emb[:, 1:] # [1,200,128] 
        # class tokens
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.pos_embed_cls
        x = torch.cat([cls_tokens, x], dim=1) # [1,125+3,128]
        x = self.pos_drop(x)

        attn_weights = []
        class_embeddings = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
            class_embeddings.append(x[:, :self.num_classes])
        
        x_cls = x[:, :self.num_classes] # [B, C, D]
        x_patch = x[:, self.num_classes:] # [B, P, D]

        # 1×1 Conv1d(head)로 클래스별 로짓 생성
        patch_feats = x_patch.transpose(1, 2)    # [B, D, P]  → Conv1d 입력
        patch_feats = self.head(patch_feats)         # [B, C, P]  (C = class_num)

        x_cls_logits = x_cls.mean(-1) # [B, C]

        feature_map = patch_feats.detach().clone()  # [B, C, P]
        feature_map = F.relu(feature_map)

        attn_weights = torch.stack(attn_weights) # [L, B, H, N, N]
        attn_weights = attn_weights.mean(2) # [L, B, N, N]
        cls_attn = attn_weights[-n_layers:].mean(0)[:,0:self.num_classes, self.num_classes:] # [B, C, P]

        patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]

        fused_attn = cls_attn * feature_map
        fused_attn = torch.sqrt(fused_attn)

        # 패치별 로짓을 내림차순 정렬 (ranking)
        sorted_patch, _ = torch.sort(patch_feats, dim=2, descending=True)

        # 지수적으로 감소하는 랭크 weight 생성
        weights = torch.logspace(
            0,                                     # rank 0 → weight=1
            sorted_patch.size(2) - 1,              # rank N‑1
            sorted_patch.size(2),
            base=self.decay_parameter,             # p<1, 논문 α (decay)
            device=x.device
        )

        # ⑤ Weighted Ranking Pooling (GWRP)
        patch_logits = (sorted_patch * weights).sum(dim=2) / weights.sum()

        output = [x_cls_logits, torch.stack(class_embeddings), patch_logits]
        return output, cls_attn, patch_attn, fused_attn
