#!/usr/bin/env python3
import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from data.load_data import UAHDataset, collate_fn, compute_mean_std
from models.models import MCTformerPlus1D
from engine import visualize_class_attention_ts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 1D ViT class-token attention")
    parser.add_argument("--checkpoint", required=True, help="pretrained .pth model file")
    parser.add_argument("--data-path", default="./data/UAH_dataset_processed")
    parser.add_argument('--depth', default=6, type=int, help='number of transformer blocks')
    parser.add_argument('--heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--mlp-ratio', default=4., type=float, help='mlp hidden dim ratio')
    parser.add_argument('--drop-rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='stochastic depth rate')
    parser.add_argument('--num-classes', default=3, type=int, help='number of classes')
    parser.add_argument('--embed-dim', default=64, type=int, help='embedding dimension')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = UAHDataset(args.data_path, split_name='train', mean=None, std=None)
    mean, std = compute_mean_std(train_dataset)
    train_dataset.mean = mean
    train_dataset.std = std

    val_dataset = UAHDataset(args.data_path, split_name='val', mean=mean, std=std)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    sample_x, _ = val_dataset[0]
    seq_len, feat_dim = sample_x.shape
    
    model = MCTformerPlus1D(
        seq_len=seq_len,
        num_features=feat_dim,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate
    ).to(device, dtype=torch.float32)

    model.load_state_dict(torch.load(args.checkpoint))

    model.eval()
    visualize_class_attention_ts(val_loader, model, device, args)
