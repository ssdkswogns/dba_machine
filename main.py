import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb
import torch.cuda.amp as amp

from engine import train_one_epoch, evaluate
from utils import PureScaler

from models.models import MCTformerPlus1D
from data.load_data import UAHDataset, collate_fn, compute_mean_std


def get_args_parser():
    parser = argparse.ArgumentParser('DBA ViT training and evaluation script', add_help=False)
    parser.add_argument('--data-path', type=str, default='./data/UAH_dataset_processed', help='dataset root')
    parser.add_argument('--model_pth', type=str, default='0715.pth', help='model path')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size per gpu')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--embed-dim', default=64, type=int, help='embedding dimension')
    parser.add_argument('--depth', default=6, type=int, help='number of transformer blocks')
    parser.add_argument('--heads', default=8, type=int, help='number of attention heads')
    parser.add_argument('--mlp-ratio', default=4., type=float, help='mlp hidden dim ratio')
    parser.add_argument('--drop-rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--drop-path-rate', default=0.1, type=float, help='stochastic depth rate')
    parser.add_argument('--num-classes', default=3, type=int, help='number of classes')
    parser.add_argument('--val-split', default=0.2, type=float, help='fraction for validation set')
    parser.add_argument('--device', default='cuda', type=str, help='device')
    parser.add_argument('--project-name', default='DBA-ViT', type=str, help='wandb project name')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DBA ViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.loss_weight = 0.5 # Loss weight for contrastive learning

    # Initialize wandb
    wandb.init(project=args.project_name, name=args.model_pth or f"run_{wandb.util.generate_id()}", config=vars(args))

    cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Dataset and DataLoader
    train_dataset = UAHDataset(args.data_path, split_name='train', mean=None, std=None)
    mean, std = compute_mean_std(train_dataset)
    train_dataset.mean = mean
    train_dataset.std = std

    val_dataset = UAHDataset(args.data_path, split_name='val', mean=mean, std=std)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    loss_scaler = PureScaler()
    # Model initialization
    sample_x, _ = train_dataset[0]
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

    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=100,
        T_mult=2, eta_min=1e-6
    )

    best_val_acc = 0
    best_model_state = None
    for epoch in range(1, args.epochs + 1):
        # Train & Eval
        train_metrics = train_one_epoch(
            model,
            train_loader, 
            optimizer, 
            device, 
            epoch, 
            loss_scaler,
            max_norm=0,
            set_training_mode=True,
            args=args)
        val_metrics = evaluate(
            model, 
            val_loader, 
            device
            )
        scheduler.step()

        # Log to wandb
        log_dict = {'epoch': epoch}
        # train
        log_dict['train_loss'] = train_metrics.get('loss', 0)
        log_dict['train_mct_loss'] = train_metrics.get('mct_loss', 0)
        log_dict['train_patch_loss'] = train_metrics.get('pat_loss',0)
        log_dict['train_attn_loss'] = train_metrics.get('attn_loss',0)
        log_dict['lr'] = optimizer.param_groups[0]['lr']
        # val
        log_dict['val_loss'] = val_metrics.get('loss', 0)
        log_dict['val_acc']  = val_metrics.get('acc', 0)
        wandb.log(log_dict, step=epoch)

        print(
            f"Epoch [{epoch}/{args.epochs}]  "
            f"Train Loss: {train_metrics['loss']:.4f}  "
            f"Val Loss: {val_metrics['loss']:.4f}  "
            f"acc: {val_metrics.get('acc', 0):.4f}  "
            f"LR: {log_dict['lr']:.2e}"
        )

        # update best model only
        if log_dict['val_acc'] > best_val_acc:
            best_val_acc = log_dict['val_acc']
            best_model_state = model.state_dict()
            print("Best model saved")

    # after all epochs, save only the best model
    if best_model_state is not None:
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = os.path.join('checkpoints', args.model_pth)
        torch.save(best_model_state, ckpt_path)
        print(f"Saved Best Model: Acc={best_val_acc:.4f} -> {ckpt_path}")

    wandb.finish()
