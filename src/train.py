"""
train.py
========
Training loop supporting all three model variants:
  --mode baseline     : LSTMTrajectoryNet
  --mode multimodal   : MultiModalLSTMTrajectoryNet  (best-of-K loss)
  --mode confidence   : MultiModalWithConfidence     (best-of-K + NLL loss)

Usage:
  python src/train.py --mode baseline
  python src/train.py --mode multimodal
  python src/train.py --mode confidence --conf_weight 0.5
  python src/train.py --mode multimodal --batch_size 16   # if RAM is tight
  python src/train.py --mode baseline   --no_wandb
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset  import TrajectoryDataset
from model    import LSTMTrajectoryNet, MultiModalLSTMTrajectoryNet, MultiModalWithConfidence
from evaluate import (compute_ade, compute_fde,
                      compute_min_ade, compute_min_fde,
                      best_of_k_loss, compute_confidence_loss)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed — training without experiment tracking.")


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when val ADE stops improving.

    With only 8 training scenes, overfitting can occur even after
    ReduceLROnPlateau has cut the learning rate. Early stopping ensures
    the final model is the best one seen, not the last one trained.

    Args:
        patience : stop if val ADE doesn't improve for this many epochs
    """
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter  = 0
        self.best_ade = float('inf')

    def step(self, val_ade: float) -> bool:
        """Returns True if training should stop."""
        if val_ade < self.best_ade - 1e-4:
            self.best_ade = val_ade
            self.counter  = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet',        type=str,   default='data/trajectories.parquet')
    p.add_argument('--checkpoint_dir', type=str,   default='checkpoints')
    p.add_argument('--mode',           type=str,   default='baseline',
                   choices=['baseline', 'multimodal', 'confidence'],
                   help='baseline | multimodal | confidence')
    p.add_argument('--K',              type=int,   default=5)
    p.add_argument('--conf_weight',    type=float, default=0.5,
                   help='Weight of NLL confidence loss (confidence mode only)')
    p.add_argument('--epochs',         type=int,   default=50)
    p.add_argument('--batch_size',     type=int,   default=32)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--hidden',         type=int,   default=128)
    p.add_argument('--early_stopping', type=int,   default=10,
                   help='Patience epochs for early stopping (0 = disabled)')
    p.add_argument('--displacement',   action='store_true',
                   help='Use displacement (dx,dy) instead of position (x,y) as input')
    p.add_argument('--no_wandb',       action='store_true')
    return p.parse_args()


# ── Train / val loops ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, device, mode, conf_weight):
    model.train()
    total_loss, n = 0.0, 0

    for past, future in loader:
        past, future = past.to(device), future.to(device)
        optimiser.zero_grad()

        if mode == 'confidence':
            pred, log_probs = model(past)
            t_loss  = best_of_k_loss(pred, future)
            c_loss  = compute_confidence_loss(pred, log_probs, future)
            loss    = t_loss + conf_weight * c_loss
        elif mode == 'multimodal':
            pred = model(past)
            loss = best_of_k_loss(pred, future)
        else:
            pred = model(past)
            loss = nn.functional.mse_loss(pred, future)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * past.size(0)
        n          += past.size(0)

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, mode):
    model.eval()
    total_ade, total_fde, n = 0.0, 0.0, 0

    for past, future in loader:
        past, future = past.to(device), future.to(device)

        if mode == 'confidence':
            pred, _ = model(past)   # discard log_probs for eval
        else:
            pred = model(past)

        if mode in ('multimodal', 'confidence'):
            ade = compute_min_ade(pred, future)
            fde = compute_min_fde(pred, future)
        else:
            ade = compute_ade(pred, future)
            fde = compute_fde(pred, future)

        b = past.size(0)
        total_ade += ade.item() * b
        total_fde += fde.item() * b
        n         += b

    return total_ade / n, total_fde / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  VehicleTrajectNet Training")
    print(f"  Device      : {device}")
    print(f"  Mode        : {args.mode}")
    print(f"  Displacement: {args.displacement}")
    print(f"  Epochs      : {args.epochs} | Batch : {args.batch_size} | LR : {args.lr}")
    if args.mode == 'confidence':
        print(f"  Conf weight : {args.conf_weight}")
    print(f"{'='*55}\n")

    # Datasets — val uses training-split velocity stats (z-score consistency)
    train_ds = TrajectoryDataset(
        args.parquet, split='train',
        augment_rotation=True,
        use_displacement_features=args.displacement,
    )
    val_ds = TrajectoryDataset(
        args.parquet, split='val',
        augment_rotation=False,
        vel_stats=train_ds.vel_stats,     # IMPORTANT: use training stats for val
        use_displacement_features=args.displacement,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            f"Training dataset is empty. Run extract_trajectories() first.\n"
            f"Expected parquet at: {args.parquet}"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} windows\n")

    # Model
    if args.mode == 'confidence':
        model    = MultiModalWithConfidence(hidden_size=args.hidden, K=args.K)
        run_name = f'confidence_K{args.K}_cw{args.conf_weight}'
    elif args.mode == 'multimodal':
        model    = MultiModalLSTMTrajectoryNet(hidden_size=args.hidden, K=args.K)
        run_name = f'multimodal_K{args.K}'
    else:
        model    = LSTMTrajectoryNet(hidden_size=args.hidden)
        run_name = f'baseline_h{args.hidden}'

    if args.displacement:
        run_name += '_disp'

    model    = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} | Params: {n_params:,}\n")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=5
    )
    stopper   = EarlyStopping(patience=args.early_stopping) if args.early_stopping > 0 else None

    # W&B
    use_wandb = WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(project='vehicletrajectnet', name=run_name, config=vars(args))

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_ade       = float('inf')
    best_ckpt_path = Path(args.checkpoint_dir) / f'{args.mode}_best.pt'
    metric_label   = f'minADE@{args.K}' if args.mode in ('multimodal','confidence') else 'ADE'

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimiser, device, args.mode, args.conf_weight)
        val_ade, val_fde = evaluate(model, val_loader, device, args.mode)
        scheduler.step(val_ade)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:>3}/{args.epochs}  |  "
            f"loss={train_loss:.4f}  |  "
            f"val {metric_label}={val_ade:.4f}m  |  "
            f"val FDE={val_fde:.4f}m  |  "
            f"{elapsed:.1f}s"
        )

        if use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': train_loss,
                       'val_ade': val_ade, 'val_fde': val_fde})

        if val_ade < best_ade:
            best_ade = val_ade
            torch.save({
                'epoch': epoch, 'model_class': model.__class__.__name__,
                'model_state': model.state_dict(), 'opt_state': optimiser.state_dict(),
                'val_ade': val_ade, 'val_fde': val_fde, 'args': vars(args),
                'vel_stats': train_ds.vel_stats,
            }, best_ckpt_path)
            print(f"  Saved best checkpoint (val {metric_label}={best_ade:.4f}m)")

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'args': vars(args)},
                       Path(args.checkpoint_dir) / f'{args.mode}_epoch_{epoch:03d}.pt')

        # Early stopping
        if stopper and stopper.step(val_ade):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stopping} epochs)")
            break

    print(f"\n{'='*55}")
    print(f"Best val {metric_label}: {best_ade:.4f} m  |  Checkpoint: {best_ckpt_path}")
    print(f"{'='*55}")

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
