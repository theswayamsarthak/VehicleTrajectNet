"""
src/inference.py
================
Clean interface for loading checkpoints and running predictions.

Handles all three model variants:
  LSTMTrajectoryNet           — returns (B, 6, 2)
  MultiModalLSTMTrajectoryNet — returns (B, K, 6, 2)
  MultiModalWithConfidence    — returns ((B, K, 6, 2), (B, K))

load_model() reads architecture args from the checkpoint so you never
have to manually specify hidden_size or K when loading.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from model   import LSTMTrajectoryNet, MultiModalLSTMTrajectoryNet, MultiModalWithConfidence
from dataset import PAST_STEPS, FUTURE_STEPS

MODEL_REGISTRY = {
    'LSTMTrajectoryNet':           LSTMTrajectoryNet,
    'MultiModalLSTMTrajectoryNet': MultiModalLSTMTrajectoryNet,
    'MultiModalWithConfidence':    MultiModalWithConfidence,
}


def load_model(
    checkpoint_path: str,
    device:          Optional[torch.device] = None,
    verbose:         bool = True,
) -> torch.nn.Module:
    """
    Load a trained model from a checkpoint file.

    Reads model_class, args, and vel_stats from the checkpoint dict,
    so you never have to remember the hyperparameters separately.

    Args:
        checkpoint_path : path to .pt file saved by train.py
        device          : auto-detected if None
        verbose         : print model summary on load

    Returns:
        model in eval() mode
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt             = torch.load(checkpoint_path, map_location=device)
    model_class_name = ckpt.get('model_class', 'LSTMTrajectoryNet')
    args             = ckpt.get('args', {})

    if model_class_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class '{model_class_name}'. "
                         f"Known: {list(MODEL_REGISTRY.keys())}")

    ModelClass   = MODEL_REGISTRY[model_class_name]
    model_kwargs = {'hidden_size': args.get('hidden', 128)}
    if 'K' in args:
        model_kwargs['K'] = args['K']

    model = ModelClass(**model_kwargs)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()

    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"Loaded  : {model_class_name}")
        print(f"  Epoch    : {ckpt.get('epoch', '?')}")
        print(f"  Val ADE  : {ckpt.get('val_ade', '?'):.4f} m")
        print(f"  Val FDE  : {ckpt.get('val_fde', '?'):.4f} m")
        print(f"  Params   : {n:,}")
        print(f"  Device   : {device}")
        if 'vel_stats' in ckpt:
            vm, vs = ckpt['vel_stats']
            print(f"  Vel norm : mean={vm:.3f} std={vs:.3f}")

    return model


def is_multimodal(model: torch.nn.Module) -> bool:
    return isinstance(model, (MultiModalLSTMTrajectoryNet, MultiModalWithConfidence))

def has_confidence(model: torch.nn.Module) -> bool:
    return isinstance(model, MultiModalWithConfidence)


@torch.no_grad()
def predict_one(model, past_seq: torch.Tensor, device=None):
    """
    Single-sample inference.

    Args:
        past_seq : (8, 4) tensor
    Returns:
        Single-mode model:     (6, 2)
        Multi-modal model:     (K, 6, 2)
        Confidence model:      ((K, 6, 2), (K,))
    """
    if device is None:
        device = next(model.parameters()).device
    out = model(past_seq.unsqueeze(0).to(device))
    if isinstance(out, tuple):
        pred, lp = out
        return pred.squeeze(0), lp.squeeze(0)
    return out.squeeze(0)


@torch.no_grad()
def predict_batch(model, past_batch: torch.Tensor, device=None):
    """Batch inference. Returns same structure as predict_one but with batch dim."""
    if device is None:
        device = next(model.parameters()).device
    return model(past_batch.to(device))


def predict_from_numpy(model, past_seq: np.ndarray, device=None) -> np.ndarray:
    """Numpy in, numpy out. Convenience wrapper for notebooks."""
    tensor = torch.from_numpy(past_seq.astype(np.float32))
    result = predict_one(model, tensor, device=device)
    if isinstance(result, tuple):
        pred, lp = result
        return pred.cpu().numpy(), lp.cpu().numpy()
    return result.cpu().numpy()


if __name__ == '__main__':
    import os, tempfile
    print("Testing inference.py...")

    model = LSTMTrajectoryNet(hidden_size=32)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp = f.name
    torch.save({'epoch': 1, 'model_class': 'LSTMTrajectoryNet',
                'model_state': model.state_dict(), 'val_ade': 1.0,
                'val_fde': 2.0, 'args': {'hidden': 32},
                'vel_stats': (2.5, 3.1)}, tmp)

    m   = load_model(tmp, verbose=True)
    p   = predict_one(m, torch.randn(PAST_STEPS, 4))
    pnp = predict_from_numpy(m, np.random.randn(PAST_STEPS, 4).astype(np.float32))

    assert p.shape   == (FUTURE_STEPS, 2)
    assert pnp.shape == (FUTURE_STEPS, 2)
    os.unlink(tmp)
    print("All checks passed.")
