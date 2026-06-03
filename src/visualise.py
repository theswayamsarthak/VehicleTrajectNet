"""
visualise.py
============
Animated GIF export for VehicleTrajectNet predictions.

BUG FIXES applied:
  1. imageio v3 API: use imageio.v2.mimsave (fps/loop removed in v3)
  2. matplotlib canvas: use buffer_rgba() instead of deprecated tostring_rgb()
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from typing import Optional

# imageio v3 changed its API. Using v2 compatibility layer explicitly so the
# code works regardless of whether the user has v2 or v3 installed.
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio  # fall back to v2 install

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TrajectoryDataset, PAST_STEPS, FUTURE_STEPS


def _draw_frame(
    ax,
    past_xy:      np.ndarray,
    future_gt:    np.ndarray,
    future_preds: np.ndarray,
    reveal_steps: int,
    sample_idx:   int = 0,
):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_facecolor('#0d1117')
    ax.set_title(
        f'Sample {sample_idx} | Future step {reveal_steps}/{FUTURE_STEPS}',
        color='#c9d1d9', fontsize=9, pad=6
    )
    ax.tick_params(colors='#484f58', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#21262d')
    ax.grid(True, color='#161b22', linewidth=0.5)

    # Past track
    ax.plot(past_xy[:, 0], past_xy[:, 1], color='#8b949e', linewidth=2.0, zorder=3)
    ax.scatter(past_xy[-1, 0], past_xy[-1, 1], color='#f0f6fc', s=50, zorder=6)

    # Predictions
    if future_preds.ndim == 2:
        future_preds = future_preds[np.newaxis]
    K = future_preds.shape[0]
    alpha_vals = np.linspace(0.80, 0.35, K) if K > 1 else [0.80]

    for k in range(K):
        pred_k = future_preds[k, :reveal_steps]
        if len(pred_k) == 0:
            continue
        traj = np.vstack([past_xy[-1:], pred_k])
        ax.plot(traj[:, 0], traj[:, 1], color='#388bfd', linewidth=1.8,
                alpha=alpha_vals[k], zorder=4)
        ax.scatter(pred_k[-1, 0], pred_k[-1, 1], color='#388bfd', s=25,
                   alpha=alpha_vals[k], zorder=4)

    # Ground truth
    gt_slice = future_gt[:reveal_steps]
    if len(gt_slice) > 0:
        gt_traj = np.vstack([past_xy[-1:], gt_slice])
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], color='#3fb950', linewidth=2.2, zorder=5)
        ax.scatter(gt_slice[-1, 0], gt_slice[-1, 1], color='#3fb950', s=35, zorder=5)

    legend_elems = [
        mpatches.Patch(facecolor='#8b949e', label='Past track'),
        mpatches.Patch(facecolor='#388bfd', label=f'Predicted ({K} mode{"s" if K > 1 else ""})'),
        mpatches.Patch(facecolor='#3fb950', label='Ground truth'),
    ]
    ax.legend(handles=legend_elems, loc='upper left', facecolor='#161b22',
              edgecolor='#21262d', labelcolor='#c9d1d9', fontsize=7, framealpha=0.9)


def _fig_to_array(fig) -> np.ndarray:
    """
    Convert a matplotlib figure to an RGB numpy array.

    BUG FIX: tostring_rgb() was deprecated in matplotlib 3.8 and removed.
    buffer_rgba() returns RGBA bytes — drop alpha channel to get RGB.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(h, w, 4)
    return buf[:, :, :3]  # drop alpha channel → (H, W, 3) RGB


def export_prediction_gif(
    model:       torch.nn.Module,
    dataset:     TrajectoryDataset,
    sample_idx:  int,
    output_path: str,
    fps:         int   = 4,
    figsize:     tuple = (6, 6),
    multimodal:  bool  = False,
    device:      Optional[torch.device] = None,
) -> str:
    """
    Export an animated GIF for one trajectory sample.

    Args:
        model       : trained LSTMTrajectoryNet or MultiModalLSTMTrajectoryNet
        dataset     : TrajectoryDataset (val split recommended)
        sample_idx  : index into dataset
        output_path : path for output .gif file
        fps         : frames per second
        multimodal  : True if model outputs (batch, K, T, 2)
        device      : auto-detected if None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval().to(device)

    past_seq, future_seq = dataset[sample_idx]
    past_input = past_seq.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(past_input)
        # Handle MultiModalWithConfidence which returns (pred, log_probs)
        if isinstance(pred, tuple):
            pred = pred[0]

    past_xy      = past_seq[:, :2].numpy()
    future_gt    = future_seq.numpy()
    future_preds = pred.squeeze(0).cpu().numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames = []

    for reveal_steps in range(1, FUTURE_STEPS + 2):
        is_hold = reveal_steps == FUTURE_STEPS + 1

        fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#0d1117')
        _draw_frame(ax, past_xy, future_gt, future_preds,
                    min(reveal_steps, FUTURE_STEPS), sample_idx)
        fig.tight_layout(pad=0.5)

        frame = _fig_to_array(fig)   # BUG FIX: was tostring_rgb()
        frames.append(frame)

        if is_hold:         # hold last frame for 1 second
            for _ in range(fps):
                frames.append(frame)

        plt.close(fig)

    # BUG FIX: imageio v3 removed fps/loop kwargs from mimsave.
    # Using imageio.v2.mimsave (imported at top) which keeps the v2 API.
    # duration= is in seconds per frame; fps=4 → duration=0.25s
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

    size_kb = Path(output_path).stat().st_size // 1024
    print(f"  GIF saved -> {output_path}  ({len(frames)} frames, {size_kb} KB)")
    return output_path


def batch_export_gifs(
    model,
    dataset,
    output_dir: str  = 'outputs/',
    n_samples:  int  = 8,
    multimodal: bool = False,
    device      = None,
    seed:       int  = 0,
) -> list[str]:
    """Export GIFs for n_samples random val samples. Skips failures silently."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    mode    = 'multimodal' if multimodal else 'baseline'
    paths   = []

    print(f"\nExporting {len(indices)} GIFs to {output_dir} ...")
    for idx in indices:
        out_path = str(Path(output_dir) / f'{mode}_sample_{int(idx):04d}.gif')
        try:
            export_prediction_gif(model, dataset, int(idx), out_path,
                                  multimodal=multimodal, device=device)
            paths.append(out_path)
        except Exception as e:
            print(f"  Skipped sample {idx}: {e}")

    print(f"\nDone. {len(paths)}/{len(indices)} GIFs exported.")
    return paths
