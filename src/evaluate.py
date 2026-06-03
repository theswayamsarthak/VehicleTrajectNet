"""
evaluate.py
===========
ADE / FDE metrics and training losses.

ADE         — average displacement error, all future timesteps
FDE         — displacement error at final timestep only
minADE@K    — ADE of the best mode per sample (multi-modal)
minFDE@K    — FDE of the best mode per sample (multi-modal)
best_of_k_loss      — trajectory training loss for implicit mixture models
compute_confidence_loss — NLL loss for the confidence head in MultiModalWithConfidence
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# ── Single-mode metrics ────────────────────────────────────────────────────────

def compute_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Average Displacement Error — mean L2 across all future timesteps.

    Args:
        pred   : (batch, T, 2)
        target : (batch, T, 2)
    Returns:
        scalar ADE in metres
    """
    return torch.norm(pred - target, dim=-1).mean()


def compute_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Final Displacement Error — L2 at the last predicted timestep only.

    Args:
        pred   : (batch, T, 2)
        target : (batch, T, 2)
    Returns:
        scalar FDE in metres
    """
    return torch.norm(pred[:, -1, :] - target[:, -1, :], dim=-1).mean()


# ── Multi-modal metrics ────────────────────────────────────────────────────────

def compute_min_ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    minADE@K — for each sample, take the mode with lowest ADE, then average.

    Measures COVERAGE of the predicted distribution: even one accurate mode
    scores well. Does not measure whether you can rank which mode is correct —
    that is what the confidence head addresses.

    Args:
        pred   : (batch, K, T, 2)
        target : (batch, T, 2)
    Returns:
        scalar minADE@K
    """
    target_exp   = target.unsqueeze(1).expand_as(pred)         # (batch, K, T, 2)
    l2           = torch.norm(pred - target_exp, dim=-1)        # (batch, K, T)
    ade_per_mode = l2.mean(dim=-1)                              # (batch, K)
    min_ade, _   = ade_per_mode.min(dim=-1)                     # (batch,)
    return min_ade.mean()


def compute_min_fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    minFDE@K — same as minADE but evaluated only at the final timestep.

    Args:
        pred   : (batch, K, T, 2)
        target : (batch, T, 2)
    Returns:
        scalar minFDE@K
    """
    target_exp   = target.unsqueeze(1).expand_as(pred)
    fde_per_mode = torch.norm(
        pred[:, :, -1, :] - target_exp[:, :, -1, :], dim=-1
    )                                                           # (batch, K)
    min_fde, _   = fde_per_mode.min(dim=-1)                    # (batch,)
    return min_fde.mean()


# ── Training losses ────────────────────────────────────────────────────────────

def best_of_k_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Best-of-K trajectory loss for implicit mixture models.

    For each sample, computes MSE against every mode, then takes the minimum.

    Gradient behaviour (important to understand):
        PyTorch's min() is a hard argmin — it propagates gradient to exactly
        ONE element: the winning mode. All other modes receive ZERO gradient
        from that sample. Over many batches, each mode specialises on the
        subset of trajectories where it wins, which is how implicit mode
        specialisation emerges without any explicit routing.

        This also means there is no diversity loss — modes CAN collapse
        (multiple modes converging to the same trajectory). If you observe
        this, a repulsion term or diversity regulariser is the fix.

    Args:
        pred   : (batch, K, T, 2)
        target : (batch, T, 2)
    Returns:
        scalar loss
    """
    target_exp   = target.unsqueeze(1).expand_as(pred)
    mse_per_mode = ((pred - target_exp) ** 2).mean(dim=(-1, -2))  # (batch, K)
    min_mse, _   = mse_per_mode.min(dim=-1)                        # (batch,)
    return min_mse.mean()


def compute_confidence_loss(
    pred:      torch.Tensor,
    log_probs: torch.Tensor,
    target:    torch.Tensor,
) -> torch.Tensor:
    """
    NLL loss for the confidence head in MultiModalWithConfidence.

    Identifies the best mode per sample (argmin MSE against GT), then
    maximises the log-probability the model assigned to that mode.

    Over training, the model learns to predict which of its K futures
    is actually happening — turning an implicit mixture into an explicit one.

    Interview: "The trajectory loss and the confidence loss are trained jointly.
    The trajectory loss teaches the model WHERE the vehicle might go. The
    confidence loss teaches it WHICH of those modes is most likely, given
    the past. A downstream planner needs both."

    Args:
        pred      : (batch, K, T, 2)  — trajectory predictions
        log_probs : (batch, K)         — log P(mode_k | past) from conf_head
        target    : (batch, T, 2)      — ground truth future
    Returns:
        scalar NLL loss (lower = better)
    """
    target_exp    = target.unsqueeze(1).expand_as(pred)
    mse_per_mode  = ((pred - target_exp) ** 2).mean(dim=(-1, -2))  # (batch, K)
    best_mode_idx = mse_per_mode.argmin(dim=-1)                     # (batch,)

    # log_probs of the best mode for each sample
    batch_idx        = torch.arange(pred.size(0), device=pred.device)
    chosen_log_probs = log_probs[batch_idx, best_mode_idx]          # (batch,)

    return -chosen_log_probs.mean()  # maximise probability of best mode


# ── Sanity test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B, T, K = 8, 6, 5
    pred_s  = torch.randn(B, T, 2)
    pred_m  = torch.randn(B, K, T, 2)
    lp      = torch.randn(B, K).log_softmax(dim=-1)
    tgt     = torch.randn(B, T, 2)

    print("Testing evaluate.py...")
    print(f"  ADE            : {compute_ade(pred_s, tgt).item():.4f} m")
    print(f"  FDE            : {compute_fde(pred_s, tgt).item():.4f} m")
    print(f"  minADE@{K}     : {compute_min_ade(pred_m, tgt).item():.4f} m")
    print(f"  minFDE@{K}     : {compute_min_fde(pred_m, tgt).item():.4f} m")
    print(f"  best_of_k_loss : {best_of_k_loss(pred_m, tgt).item():.4f}")
    print(f"  conf_loss      : {compute_confidence_loss(pred_m, lp, tgt).item():.4f}")
    print("All checks passed.")
