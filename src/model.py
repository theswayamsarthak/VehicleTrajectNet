"""
model.py
========
Refactor: all three models now inherit from _LSTMEncoderBase.
_encode was duplicated across LSTMTrajectoryNet, MultiModalLSTMTrajectoryNet,
and MultiModalWithConfidence — extracted into a single base class.

Models:
  LSTMTrajectoryNet          — single-mode baseline
  MultiModalLSTMTrajectoryNet — K-mode implicit mixture
  MultiModalWithConfidence    — K-mode + trainable confidence head
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PAST_STEPS, FUTURE_STEPS


# ── Shared encoder base ───────────────────────────────────────────────────────

class _LSTMEncoderBase(nn.Module):
    """
    Shared LSTM encoder inherited by all trajectory models.

    Extracts this once to avoid copy-pasting _encode across every subclass.
    The context vector is cat(h_n[-1], c_n[-1]) — both the hidden state
    (what the LSTM considers important RIGHT NOW) and the cell state
    (accumulated memory over the full 4-second past window).

    context_size = hidden_size * 2, exposed so subclasses can build
    their decoders without hardcoding the dimension.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.context_size = hidden_size * 2  # cat(h_n, c_n)

        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.1,
        )

    def encode(self, past_seq: torch.Tensor) -> torch.Tensor:
        """
        Run LSTM over past sequence and return context vector.

        Args:
            past_seq : (batch, PAST_STEPS, input_size)
        Returns:
            context  : (batch, 2 * hidden_size)
        """
        _, (h_n, c_n) = self.encoder(past_seq)
        return torch.cat([h_n[-1], c_n[-1]], dim=-1)


# ── Baseline: single-mode ─────────────────────────────────────────────────────

class LSTMTrajectoryNet(_LSTMEncoderBase):
    """
    Single-mode trajectory predictor.

    Encoder: 2-layer LSTM   input=4  hidden=128
    Decoder: MLP            256 → 64 → (future_steps * 2)

    Input : (batch, 8, 4)   past [x, y, heading, velocity] agent-centric
    Output: (batch, 6, 2)   predicted future [x, y]
    """

    def __init__(
        self,
        input_size:   int = 4,
        hidden_size:  int = 128,
        num_layers:   int = 2,
        future_steps: int = FUTURE_STEPS,
    ):
        super().__init__(input_size, hidden_size, num_layers)
        self.future_steps = future_steps

        self.decoder = nn.Sequential(
            nn.Linear(self.context_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, future_steps * 2),
        )

    def forward(self, past_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:   past_seq : (batch, 8, 4)
        Returns: pred    : (batch, 6, 2)
        """
        context = self.encode(past_seq)
        return self.decoder(context).view(-1, self.future_steps, 2)


# ── K-mode implicit mixture ───────────────────────────────────────────────────

class MultiModalLSTMTrajectoryNet(_LSTMEncoderBase):
    """
    K-mode trajectory predictor — implicit mixture, no confidence scores.

    Decoder outputs K futures simultaneously. Training uses best-of-K loss
    (see evaluate.py::best_of_k_loss), which gives gradient only to the
    mode closest to ground truth per sample, encouraging mode specialisation.

    No mode probabilities are learned — this is the limitation that
    MultiModalWithConfidence addresses.

    Input : (batch, 8, 4)       past trajectory
    Output: (batch, K=5, 6, 2)  K candidate futures
    """

    def __init__(
        self,
        input_size:   int = 4,
        hidden_size:  int = 128,
        num_layers:   int = 2,
        future_steps: int = FUTURE_STEPS,
        K:            int = 5,
    ):
        super().__init__(input_size, hidden_size, num_layers)
        self.future_steps = future_steps
        self.K = K

        self.decoder = nn.Sequential(
            nn.Linear(self.context_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, K * future_steps * 2),
        )

    def forward(self, past_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:   past_seq : (batch, 8, 4)
        Returns: pred    : (batch, K, 6, 2)
        """
        context = self.encode(past_seq)
        return self.decoder(context).view(-1, self.K, self.future_steps, 2)


# ── K-mode explicit mixture (with trainable confidence) ───────────────────────

class MultiModalWithConfidence(_LSTMEncoderBase):
    """
    K-mode trajectory predictor with explicit mode probability scores.

    Adds a classification head (conf_head) that learns log P(mode_k | past).
    Trained with a combined loss:
        total = best_of_k_loss(pred, target)
              + conf_weight * confidence_loss(pred, log_probs, target)

    The confidence_loss asks: "which mode was best?" and penalises the model
    if that mode had low predicted probability. Over training, the model learns
    to rank its own predictions.

    Interview: "The trajectory loss teaches the model WHERE the vehicle might go.
    The confidence loss teaches it WHICH of those futures is actually happening.
    A downstream planner needs both to make safe decisions."

    Input : (batch, 8, 4)
    Output: pred      (batch, K, 6, 2)  — K candidate futures
            log_probs (batch, K)         — log P(mode_k | past)
    """

    def __init__(
        self,
        input_size:   int = 4,
        hidden_size:  int = 128,
        num_layers:   int = 2,
        future_steps: int = FUTURE_STEPS,
        K:            int = 5,
    ):
        super().__init__(input_size, hidden_size, num_layers)
        self.future_steps = future_steps
        self.K = K

        self.traj_decoder = nn.Sequential(
            nn.Linear(self.context_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, K * future_steps * 2),
        )

        self.conf_head = nn.Sequential(
            nn.Linear(self.context_size, 32),
            nn.ReLU(),
            nn.Linear(32, K),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, past_seq: torch.Tensor):
        """
        Args:
            past_seq  : (batch, 8, 4)
        Returns:
            pred      : (batch, K, 6, 2)
            log_probs : (batch, K)
        """
        context   = self.encode(past_seq)
        pred      = self.traj_decoder(context).view(-1, self.K, self.future_steps, 2)
        log_probs = self.conf_head(context)
        return pred, log_probs


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    B = 4
    x = torch.randn(B, PAST_STEPS, 4)
    print("Testing model.py...")

    m1 = LSTMTrajectoryNet()
    o1 = m1(x)
    assert o1.shape == (B, FUTURE_STEPS, 2)
    print(f"  LSTMTrajectoryNet:           {o1.shape} | {sum(p.numel() for p in m1.parameters()):,} params")

    m2 = MultiModalLSTMTrajectoryNet(K=5)
    o2 = m2(x)
    assert o2.shape == (B, 5, FUTURE_STEPS, 2)
    print(f"  MultiModalLSTMTrajectoryNet: {o2.shape} | {sum(p.numel() for p in m2.parameters()):,} params")

    m3 = MultiModalWithConfidence(K=5)
    o3, lp = m3(x)
    assert o3.shape == (B, 5, FUTURE_STEPS, 2) and lp.shape == (B, 5)
    print(f"  MultiModalWithConfidence:    {o3.shape} + {lp.shape} | {sum(p.numel() for p in m3.parameters()):,} params")

    print("All shape checks passed.")
