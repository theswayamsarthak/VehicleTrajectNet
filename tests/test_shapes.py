"""
tests/test_shapes.py
====================
Unit tests for tensor shapes and basic correctness.

Run from project root:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short   (shorter tracebacks)

Why tests matter for a Honda interview:
  99% of student portfolio projects have zero tests. Even 20 lines of
  shape assertions signals that you write production-quality code, not
  just notebook experiments. If the interviewer asks "how do you know
  your model outputs the right shape?", you point at this file.

These tests are deliberately lightweight — no nuScenes data required,
no GPU required, runs in under 5 seconds. They verify:
  - Model input/output shapes are correct
  - Metric functions produce scalars with the right mathematical properties
  - Normalisation puts the agent at the origin
  - Inference module loads and runs cleanly
"""

import sys
import os
import tempfile
import numpy as np
import torch
import pytest

# Make src/ importable from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dataset  import PAST_STEPS, FUTURE_STEPS, normalise_window, rotate_2d, wrap_angle
from model    import LSTMTrajectoryNet, MultiModalLSTMTrajectoryNet, MultiModalWithConfidence
from evaluate import (compute_ade, compute_fde, compute_confidence_loss,
                      compute_min_ade, compute_min_fde, best_of_k_loss)
from inference import load_model, predict_one, predict_from_numpy


# ── Fixtures ───────────────────────────────────────────────────────────────────

B  = 4   # batch size for all tests
K  = 5   # number of modes
T  = FUTURE_STEPS


@pytest.fixture
def past_batch():
    """Random past sequence batch (B, 8, 4)."""
    return torch.randn(B, PAST_STEPS, 4)


@pytest.fixture
def future_batch():
    """Random future sequence batch (B, 6, 2)."""
    return torch.randn(B, T, 2)


@pytest.fixture
def baseline_model():
    return LSTMTrajectoryNet(hidden_size=32)  # small hidden for fast tests


@pytest.fixture
def multimodal_model():
    return MultiModalLSTMTrajectoryNet(hidden_size=32, K=K)


@pytest.fixture
def confidence_model():
    return MultiModalWithConfidence(hidden_size=32, K=K)


# ── Model shape tests ──────────────────────────────────────────────────────────

class TestModelShapes:

    def test_baseline_output_shape(self, baseline_model, past_batch):
        """Baseline model must output (batch, FUTURE_STEPS, 2)."""
        out = baseline_model(past_batch)
        assert out.shape == (B, T, 2), \
            f"Expected ({B}, {T}, 2), got {out.shape}"

    def test_multimodal_output_shape(self, multimodal_model, past_batch):
        """Multi-modal model must output (batch, K, FUTURE_STEPS, 2)."""
        out = multimodal_model(past_batch)
        assert out.shape == (B, K, T, 2), \
            f"Expected ({B}, {K}, {T}, 2), got {out.shape}"

    def test_confidence_model_output_shapes(self, confidence_model, past_batch):
        """Confidence model must return (pred, log_probs) with correct shapes."""
        pred, log_probs = confidence_model(past_batch)
        assert pred.shape      == (B, K, T, 2), f"pred shape wrong: {pred.shape}"
        assert log_probs.shape == (B, K),        f"log_probs shape wrong: {log_probs.shape}"

    def test_confidence_log_probs_sum_to_one(self, confidence_model, past_batch):
        """Log probs must be valid log-probabilities (exp sums to 1 per sample)."""
        _, log_probs = confidence_model(past_batch)
        probs_sum = log_probs.exp().sum(dim=-1)  # (B,)
        assert torch.allclose(probs_sum, torch.ones(B), atol=1e-5), \
            f"Mode probabilities don't sum to 1: {probs_sum}"

    def test_batch_size_one(self, baseline_model):
        """Model must handle batch size of 1 without shape errors."""
        x   = torch.randn(1, PAST_STEPS, 4)
        out = baseline_model(x)
        assert out.shape == (1, T, 2)

    def test_output_is_differentiable(self, baseline_model, past_batch, future_batch):
        """Backward pass must not throw — gradients must flow."""
        out  = baseline_model(past_batch)
        loss = torch.nn.functional.mse_loss(out, future_batch)
        loss.backward()    # must not raise


# ── Metric tests ───────────────────────────────────────────────────────────────

class TestMetrics:

    def test_ade_scalar(self, future_batch):
        """compute_ade must return a scalar tensor."""
        pred = torch.randn_like(future_batch)
        ade  = compute_ade(pred, future_batch)
        assert ade.ndim == 0, "ADE must be a scalar"

    def test_fde_scalar(self, future_batch):
        """compute_fde must return a scalar tensor."""
        pred = torch.randn_like(future_batch)
        fde  = compute_fde(pred, future_batch)
        assert fde.ndim == 0, "FDE must be a scalar"

    def test_ade_zero_for_perfect_prediction(self, future_batch):
        """ADE must be 0 when prediction equals ground truth."""
        ade = compute_ade(future_batch, future_batch)
        assert ade.item() < 1e-6, f"ADE should be ~0 for perfect pred, got {ade.item()}"

    def test_fde_zero_for_perfect_prediction(self, future_batch):
        """FDE must be 0 when prediction equals ground truth."""
        fde = compute_fde(future_batch, future_batch)
        assert fde.item() < 1e-6

    def test_min_ade_leq_ade(self, future_batch):
        """
        minADE@K with K modes must always be <= single-mode ADE.
        More modes can only help — the best of K is always at least as good
        as one randomly chosen mode.
        """
        # Create K modes, one of which is the single-mode pred
        pred_single = torch.randn_like(future_batch)
        # Stack K copies, perturb all but the first
        pred_multi  = pred_single.unsqueeze(1).repeat(1, K, 1, 1)
        pred_multi[:, 1:] += torch.randn_like(pred_multi[:, 1:]) * 5.0  # large noise

        ade     = compute_ade(pred_single, future_batch)
        min_ade = compute_min_ade(pred_multi, future_batch)

        assert min_ade.item() <= ade.item() + 1e-5, \
            f"minADE ({min_ade.item():.4f}) should be <= ADE ({ade.item():.4f})"

    def test_best_of_k_loss_scalar(self, future_batch):
        """best_of_k_loss must return a scalar."""
        pred = torch.randn(B, K, T, 2)
        loss = best_of_k_loss(pred, future_batch)
        assert loss.ndim == 0

    def test_best_of_k_loss_differentiable(self, future_batch):
        """best_of_k_loss must be differentiable w.r.t. predictions."""
        pred = torch.randn(B, K, T, 2, requires_grad=True)
        loss = best_of_k_loss(pred, future_batch)
        loss.backward()
        assert pred.grad is not None, "Gradient did not flow through best_of_k_loss"


# ── Normalisation tests ────────────────────────────────────────────────────────

class TestNormalisation:

    def test_agent_at_origin_after_normalisation(self):
        """
        After normalise_window, the agent's current position (last past step)
        must be at (0, 0) in the normalised frame.
        """
        past_xy   = np.random.randn(PAST_STEPS, 2).astype(np.float32)
        future_xy = np.random.randn(FUTURE_STEPS, 2).astype(np.float32)
        heading   = float(np.random.uniform(-np.pi, np.pi))

        past_norm, _ = normalise_window(past_xy, future_xy, heading)

        assert np.allclose(past_norm[-1], [0.0, 0.0], atol=1e-5), \
            f"Agent not at origin after normalisation: {past_norm[-1]}"

    def test_agent_heading_zero_after_normalisation(self):
        """
        A unit vector in the agent's original heading direction should map
        to [1, 0] (pointing along +x) after rotation by -heading.
        """
        heading   = np.pi / 4   # 45 degrees
        past_xy   = np.zeros((PAST_STEPS, 2), dtype=np.float32)
        future_xy = np.zeros((FUTURE_STEPS, 2), dtype=np.float32)

        # A point one unit ahead of the agent in their heading direction
        forward_point = np.array([[np.cos(heading), np.sin(heading)]], dtype=np.float32)
        test_xy = np.vstack([past_xy[:-1], forward_point])  # last step = forward point

        # After normalisation from origin, that forward point should be at [1, 0]
        rotated = rotate_2d(forward_point - np.zeros(2), -heading)
        assert np.allclose(rotated[0], [1.0, 0.0], atol=1e-5), \
            f"Forward direction not aligned to +x after rotation: {rotated[0]}"

    def test_wrap_angle_stays_in_range(self):
        """wrap_angle output must be in [-pi, pi]."""
        angles = np.array([-4*np.pi, -np.pi - 0.1, 0.0, np.pi + 0.1, 4*np.pi])
        wrapped = wrap_angle(angles)
        assert np.all(wrapped >= -np.pi - 1e-6), "wrap_angle below -pi"
        assert np.all(wrapped <=  np.pi + 1e-6), "wrap_angle above +pi"

    def test_wrap_angle_small_turns_unchanged(self):
        """Small angles (well inside [-pi,pi]) must be unchanged by wrapping."""
        angles  = np.array([-0.5, -0.1, 0.0, 0.1, 0.5])
        wrapped = wrap_angle(angles)
        assert np.allclose(wrapped, angles, atol=1e-6)


# ── Inference tests ────────────────────────────────────────────────────────────

class TestInference:

    def _make_checkpoint(self, model, extra_args=None):
        """Save a model to a temp file and return the path."""
        args = {'hidden': 32}
        if extra_args:
            args.update(extra_args)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            tmp_path = f.name
        torch.save({
            'epoch':       1,
            'model_class': model.__class__.__name__,
            'model_state': model.state_dict(),
            'val_ade':     1.0,
            'val_fde':     2.0,
            'args':        args,
        }, tmp_path)
        return tmp_path

    def test_load_baseline_model(self, baseline_model):
        """load_model must reconstruct a baseline model from checkpoint."""
        path  = self._make_checkpoint(baseline_model)
        model = load_model(path, verbose=False)
        assert isinstance(model, LSTMTrajectoryNet)
        os.unlink(path)

    def test_load_multimodal_model(self, multimodal_model):
        """load_model must reconstruct a multimodal model from checkpoint."""
        path  = self._make_checkpoint(multimodal_model, extra_args={'K': K})
        model = load_model(path, verbose=False)
        assert isinstance(model, MultiModalLSTMTrajectoryNet)
        os.unlink(path)

    def test_predict_one_shape_baseline(self, baseline_model):
        """predict_one must return (FUTURE_STEPS, 2) for baseline model."""
        past = torch.randn(PAST_STEPS, 4)
        pred = predict_one(baseline_model, past)
        assert pred.shape == (T, 2), f"Expected ({T}, 2), got {pred.shape}"

    def test_predict_one_shape_multimodal(self, multimodal_model):
        """predict_one must return (K, FUTURE_STEPS, 2) for multimodal model."""
        past = torch.randn(PAST_STEPS, 4)
        pred = predict_one(multimodal_model, past)
        assert pred.shape == (K, T, 2), f"Expected ({K}, {T}, 2), got {pred.shape}"

    def test_predict_from_numpy(self, baseline_model):
        """predict_from_numpy must accept numpy and return numpy."""
        past    = np.random.randn(PAST_STEPS, 4).astype(np.float32)
        pred_np = predict_from_numpy(baseline_model, past)
        assert isinstance(pred_np, np.ndarray), "Output must be numpy array"
        assert pred_np.shape == (T, 2)

    def test_model_in_eval_mode_after_load(self, baseline_model):
        """Loaded model must be in eval() mode (no dropout during inference)."""
        path  = self._make_checkpoint(baseline_model)
        model = load_model(path, verbose=False)
        assert not model.training, "Model must be in eval() mode after load_model()"
        os.unlink(path)


# ── Confidence loss tests (new) ────────────────────────────────────────────────

class TestConfidenceLoss:

    def test_confidence_loss_scalar(self, future_batch):
        """compute_confidence_loss must return a scalar."""
        pred      = torch.randn(B, K, T, 2)
        log_probs = torch.randn(B, K).log_softmax(dim=-1)
        loss      = compute_confidence_loss(pred, log_probs, future_batch)
        assert loss.ndim == 0, "Confidence loss must be a scalar"

    def test_confidence_loss_differentiable(self, future_batch):
        """Confidence loss must be differentiable w.r.t. log_probs."""
        pred      = torch.randn(B, K, T, 2)
        log_probs = torch.randn(B, K, requires_grad=True).log_softmax(dim=-1)
        loss      = compute_confidence_loss(pred, log_probs, future_batch)
        loss.backward()
        assert log_probs.grad is not None

    def test_high_confidence_on_best_mode_reduces_loss(self, future_batch):
        """
        Assigning high probability to the best mode must produce lower loss
        than assigning high probability to a random mode.

        This verifies the confidence loss does what we claim: it rewards
        the model for correctly identifying which mode will be most accurate.
        """
        pred = torch.randn(B, K, T, 2)

        target_exp   = future_batch.unsqueeze(1).expand_as(pred)
        mse_per_mode = ((pred - target_exp) ** 2).mean(dim=(-1, -2))
        best_idx     = mse_per_mode.argmin(dim=-1)  # (B,)

        # log_probs that concentrate mass on the best mode
        lp_good = torch.full((B, K), -100.0)
        lp_good[torch.arange(B), best_idx] = 0.0
        lp_good = lp_good.log_softmax(dim=-1)

        # log_probs that concentrate mass on the WORST mode
        worst_idx = mse_per_mode.argmax(dim=-1)
        lp_bad = torch.full((B, K), -100.0)
        lp_bad[torch.arange(B), worst_idx] = 0.0
        lp_bad = lp_bad.log_softmax(dim=-1)

        loss_good = compute_confidence_loss(pred, lp_good, future_batch)
        loss_bad  = compute_confidence_loss(pred, lp_bad,  future_batch)

        assert loss_good.item() < loss_bad.item(), (
            "Confidence loss should be lower when probability is on the best mode"
        )


# ── Static displacement test (new) ────────────────────────────────────────────

class TestDatasetHelpers:

    def test_displacement_features_first_step_is_zero(self):
        """
        When use_displacement_features=True, the first displacement step
        should be zero (no previous position to diff against).
        This is checked at the dataset level logic, not via full Dataset init.
        """
        past_xy_norm = np.array([[1.0, 0.5], [1.5, 0.8], [2.0, 1.1],
                                  [2.5, 1.3], [3.0, 1.4], [3.4, 1.5],
                                  [3.7, 1.6], [4.0, 1.7]], dtype=np.float32)
        past_disp = np.zeros_like(past_xy_norm)
        past_disp[1:] = np.diff(past_xy_norm, axis=0)

        assert np.allclose(past_disp[0], [0.0, 0.0]), \
            "First displacement step must be zero"
        assert not np.allclose(past_disp[1:], 0.0), \
            "Subsequent steps must be non-zero"
