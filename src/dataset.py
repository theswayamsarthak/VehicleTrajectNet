"""
dataset.py
==========
1. extract_trajectories() — walks nuScenes, saves raw trajectories to parquet
2. TrajectoryDataset       — PyTorch Dataset with normalisation + augmentation

Key design decisions:
  - Static objects filtered at extraction time (parked cars add only noise)
  - Normalisation in Dataset.__init__, not extraction (keeps parquet reusable)
  - Velocity z-scored using training-split statistics (not a heuristic /10)
  - Scene-level train/val split (window-level split leaks — 13/14 steps shared)
  - Optional displacement input features (dx, dy instead of raw x, y)
  - Rotation augmentation on training split only
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ── Constants ──────────────────────────────────────────────────────────────────
PAST_STEPS   = 8
FUTURE_STEPS = 6
DT           = 0.5
WINDOW       = PAST_STEPS + FUTURE_STEPS

VEHICLE_CATEGORIES = {
    'vehicle.car', 'vehicle.truck', 'vehicle.bus',
    'vehicle.construction', 'vehicle.emergency',
    'vehicle.motorcycle', 'vehicle.trailer',
}

# Minimum total displacement (metres) for a trajectory to be included.
# Filters parked/static vehicles that contribute nothing to motion learning.
MIN_DISPLACEMENT_M = 1.0


# ── Geometry helpers ───────────────────────────────────────────────────────────

def quaternion_to_yaw(q: list) -> float:
    """Convert nuScenes quaternion [w, x, y, z] to yaw in radians."""
    from pyquaternion import Quaternion
    return Quaternion(q).yaw_pitch_roll[0]


def rotate_2d(points: np.ndarray, angle: float) -> np.ndarray:
    """Rotate (N, 2) array of [x, y] by angle radians (counter-clockwise)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return points @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angles to [-pi, pi].

    Simple subtraction (heading - origin_heading) breaks at the ±pi boundary:
    a 5-degree turn near heading=pi looks like a 355-degree turn. arctan2
    folds everything into [-pi, pi] correctly.
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


# ── Trajectory extraction ──────────────────────────────────────────────────────

def extract_trajectories(
    nusc,
    output_path: str = 'data/trajectories.parquet',
) -> pd.DataFrame:
    """
    Extract per-agent trajectory windows from all nuScenes scenes.

    Static object filtering:
        Agents with total displacement < MIN_DISPLACEMENT_M across their
        entire visible segment are skipped. Parked cars teach the model
        nothing about motion dynamics and add noise.

    Output columns:
        scene_token, instance_token, window_id, step,
        x, y, heading, velocity, is_past
    """
    records = []

    for scene in nusc.scene:
        scene_token  = scene['token']
        sample_token = scene['first_sample_token']
        sample_tokens = []
        while sample_token:
            sample_tokens.append(sample_token)
            sample_token = nusc.get('sample', sample_token)['next']

        instance_timeline: dict[str, list] = {}
        for step_idx, s_token in enumerate(sample_tokens):
            for ann_token in nusc.get('sample', s_token)['anns']:
                ann      = nusc.get('sample_annotation', ann_token)
                cat_name = nusc.get('category', ann['category_token'])['name']
                if not any(cat_name.startswith(v) for v in VEHICLE_CATEGORIES):
                    continue
                inst = ann['instance_token']
                instance_timeline.setdefault(inst, []).append((step_idx, ann))

        for inst_token, timeline in instance_timeline.items():
            timeline.sort(key=lambda x: x[0])

            # Split into contiguous segments
            segments, seg = [], [timeline[0]]
            for i in range(1, len(timeline)):
                if timeline[i][0] == timeline[i-1][0] + 1:
                    seg.append(timeline[i])
                else:
                    segments.append(seg); seg = [timeline[i]]
            segments.append(seg)

            for seg in segments:
                if len(seg) < WINDOW:
                    continue

                xs, ys, headings, velocities = [], [], [], []
                for _, ann in seg:
                    x, y, _ = ann['translation']
                    heading  = quaternion_to_yaw(ann['rotation'])
                    vx, vy, _ = nusc.box_velocity(ann['token'])
                    speed = float(np.sqrt(vx**2 + vy**2)) if not (np.isnan(vx) or np.isnan(vy)) else 0.0
                    xs.append(x); ys.append(y)
                    headings.append(heading); velocities.append(speed)

                # FIX: skip static/parked agents — they teach the model nothing
                positions  = np.stack([xs, ys], axis=1)
                total_dist = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
                if total_dist < MIN_DISPLACEMENT_M:
                    continue

                for w_start in range(len(seg) - WINDOW + 1):
                    for local_step in range(WINDOW):
                        records.append({
                            'scene_token':    scene_token,
                            'instance_token': inst_token,
                            'window_id':      w_start,
                            'step':           local_step,
                            'x':              xs[w_start + local_step],
                            'y':              ys[w_start + local_step],
                            'heading':        headings[w_start + local_step],
                            'velocity':       velocities[w_start + local_step],
                            'is_past':        local_step < PAST_STEPS,
                        })

    df = pd.DataFrame(records)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    n_windows = df.groupby(['scene_token', 'instance_token', 'window_id']).ngroups
    print(f"Saved {len(df):,} rows ({n_windows:,} windows) -> {output_path}")
    return df


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalise_window(
    past_xy: np.ndarray,
    future_xy: np.ndarray,
    origin_heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Agent-centric normalisation: translate to origin, rotate to face +x.

    Uses the LAST past step as the reference frame (current moment of the agent).
    Interview: "This forces the LSTM to learn motion dynamics, not map geography."
    """
    origin         = past_xy[-1].copy()
    past_xy_norm   = rotate_2d(past_xy   - origin, -origin_heading)
    future_xy_norm = rotate_2d(future_xy - origin, -origin_heading)
    return past_xy_norm, future_xy_norm


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class TrajectoryDataset(Dataset):
    """
    Loads trajectory windows, applies normalisation, returns tensors.

    Key parameters:
        vel_stats               : (mean, std) computed from the TRAINING split.
                                  Pass train_ds.vel_stats to the val dataset.
                                  If None and split='train', computed automatically.
        use_displacement_features: If True, replace raw (x, y) in past_seq
                                  with incremental (dx, dy). More natural for
                                  sequence prediction; constant-velocity prior
                                  is trivially encoded as dx_t = dx_{t-1}.
    """

    def __init__(
        self,
        parquet_path:             str   = 'data/trajectories.parquet',
        split:                    str   = 'train',
        val_fraction:             float = 0.2,
        seed:                     int   = 42,
        vel_stats:                tuple = None,     # (mean, std) from training split
        augment_rotation:         bool  = True,
        aug_angle_range:          float = np.pi / 6,
        use_displacement_features: bool = False,
    ):
        super().__init__()
        self.split                    = split
        self.augment_rotation         = augment_rotation and (split == 'train')
        self.aug_angle_range          = aug_angle_range
        self.use_displacement_features = use_displacement_features

        df = pd.read_parquet(parquet_path)

        # Scene-level split
        all_scenes = df['scene_token'].unique()
        rng        = np.random.default_rng(seed)
        rng.shuffle(all_scenes)
        n_val        = max(1, int(len(all_scenes) * val_fraction))
        val_scenes   = set(all_scenes[:n_val])
        train_scenes = set(all_scenes[n_val:])

        if split == 'val' and len(val_scenes) == 0:
            raise ValueError("val_fraction produced zero val scenes.")
        if split == 'train' and len(train_scenes) == 0:
            raise ValueError("val_fraction produced zero training scenes.")

        chosen = val_scenes if split == 'val' else train_scenes
        df     = df[df['scene_token'].isin(chosen)].copy()
        print(f"[{split}] {len(chosen)} scene(s)")

        # Velocity normalisation statistics
        # IMPROVEMENT: z-score from training data, not heuristic /10.
        # Val dataset MUST use training split stats — passing vel_stats ensures this.
        if vel_stats is not None:
            self.vel_mean, self.vel_std = vel_stats
        elif split == 'train':
            raw_vel       = df['velocity'].values
            self.vel_mean = float(raw_vel.mean())
            self.vel_std  = float(raw_vel.std()) + 1e-6
            print(f"[train] Velocity stats: mean={self.vel_mean:.3f} m/s, std={self.vel_std:.3f} m/s")
        else:
            raise ValueError(
                "Val dataset needs velocity stats from the training dataset.\n"
                "Use: val_ds = TrajectoryDataset(..., split='val', vel_stats=train_ds.vel_stats)"
            )

        # Pre-process and cache all windows
        self.past_seqs   = []
        self.future_seqs = []

        grouped = df.groupby(['scene_token', 'instance_token', 'window_id'], sort=False)
        for _, window_df in grouped:
            window_df = window_df.sort_values('step')
            past_df   = window_df[window_df['is_past']]
            future_df = window_df[~window_df['is_past']]

            if len(past_df) != PAST_STEPS or len(future_df) != FUTURE_STEPS:
                continue

            past_xy      = past_df[['x', 'y']].values.astype(np.float32)
            past_heading = past_df['heading'].values.astype(np.float32)
            past_vel     = past_df['velocity'].values.astype(np.float32)
            future_xy    = future_df[['x', 'y']].values.astype(np.float32)

            origin_heading              = float(past_heading[-1])
            past_xy_norm, future_xy_norm = normalise_window(past_xy, future_xy, origin_heading)
            past_heading_norm           = wrap_angle(past_heading - origin_heading).astype(np.float32)
            past_vel_norm               = ((past_vel - self.vel_mean) / self.vel_std).astype(np.float32)

            if use_displacement_features:
                # Replace (x, y) with (dx, dy): incremental displacements.
                # dx[0] = 0 (no previous step to diff against).
                # Interview: "Displacement features encode a constant-velocity
                # prior naturally — dx_t ≈ dx_{t-1} for uniform motion —
                # whereas absolute positions require the LSTM to learn this."
                past_disp = np.zeros_like(past_xy_norm)
                past_disp[1:] = np.diff(past_xy_norm, axis=0)
                spatial_feat = past_disp
            else:
                spatial_feat = past_xy_norm

            past_seq = np.stack([
                spatial_feat[:, 0],
                spatial_feat[:, 1],
                past_heading_norm,
                past_vel_norm,
            ], axis=-1).astype(np.float32)

            self.past_seqs.append(torch.from_numpy(past_seq))
            self.future_seqs.append(torch.from_numpy(future_xy_norm))

        print(f"[{split}] {len(self.past_seqs)} trajectory windows loaded.")

    @property
    def vel_stats(self) -> tuple:
        """Returns (vel_mean, vel_std) for passing to the val dataset."""
        return (self.vel_mean, self.vel_std)

    def __len__(self) -> int:
        return len(self.past_seqs)

    def __getitem__(self, idx: int):
        past_seq   = self.past_seqs[idx].clone()
        future_seq = self.future_seqs[idx].clone()

        if self.augment_rotation:
            angle = np.random.uniform(-self.aug_angle_range, self.aug_angle_range)
            cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
            R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)
            # Heading (col 2) and velocity (col 3) are rotation-invariant scalars —
            # only the spatial coordinates (cols 0,1) are rotated.
            past_seq[:, :2]  = past_seq[:, :2]  @ R.T
            future_seq[:, :] = future_seq[:, :] @ R.T

        return past_seq, future_seq
