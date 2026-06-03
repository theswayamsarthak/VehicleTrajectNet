"""
scripts/extract_data.py
=======================
One-command entrypoint for trajectory extraction.

Run from the project root:
    python scripts/extract_data.py
    python scripts/extract_data.py --dataroot data/nuscenes --version v1.0-mini
    python scripts/extract_data.py --dataroot data/nuscenes --version v1.0-trainval

This is a thin wrapper around src/dataset.py::extract_trajectories().
Having a clean script (rather than a one-liner in a notebook cell) is the
professional way to handle data pipelines — you can run it from CI, from
Kaggle, or hand it to a colleague without explanation.
"""

import argparse
import sys
import time
from pathlib import Path

# Make src/ importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import extract_trajectories, PAST_STEPS, FUTURE_STEPS, WINDOW


def parse_args():
    p = argparse.ArgumentParser(
        description='Extract vehicle trajectories from nuScenes into a parquet file.'
    )
    p.add_argument(
        '--dataroot', type=str, default='data/nuscenes',
        help='Path to the nuScenes dataset root (default: data/nuscenes)'
    )
    p.add_argument(
        '--version', type=str, default='v1.0-mini',
        help='nuScenes split version (default: v1.0-mini)'
    )
    p.add_argument(
        '--output', type=str, default='data/trajectories.parquet',
        help='Output path for the parquet file (default: data/trajectories.parquet)'
    )
    p.add_argument(
        '--verbose', action='store_true', default=False,
        help='Show nuScenes devkit loading logs'
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*55}")
    print(f"  VehicleTrajectNet — Data Extraction")
    print(f"  Dataset  : {args.version} @ {args.dataroot}")
    print(f"  Output   : {args.output}")
    print(f"  Window   : {PAST_STEPS} past + {FUTURE_STEPS} future = {WINDOW} steps")
    print(f"{'='*55}\n")

    # Validate dataroot exists
    dataroot = Path(args.dataroot)
    if not dataroot.exists():
        print(f"ERROR: dataroot not found: {dataroot.resolve()}")
        print("\nExpected layout:")
        print("  data/")
        print("  └── nuscenes/")
        print("      └── v1.0-mini/")
        print("          ├── maps/")
        print("          ├── samples/")
        print("          └── *.json")
        print("\nDownload from: https://www.nuscenes.org/nuscenes#download")
        sys.exit(1)

    # Load nuScenes
    print("Loading nuScenes devkit...")
    t0 = time.time()
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=args.verbose)
    except Exception as e:
        print(f"\nERROR loading nuScenes: {e}")
        print("Make sure nuscenes-devkit is installed: pip install nuscenes-devkit")
        sys.exit(1)

    print(f"Loaded {len(nusc.scene)} scenes in {time.time()-t0:.1f}s\n")

    # Extract
    print("Extracting trajectories...")
    t0 = time.time()
    df = extract_trajectories(nusc, output_path=args.output)
    elapsed = time.time() - t0

    # Summary
    n_windows = df.groupby(['scene_token', 'instance_token', 'window_id']).ngroups
    n_scenes  = df['scene_token'].nunique()
    n_agents  = df['instance_token'].nunique()

    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Scenes     : {n_scenes}")
    print(f"  Agents     : {n_agents}")
    print(f"  Windows    : {n_windows}")
    print(f"  Output     : {Path(args.output).resolve()}")
    print(f"\nNext step: python src/train.py")


if __name__ == '__main__':
    main()
