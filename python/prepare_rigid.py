#!/usr/bin/env python3
"""Prepare the deterministic real rigid-registration fixture from inlier-data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def evenly_spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= limit:
        return np.arange(count)
    return np.linspace(0, count - 1, limit, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/inlier-data"))
    parser.add_argument("--max-correspondences", type=int, default=1_024)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.max_correspondences < 3 or args.threshold <= 0:
        raise SystemExit("max-correspondences must be at least three and threshold must be positive")

    os.environ["INLIER_DATA_DIR"] = str(args.cache_dir.resolve())
    from inlier_data import TEST_DATA

    points_path = Path(TEST_DATA.fetch("rigid_pose_example_points.txt"))
    transform_path = Path(TEST_DATA.fetch("rigid_pose_example_gt.txt"))
    points = np.loadtxt(points_path, dtype=np.float64)
    transform = np.loadtxt(transform_path, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 6 or transform.shape != (4, 4):
        raise RuntimeError("invalid rigid-registration fixture shape")
    indices = evenly_spaced_indices(len(points), args.max_correspondences)
    selected = points[indices]
    payload = {
        "schema_version": 1,
        "dataset": "rigid-registration-example",
        "threshold": args.threshold,
        "pairs": [
            {
                "scene": "rigid_pose_example",
                "points_src": selected[:, :3].tolist(),
                "points_tgt": selected[:, 3:].tolist(),
                "transform": transform.tolist(),
            }
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"prepared {len(selected)} rigid correspondences: {args.output}")


if __name__ == "__main__":
    main()
