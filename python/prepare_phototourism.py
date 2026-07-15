#!/usr/bin/env python3
"""Fetch PhotoTourism with Pooch and prepare deterministic RANSAC inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

ARCHIVE_NAME = "phototourism-ransac-val-v1.tar.zst"
ARCHIVE_ROOT = "phototourism-ransac-val"
SCENES = ("sacre_coeur", "st_peters_square")


def extract_archive(archive: Path, destination: Path) -> Path:
    root = destination / ARCHIVE_ROOT
    complete = destination / f".{ARCHIVE_ROOT}.complete"
    if root.is_dir() and complete.is_file():
        return root

    destination.mkdir(parents=True, exist_ok=True)
    if root.exists():
        shutil.rmtree(root)
    with subprocess.Popen(["zstd", "-dc", archive], stdout=subprocess.PIPE) as decompressor:
        assert decompressor.stdout is not None
        subprocess.run(["tar", "-xf", "-", "-C", destination], stdin=decompressor.stdout, check=True)
    if decompressor.wait() != 0:
        raise RuntimeError(f"Could not decompress {archive}")
    if not root.is_dir():
        raise RuntimeError(f"Archive did not contain {ARCHIVE_ROOT}")
    complete.touch()
    return root


def select_pairs(root: Path, count: int, max_correspondences: int) -> list[dict[str, object]]:
    candidates: list[tuple[int, str, str]] = []
    for scene in SCENES:
        scene_root = root / scene
        with h5py.File(scene_root / "Fgt.h5", "r") as fundamental, h5py.File(
            scene_root / "matches.h5", "r"
        ) as matches:
            for pair in fundamental.keys():
                shape = matches[pair].shape
                if len(shape) == 2 and shape[1] == 4 and shape[0] >= 8:
                    candidates.append((int(shape[0]), scene, pair))

    selected = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))[:count]
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} pairs but found only {len(selected)}")

    output = []
    for _, scene, pair in selected:
        scene_root = root / scene
        with h5py.File(scene_root / "matches.h5", "r") as matches, h5py.File(
            scene_root / "match_conf.h5", "r"
        ) as confidence, h5py.File(scene_root / "Fgt.h5", "r") as fundamental:
            points = np.asarray(matches[pair], dtype=np.float64)
            scores = np.asarray(confidence[pair], dtype=np.float64).reshape(-1)
            if len(scores) != len(points):
                raise RuntimeError(f"Confidence count does not match correspondences for {scene}/{pair}")
            # The tutorial's match confidence is an error-like score: lower is better.
            order = np.argsort(scores, kind="stable")[:max_correspondences]
            points = points[order]
            if not np.isfinite(points).all():
                raise RuntimeError(f"Non-finite correspondences for {scene}/{pair}")
            output.append(
                {
                    "scene": scene,
                    "pair": pair,
                    "points1": points[:, :2].tolist(),
                    "points2": points[:, 2:].tolist(),
                    "fundamental": np.asarray(fundamental[pair], dtype=np.float64).tolist(),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/inlier-data"))
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--max-correspondences", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=1.0)
    args = parser.parse_args()
    if args.pairs < 1 or args.max_correspondences < 8 or args.threshold <= 0:
        raise SystemExit("pairs must be positive, max-correspondences >= 8, and threshold positive")

    os.environ["INLIER_DATA_DIR"] = str(args.cache_dir.resolve())
    from inlier_data import TEST_DATA

    archive = Path(TEST_DATA.fetch(ARCHIVE_NAME))
    root = extract_archive(archive, args.cache_dir.resolve() / "extracted")
    payload = {
        "schema_version": 1,
        "dataset": "phototourism-val",
        "threshold": args.threshold,
        "pairs": select_pairs(root, args.pairs, args.max_correspondences),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"prepared {len(payload['pairs'])} PhotoTourism pair(s): {args.output}")


if __name__ == "__main__":
    main()
