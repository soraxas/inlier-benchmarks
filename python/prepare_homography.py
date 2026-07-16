#!/usr/bin/env python3
"""Fetch the homography fixture and prepare deterministic RANSAC inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

ARCHIVE_NAME = "homography-ransac-val-v1.tar.zst"
ARCHIVE_ROOT = "homography-ransac-val"
DATASETS = ("HPatchesSeq", "EVD")


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
    candidates: dict[str, list[tuple[int, str]]] = {}
    for dataset in DATASETS:
        dataset_root = root / dataset / "val"
        with h5py.File(dataset_root / "matches.h5", "r") as matches, h5py.File(
            dataset_root / "Hgt.h5", "r"
        ) as homographies:
            candidates[dataset] = sorted(
                (
                    (int(matches[pair].shape[0]), pair)
                    for pair in homographies.keys()
                    if pair in matches and matches[pair].ndim == 2 and matches[pair].shape[1] == 4
                    and matches[pair].shape[0] >= 4
                ),
                key=lambda candidate: (-candidate[0], candidate[1]),
            )

    selected: list[tuple[str, str]] = []
    while len(selected) < count:
        added = False
        for dataset in DATASETS:
            if candidates[dataset] and len(selected) < count:
                _, pair = candidates[dataset].pop(0)
                selected.append((dataset, pair))
                added = True
        if not added:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} pairs but found only {len(selected)}")

    output = []
    for dataset, pair in selected:
        dataset_root = root / dataset / "val"
        with h5py.File(dataset_root / "matches.h5", "r") as matches, h5py.File(
            dataset_root / "match_conf.h5", "r"
        ) as confidence, h5py.File(dataset_root / "Hgt.h5", "r") as homographies:
            points = np.asarray(matches[pair], dtype=np.float64)
            scores = np.asarray(confidence[pair], dtype=np.float64).reshape(-1)
            homography = np.asarray(homographies[pair], dtype=np.float64)
            if len(scores) != len(points):
                raise RuntimeError(f"Confidence count does not match correspondences for {dataset}/{pair}")
            if homography.shape != (3, 3) or not np.isfinite(homography).all():
                raise RuntimeError(f"Invalid ground-truth homography for {dataset}/{pair}")
            # Tutorial match confidence is an error-like quantity: lower is better.
            order = np.argsort(scores, kind="stable")
            if len(order) > max_correspondences:
                positions = np.linspace(0, len(order) - 1, max_correspondences, dtype=int)
                order = order[positions]
            points = points[order]
            if not np.isfinite(points).all():
                raise RuntimeError(f"Non-finite correspondences for {dataset}/{pair}")
            output.append(
                {
                    "dataset": dataset,
                    "pair": pair,
                    "points1": points[:, :2].tolist(),
                    "points2": points[:, 2:].tolist(),
                    "homography": homography.tolist(),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/inlier-data"))
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--max-correspondences", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    if args.pairs < 1 or args.max_correspondences < 4 or args.threshold <= 0:
        raise SystemExit("pairs must be positive, max-correspondences >= 4, and threshold positive")

    os.environ["INLIER_DATA_DIR"] = str(args.cache_dir.resolve())
    from inlier_data import TEST_DATA

    archive = Path(TEST_DATA.fetch(ARCHIVE_NAME))
    root = extract_archive(archive, args.cache_dir.resolve() / "extracted")
    payload = {
        "schema_version": 1,
        "dataset": "homography-ransac-val",
        "threshold": args.threshold,
        "pairs": select_pairs(root, args.pairs, args.max_correspondences),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"prepared {len(payload['pairs'])} homography pair(s): {args.output}")


if __name__ == "__main__":
    main()
