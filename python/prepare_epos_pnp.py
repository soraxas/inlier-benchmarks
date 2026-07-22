#!/usr/bin/env python3
"""Fetch EPOS PnP correspondences and prepare normalized absolute-pose inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ARCHIVE_NAME = "epos-pnp-ransac-val-v1.tar.zst"
ARCHIVE_ROOT = "epos-pnp-ransac-val"
CASES_ROOT = "epos_corr_lmo"


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
    if not (root / CASES_ROOT).is_dir():
        raise RuntimeError(f"Archive did not contain {ARCHIVE_ROOT}/{CASES_ROOT}")
    complete.touch()
    return root


def parse_case(path: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) < 10:
        raise ValueError(f"PnP case is too short: {path}")
    scene, image, object_id = map(int, lines[0].split())
    intrinsics = np.array([[float(value) for value in lines[row].split()] for row in range(1, 4)])
    pose_count = int(lines[4])
    if pose_count != 1:
        raise ValueError(f"Expected one ground-truth pose in {path}, found {pose_count}")
    pose = np.array([[float(value) for value in lines[row].split()] for row in range(5, 8)])
    correspondence_count_index = 5 + 3 * pose_count
    correspondence_count = int(lines[correspondence_count_index])
    start = correspondence_count_index + 1
    raw_rows = [[float(value) for value in lines[index].split()] for index in range(start, start + correspondence_count)]
    # Some EPOS files append per-instance confidence fields. The tutorial's
    # common 2D--3D input is the first ten fields, so retain that stable prefix.
    if any(len(row) < 10 for row in raw_rows):
        raise ValueError(f"Tentative correspondence is missing required fields in {path}")
    rows = np.asarray([row[:10] for row in raw_rows], dtype=np.float64)
    if rows.shape != (correspondence_count, 10) or not np.isfinite(rows).all():
        raise ValueError(f"Invalid tentative correspondences in {path}")
    return {
        "scene_id": scene,
        "image_id": image,
        "object_id": object_id,
        "intrinsics": intrinsics,
        "pose": pose,
        "rows": rows,
    }


def select_cases(root: Path, count: int, max_correspondences: int) -> list[dict[str, object]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted((root / CASES_ROOT).glob("*.txt")):
        parsed = parse_case(path)
        if len(parsed["rows"]) >= 3:
            grouped[int(parsed["object_id"])].append(path)
    if not grouped:
        raise RuntimeError("No usable EPOS PnP cases found")

    selected: list[Path] = []
    while len(selected) < count:
        added = False
        for object_id in sorted(grouped):
            if grouped[object_id] and len(selected) < count:
                selected.append(grouped[object_id].pop(0))
                added = True
        if not added:
            break
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} cases but found only {len(selected)}")

    payload = []
    for path in selected:
        parsed = parse_case(path)
        rows = np.asarray(parsed["rows"], dtype=np.float64)
        # EPOS confidence is a probability-like score: larger values are better.
        order = np.argsort(-rows[:, 9], kind="stable")
        if len(order) > max_correspondences:
            positions = np.linspace(0, len(order) - 1, max_correspondences, dtype=int)
            order = order[positions]
        rows = rows[order]
        intrinsics = np.asarray(parsed["intrinsics"], dtype=np.float64)
        homogeneous = np.column_stack((rows[:, :2], np.ones(len(rows))))
        normalized = homogeneous @ np.linalg.inv(intrinsics).T
        normalized = normalized[:, :2] / normalized[:, 2:]
        pose = np.asarray(parsed["pose"], dtype=np.float64)
        payload.append(
            {
                "scene": f"lmo/{path.stem}",
                "points3d": rows[:, 2:5].tolist(),
                "points2d": normalized.tolist(),
                "match_scores": rows[:, 9].tolist(),
                "rotation": pose[:, :3].tolist(),
                "translation": pose[:, 3].tolist(),
            }
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/inlier-data"))
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--max-correspondences", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.cases < 1 or args.max_correspondences < 3 or args.threshold <= 0:
        raise SystemExit("cases and threshold must be positive; max-correspondences must be >= 3")

    os.environ["INLIER_DATA_DIR"] = str(args.cache_dir.resolve())
    from inlier_data import TEST_DATA

    archive = Path(TEST_DATA.fetch(ARCHIVE_NAME))
    root = extract_archive(archive, args.cache_dir.resolve() / "extracted")
    payload = {
        "schema_version": 1,
        "dataset": "epos-pnp-val",
        # Normalized-coordinate threshold, approximately 5.7 px for EPOS intrinsics.
        "threshold": args.threshold,
        "pairs": select_cases(root, args.cases, args.max_correspondences),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"prepared {len(payload['pairs'])} EPOS PnP case(s): {args.output}")


if __name__ == "__main__":
    main()
