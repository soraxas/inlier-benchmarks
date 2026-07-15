#!/usr/bin/env python3
"""Compute SuperRANSAC-compatible relative-pose errors for PhotoTourism trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def normalize_keypoints(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized = homogeneous @ np.linalg.inv(intrinsics).T
    return normalized[:, :2] / normalized[:, 2:]


def pose_error_degrees(pair: dict, trial: dict) -> float:
    matrix = trial.get("fundamental_matrix")
    indices = trial.get("inlier_indices")
    if matrix is None or indices is None or len(indices) < 5:
        return 180.0
    try:
        indices = np.asarray(indices, dtype=int)
        points1 = np.asarray(pair["points1"], dtype=np.float64)[indices]
        points2 = np.asarray(pair["points2"], dtype=np.float64)[indices]
        intrinsics1 = np.asarray(pair["intrinsics1"], dtype=np.float64)
        intrinsics2 = np.asarray(pair["intrinsics2"], dtype=np.float64)
        essential = intrinsics2.T @ np.asarray(matrix, dtype=np.float64) @ intrinsics1
        _, rotation, translation, _ = cv2.recoverPose(
            essential,
            normalize_keypoints(points1, intrinsics1),
            normalize_keypoints(points2, intrinsics2),
            np.eye(3),
        )
        rotation_gt = np.asarray(pair["relative_rotation"], dtype=np.float64)
        translation_gt = np.asarray(pair["relative_translation"], dtype=np.float64).reshape(3)
        trace = np.trace(rotation_gt.T @ rotation)
        rotation_error = np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        translation = translation.reshape(3)
        cosine = abs(np.dot(translation_gt, translation)) / (
            np.linalg.norm(translation_gt) * np.linalg.norm(translation)
        )
        translation_error = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        error = max(float(rotation_error), float(translation_error))
        return error if np.isfinite(error) else 180.0
    except (cv2.error, KeyError, ValueError, np.linalg.LinAlgError):
        return 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    pairs = {
        f"{pair['scene']}/{pair['pair']}": pair
        for pair in json.loads(args.input.read_text())["pairs"]
    }
    trials = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    evaluated = 0
    for trial in trials:
        if trial["suite"] != "phototourism-val":
            continue
        trial["pose_error_deg"] = pose_error_degrees(pairs[trial["scene"]], trial)
        evaluated += 1
    args.results.write_text("".join(json.dumps(trial) + "\n" for trial in trials))
    print(f"evaluated {evaluated} PhotoTourism pose trial(s)")


if __name__ == "__main__":
    main()
