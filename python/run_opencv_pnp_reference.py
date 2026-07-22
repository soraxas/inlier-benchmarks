#!/usr/bin/env python3
"""Run OpenCV solvePnPRansac on the prepared EPOS absolute-pose inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

PROFILES = {
    "fast": (250, 0.95),
    "balanced": (1_000, 0.99),
    "thorough": (5_000, 0.999),
}


def trial(pair: dict, profile: str, seed: int, threshold: float) -> dict:
    max_iterations, confidence = PROFILES[profile]
    points3d = np.asarray(pair["points3d"], dtype=np.float64)
    points2d = np.asarray(pair["points2d"], dtype=np.float64)
    if points3d.shape[0] < 4 or points3d.shape != (len(points2d), 3) or points2d.shape[1:] != (2,):
        raise ValueError("invalid EPOS PnP correspondence shape")
    cv2.setRNGSeed(seed & 0x7FFF_FFFF)
    start = time.perf_counter()
    ok, rotation, translation, inliers = cv2.solvePnPRansac(
        points3d,
        points2d,
        np.eye(3),
        None,
        iterationsCount=max_iterations,
        reprojectionError=threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    runtime_ms = (time.perf_counter() - start) * 1_000.0
    pose = None
    indices = None
    if ok and rotation is not None and translation is not None:
        matrix, _ = cv2.Rodrigues(rotation)
        pose = np.column_stack((matrix, translation.reshape(3))).tolist()
        indices = np.asarray(inliers if inliers is not None else [], dtype=int).reshape(-1).tolist()
    return {
        "suite": "epos-pnp-val",
        "suite_version": 1,
        "estimator": "absolute_pose",
        "scoring_mode": "opencv_ransac",
        "sampler": "opencv",
        "variant": "default",
        "threshold_scale": 1.0,
        "profile": profile,
        "scene": pair["scene"],
        "seed": seed,
        "runtime_ms": runtime_ms,
        "iterations": max_iterations,
        "inlier_precision": 0.0,
        "inlier_recall": 0.0,
        "normalized_model_error": 0.0 if pose is not None else 1.0e9,
        "inlier_classification_error": 0.0,
        "epipolar_matrix": None,
        "absolute_pose": pose,
        "inlier_indices": indices,
        "homography_auc_3": None,
        "diagnostics": None,
        "success": pose is not None,
        "failure_reason": None if pose is not None else "OpenCV solvePnPRansac failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    profiles = ["balanced"] if args.smoke else list(PROFILES)
    trials = []
    for pair in payload["pairs"]:
        for profile in profiles:
            for index in range(args.seeds):
                seed = 0x5EED_CAFE_D00D_BAAD ^ index
                trials.append(trial(pair, profile, seed, payload["threshold"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, allow_nan=False) + "\n" for item in trials))
    print(f"wrote {len(trials)} OpenCV EPOS PnP reference trial(s): {args.output}")


if __name__ == "__main__":
    main()
