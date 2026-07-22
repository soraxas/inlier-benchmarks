#!/usr/bin/env python3
"""Run OpenCV robust estimators on the same immutable benchmark fixtures as inlier."""

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

OPENCV_METHODS = {
    "opencv_ransac": cv2.RANSAC,
    "opencv_usac_prosac": cv2.USAC_PROSAC,
    "opencv_usac_magsac": cv2.USAC_MAGSAC,
}
ESSENTIAL_THRESHOLD_SCALES = (0.25, 0.5, 2.0, 4.0)


def sampson_errors(matrix: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    first = np.column_stack((points1, np.ones(len(points1))))
    second = np.column_stack((points2, np.ones(len(points2))))
    numerator = np.einsum("ni,ij,nj->n", second, matrix, first)
    forward = first @ matrix.T
    backward = second @ matrix
    denominator = np.sqrt(
        forward[:, 0] ** 2 + forward[:, 1] ** 2 + backward[:, 0] ** 2 + backward[:, 1] ** 2
    )
    return np.divide(
        np.abs(numerator), denominator, out=np.full(len(points1), np.inf), where=denominator > 1e-12
    )


def transfer_errors(matrix: np.ndarray, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    source = np.column_stack((points1, np.ones(len(points1))))
    projected = source @ matrix.T
    projected = np.divide(
        projected[:, :2],
        projected[:, 2:3],
        out=np.full((len(points1), 2), np.inf),
        where=np.abs(projected[:, 2:3]) > 1e-12,
    )
    return np.linalg.norm(projected - points2, axis=1)


def auc_at_threshold(errors: np.ndarray, threshold: float) -> float:
    finite = np.sort(errors[np.isfinite(errors) & (errors >= 0.0)])
    if not len(finite):
        return 0.0
    previous_error = 0.0
    previous_recall = 0.0
    area = 0.0
    for index, error in enumerate(finite, start=1):
        if error > threshold:
            break
        recall = index / len(finite)
        area += (error - previous_error) * (previous_recall + recall) * 0.5
        previous_error = error
        previous_recall = recall
    area += (threshold - previous_error) * previous_recall
    return float(area / threshold)


def classification(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    selected = np.zeros(len(truth), dtype=bool)
    selected[predicted] = True
    true_positive = int(np.count_nonzero(selected & truth))
    precision = true_positive / max(int(np.count_nonzero(selected)), 1)
    recall = true_positive / max(int(np.count_nonzero(truth)), 1)
    return precision, recall, 1.0 - min(precision, recall)


def base_trial(
    *,
    suite: str,
    estimator: str,
    scoring_mode: str,
    variant: str,
    threshold_scale: float,
    profile: str,
    scene: str,
    seed: int,
    runtime_ms: float,
) -> dict:
    return {
        "suite": suite,
        "suite_version": 1,
        "estimator": estimator,
        "scoring_mode": scoring_mode,
        "sampler": "opencv",
        "variant": variant,
        "threshold_scale": threshold_scale,
        "profile": profile,
        "scene": scene,
        "seed": seed,
        "runtime_ms": runtime_ms,
        "iterations": PROFILES[profile][0],
        "inlier_precision": 0.0,
        "inlier_recall": 0.0,
        "normalized_model_error": 1_000_000_000.0,
        "inlier_classification_error": 1.0,
        "epipolar_matrix": None,
        "inlier_indices": None,
        "homography_auc_3": None,
        "diagnostics": None,
        "success": False,
        "failure_reason": f"OpenCV {scoring_mode} did not produce a model",
    }


def run_fundamental(
    pair: dict,
    threshold: float,
    profile: str,
    seed: int,
    scoring_mode: str = "opencv_usac_magsac",
    *,
    variant: str = "default",
    threshold_scale: float = 1.0,
) -> dict:
    points1 = np.asarray(pair["points1"], dtype=np.float64)
    points2 = np.asarray(pair["points2"], dtype=np.float64)
    max_iterations, confidence = PROFILES[profile]
    method = OPENCV_METHODS[scoring_mode]
    cv2.setRNGSeed(seed & 0x7FFF_FFFF)
    start = time.perf_counter_ns()
    matrix, mask = cv2.findFundamentalMat(
        points1, points2, method, threshold, confidence, max_iterations
    )
    runtime_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    trial = base_trial(
        suite="phototourism-val",
        estimator="fundamental",
        scoring_mode=scoring_mode,
        variant=variant,
        threshold_scale=threshold_scale,
        profile=profile,
        scene=f"{pair['scene']}/{pair['pair']}",
        seed=seed,
        runtime_ms=runtime_ms,
    )
    if matrix is None or mask is None or np.asarray(matrix).shape != (3, 3):
        return trial
    matrix = np.asarray(matrix, dtype=np.float64)
    selected = np.flatnonzero(np.asarray(mask).reshape(-1) != 0)
    if len(selected) < 5 or not np.isfinite(matrix).all():
        return trial
    truth = sampson_errors(np.asarray(pair["fundamental"], dtype=np.float64), points1, points2) <= threshold
    residuals = sampson_errors(matrix, points1, points2)
    precision, recall, classification_error = classification(selected, truth)
    trial.update(
        {
            "inlier_precision": precision,
            "inlier_recall": recall,
            "normalized_model_error": float(np.median(residuals[truth]) / threshold)
            if np.any(truth)
            else 1_000_000_000.0,
            "inlier_classification_error": classification_error,
            "epipolar_matrix": matrix.tolist(),
            "inlier_indices": selected.tolist(),
            "success": True,
            "failure_reason": None,
        }
    )
    return trial


def run_essential(
    pair: dict,
    threshold: float,
    profile: str,
    seed: int,
    scoring_mode: str = "opencv_usac_magsac",
    *,
    variant: str = "default",
    threshold_scale: float = 1.0,
) -> dict:
    points1 = np.asarray(pair["points1"], dtype=np.float64)
    points2 = np.asarray(pair["points2"], dtype=np.float64)
    intrinsics1 = np.asarray(pair["intrinsics1"], dtype=np.float64)
    intrinsics2 = np.asarray(pair["intrinsics2"], dtype=np.float64)
    max_iterations, confidence = PROFILES[profile]
    focal_scale = np.mean(
        [intrinsics1[0, 0], intrinsics1[1, 1], intrinsics2[0, 0], intrinsics2[1, 1]]
    )
    normalized_threshold = threshold * threshold_scale / focal_scale
    method = OPENCV_METHODS[scoring_mode]
    cv2.setRNGSeed(seed & 0x7FFF_FFFF)
    start = time.perf_counter_ns()
    normalized1 = cv2.undistortPoints(points1.reshape(-1, 1, 2), intrinsics1, None).reshape(-1, 2)
    normalized2 = cv2.undistortPoints(points2.reshape(-1, 1, 2), intrinsics2, None).reshape(-1, 2)
    matrix, mask = cv2.findEssentialMat(
        normalized1,
        normalized2,
        np.eye(3),
        method,
        confidence,
        normalized_threshold,
        max_iterations,
    )
    runtime_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    trial = base_trial(
        suite="phototourism-val",
        estimator="essential",
        scoring_mode=scoring_mode,
        variant=variant,
        threshold_scale=threshold_scale,
        profile=profile,
        scene=f"{pair['scene']}/{pair['pair']}",
        seed=seed,
        runtime_ms=runtime_ms,
    )
    if matrix is None or mask is None or np.asarray(matrix).shape != (3, 3):
        return trial
    matrix = np.asarray(matrix, dtype=np.float64)
    selected = np.flatnonzero(np.asarray(mask).reshape(-1) != 0)
    if len(selected) < 5 or not np.isfinite(matrix).all():
        return trial
    truth = sampson_errors(np.asarray(pair["essential"], dtype=np.float64), normalized1, normalized2) <= normalized_threshold
    residuals = sampson_errors(matrix, normalized1, normalized2)
    precision, recall, classification_error = classification(selected, truth)
    trial.update(
        {
            "inlier_precision": precision,
            "inlier_recall": recall,
            "normalized_model_error": float(np.median(residuals[truth]) / normalized_threshold)
            if np.any(truth)
            else 1_000_000_000.0,
            "inlier_classification_error": classification_error,
            "epipolar_matrix": matrix.tolist(),
            "inlier_indices": selected.tolist(),
            "success": True,
            "failure_reason": None,
        }
    )
    return trial


def run_homography(
    pair: dict,
    threshold: float,
    profile: str,
    seed: int,
    scoring_mode: str = "opencv_usac_magsac",
    *,
    variant: str = "default",
    threshold_scale: float = 1.0,
) -> dict:
    points1 = np.asarray(pair["points1"], dtype=np.float64)
    points2 = np.asarray(pair["points2"], dtype=np.float64)
    max_iterations, confidence = PROFILES[profile]
    method = OPENCV_METHODS[scoring_mode]
    cv2.setRNGSeed(seed & 0x7FFF_FFFF)
    start = time.perf_counter_ns()
    matrix, mask = cv2.findHomography(
        points1, points2, method, threshold, None, max_iterations, confidence
    )
    runtime_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    trial = base_trial(
        suite="homography-ransac-val",
        estimator="homography",
        scoring_mode=scoring_mode,
        variant=variant,
        threshold_scale=threshold_scale,
        profile=profile,
        scene=f"{pair['dataset']}/{pair['pair']}",
        seed=seed,
        runtime_ms=runtime_ms,
    )
    if matrix is None or mask is None or np.asarray(matrix).shape != (3, 3):
        return trial
    matrix = np.asarray(matrix, dtype=np.float64)
    selected = np.flatnonzero(np.asarray(mask).reshape(-1) != 0)
    if len(selected) < 4 or not np.isfinite(matrix).all():
        return trial
    ground_truth = np.asarray(pair["homography"], dtype=np.float64)
    truth = transfer_errors(ground_truth, points1, points2) <= threshold
    residuals = transfer_errors(matrix, points1, points2)
    targets = np.column_stack((points1, np.ones(len(points1)))) @ ground_truth.T
    targets = targets[:, :2] / targets[:, 2:3]
    quality = auc_at_threshold(transfer_errors(matrix, points1, targets), threshold)
    precision, recall, classification_error = classification(selected, truth)
    trial.update(
        {
            "inlier_precision": precision,
            "inlier_recall": recall,
            "normalized_model_error": float(np.median(residuals[truth]) / threshold)
            if np.any(truth)
            else 1_000_000_000.0,
            "inlier_classification_error": classification_error,
            "homography_auc_3": quality,
            "success": True,
            "failure_reason": None,
        }
    )
    return trial


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phototourism-input", type=Path, required=True)
    parser.add_argument("--homography-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("seeds must be positive")

    photo = json.loads(args.phototourism_input.read_text())
    homography = json.loads(args.homography_input.read_text())
    profiles = ("balanced",) if args.smoke else tuple(PROFILES)
    seeds = 1 if args.smoke else args.seeds
    trials = []
    for profile in profiles:
        for index in range(seeds):
            seed = 0x5EED_CAFE_D00D_BAAD ^ index
            for scoring_mode in OPENCV_METHODS:
                trials.extend(
                    run_fundamental(pair, photo["threshold"], profile, seed, scoring_mode)
                    for pair in photo["pairs"]
                )
                trials.extend(
                    run_essential(pair, photo["threshold"], profile, seed, scoring_mode)
                    for pair in photo["pairs"]
                )
                trials.extend(
                    run_homography(pair, homography["threshold"], profile, seed, scoring_mode)
                    for pair in homography["pairs"]
                )
            if profile == "balanced":
                trials.extend(
                    run_essential(
                        pair,
                        photo["threshold"],
                        profile,
                        seed,
                        "opencv_usac_magsac",
                        variant="essential_threshold_sweep",
                        threshold_scale=scale,
                    )
                    for scale in ESSENTIAL_THRESHOLD_SCALES
                    for pair in photo["pairs"]
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(trial, allow_nan=False) + "\n" for trial in trials))
    print(f"ran {len(trials)} OpenCV reference trial(s): {args.output}")


if __name__ == "__main__":
    main()
