#!/usr/bin/env python3
"""Evaluate EPOS absolute-pose trials with a scale-normalized pose error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def pose_error_degrees(pair: dict, trial: dict) -> float:
    """Return max(rotation error, scale-normalized translation error) in degrees.

    Translation is converted with atan2(||dt||, ||t_gt||), so the metric is
    dimensionless and shares the 10-degree AUC scale used by relative pose.
    """

    pose = trial.get("absolute_pose")
    if pose is None:
        return 180.0
    try:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (3, 4) or not np.isfinite(pose).all():
            return 180.0
        rotation = pose[:, :3]
        translation = pose[:, 3]
        rotation_gt = np.asarray(pair["rotation"], dtype=np.float64)
        translation_gt = np.asarray(pair["translation"], dtype=np.float64)
        trace = np.trace(rotation_gt.T @ rotation)
        rotation_error = np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))
        translation_scale = max(float(np.linalg.norm(translation_gt)), 1e-12)
        translation_error = np.degrees(
            np.arctan2(np.linalg.norm(translation - translation_gt), translation_scale)
        )
        error = max(float(rotation_error), float(translation_error))
        return error if np.isfinite(error) else 180.0
    except (KeyError, ValueError):
        return 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    pairs = {pair["scene"]: pair for pair in json.loads(args.input.read_text())["pairs"]}
    trials = [json.loads(line) for line in args.results.read_text().splitlines() if line]
    evaluated = 0
    for trial in trials:
        if trial["suite"] != "epos-pnp-val":
            continue
        trial["pose_error_deg"] = pose_error_degrees(pairs[trial["scene"]], trial)
        # EPOS quality is pose AUC, not correspondence-label precision/recall.
        trial["success"] = trial["pose_error_deg"] <= 10.0
        evaluated += 1
    args.results.write_text("".join(json.dumps(trial) + "\n" for trial in trials))
    print(f"evaluated {evaluated} EPOS PnP pose trial(s)")


if __name__ == "__main__":
    main()
