#!/usr/bin/env python3
"""Aggregate raw benchmark JSONL into a stable, dashboard-friendly summary."""

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def pose_auc_at_10(errors: list[float]) -> float | None:
    if not errors:
        return None
    points = [(0.0, 0.0)]
    for index, error in enumerate(sorted(errors), start=1):
        if error > 10.0:
            break
        points.append((error, index / len(errors)))
    points.append((10.0, points[-1][1]))
    area = sum(
        (right_x - left_x) * (left_y + right_y) * 0.5
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / 10.0


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def pose_auc_standard_error(errors: list[float]) -> float | None:
    """Estimate AUC uncertainty with a deterministic non-parametric bootstrap."""
    if not errors:
        return None
    if len(errors) == 1:
        return 0.0
    generator = random.Random(0)
    samples = [
        pose_auc_at_10([generator.choice(errors) for _ in errors])
        for _ in range(250)
    ]
    return statistics.stdev(samples)


def pareto(rows: list[dict]) -> list[dict]:
    """Keep points not dominated by higher success and lower iteration cost."""
    result = []
    for row in rows:
        dominated = any(
            other is not row
            and other["success_rate"] >= row["success_rate"]
            and other["median_iterations"] <= row["median_iterations"]
            and (
                other["success_rate"] > row["success_rate"]
                or other["median_iterations"] < row["median_iterations"]
            )
            for other in rows
        )
        if not dominated:
            result.append(row)
    return result


def summarize(records: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in fields)
        groups[key].append(record)

    summaries = []
    for key, trials in sorted(groups.items()):
        summary = dict(zip(fields, key))
        successes = [trial["success"] for trial in trials]
        runtimes = [trial["runtime_ms"] for trial in trials]
        pose_errors = [
            trial["pose_error_deg"]
            for trial in trials
            if trial.get("pose_error_deg") is not None
        ]
        homography_aucs = [
            trial["homography_auc_3"]
            for trial in trials
            if trial.get("homography_auc_3") is not None
        ]
        summary.update(
            {
                "trials": len(trials),
                "success_rate": sum(successes) / len(successes),
                "mean_runtime_ms": statistics.fmean(runtimes),
                "runtime_se_ms": standard_error(runtimes),
                "median_runtime_ms": median(runtimes),
                "median_iterations": median([trial["iterations"] for trial in trials]),
                "median_normalized_model_error": median(
                    [trial["normalized_model_error"] for trial in trials]
                ),
                "median_inlier_classification_error": median(
                    [trial["inlier_classification_error"] for trial in trials]
                ),
                "median_precision": median([trial["inlier_precision"] for trial in trials]),
                "median_recall": median([trial["inlier_recall"] for trial in trials]),
                "failures": sum(not success for success in successes),
                "scene_count": len({trial["scene"] for trial in trials}),
                "auc_pose_10": pose_auc_at_10(pose_errors),
                "auc_pose_10_se": pose_auc_standard_error(pose_errors),
                "auc_homography_3": statistics.fmean(homography_aucs)
                if homography_aucs
                else None,
                "auc_homography_3_se": standard_error(homography_aucs)
                if homography_aucs
                else None,
            }
        )
        summaries.append(summary)
    return summaries


def main(raw_path: str, output_path: str) -> None:
    records = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line]
    summaries = summarize(
        records, ("suite", "estimator", "scoring_mode", "profile", "scene")
    )
    dataset_summaries = summarize(records, ("suite", "estimator", "scoring_mode", "profile"))

    frontiers = {}
    for suite in sorted({row["suite"] for row in summaries}):
        for estimator in sorted({row["estimator"] for row in summaries}):
            for scene in sorted({row["scene"] for row in summaries}):
                rows = [
                    row
                    for row in summaries
                    if row["suite"] == suite
                    and row["estimator"] == estimator
                    and row["scene"] == scene
                ]
                if rows:
                    frontiers[f"{suite}/{estimator}/{scene}"] = pareto(rows)

    result = {
        "schema_version": 2,
        "groups": summaries,
        "dataset_groups": dataset_summaries,
        "pareto_frontiers": frontiers,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
