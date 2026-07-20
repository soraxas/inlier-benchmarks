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


def paired_quality_delta(
    suite: str, pairs: list[tuple[dict, dict]]
) -> tuple[float | None, float | None, int]:
    """Return quality(target) - quality(fast) from matched scene/seed trials."""
    if not pairs:
        return None, None, 0
    if suite == "homography-ransac-val":
        deltas = [
            target["homography_auc_3"] - baseline["homography_auc_3"]
            for target, baseline in pairs
            if target.get("homography_auc_3") is not None
            and baseline.get("homography_auc_3") is not None
        ]
        return (
            statistics.fmean(deltas) if deltas else None,
            standard_error(deltas) if deltas else None,
            len(deltas),
        )

    if suite == "phototourism-val":
        errors = [
            (target.get("pose_error_deg"), baseline.get("pose_error_deg"))
            for target, baseline in pairs
            if target.get("pose_error_deg") is not None
            and baseline.get("pose_error_deg") is not None
        ]
        if not errors:
            return None, None, 0
        point_estimate = pose_auc_at_10([target for target, _ in errors]) - pose_auc_at_10(
            [baseline for _, baseline in errors]
        )
        generator = random.Random(0)
        samples = []
        for _ in range(250):
            sample = [errors[generator.randrange(len(errors))] for _ in errors]
            samples.append(
                pose_auc_at_10([target for target, _ in sample])
                - pose_auc_at_10([baseline for _, baseline in sample])
            )
        return point_estimate, statistics.stdev(samples), len(errors)

    return None, None, 0


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
        diagnostics = [trial["diagnostics"] for trial in trials if trial.get("diagnostics")]
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
                "mean_sampling_attempts": statistics.fmean(
                    diagnostic["sampling_attempts"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
                "mean_rejected_samples": statistics.fmean(
                    diagnostic["rejected_samples"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
                "mean_model_estimation_failures": statistics.fmean(
                    diagnostic["model_estimation_failures"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
                "mean_scored_models": statistics.fmean(
                    diagnostic["scored_models"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
                "mean_local_optimization_runs": statistics.fmean(
                    diagnostic["local_optimization_runs"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
                "mean_inlier_ratio": statistics.fmean(
                    diagnostic["inlier_ratio"] for diagnostic in diagnostics
                )
                if diagnostics
                else None,
            }
        )
        summaries.append(summary)
    return summaries


def add_paired_deltas(summaries: list[dict], records: list[dict]) -> None:
    """Annotate every group with its matched quality difference from fast."""
    fast = {
        (
            trial["suite"],
            trial["estimator"],
            trial["scoring_mode"],
            trial["sampler"],
            trial["scene"],
            trial["seed"],
        ): trial
        for trial in records
        if trial["profile"] == "fast"
    }
    for summary in summaries:
        trials = [
            trial
            for trial in records
            if all(
                trial[field] == summary[field]
                for field in ("suite", "estimator", "scoring_mode", "sampler", "profile")
            )
            and ("scene" not in summary or trial["scene"] == summary["scene"])
        ]
        pairs = [
            (trial, fast[key])
            for trial in trials
            if (
                key := (
                    trial["suite"],
                    trial["estimator"],
                    trial["scoring_mode"],
                    trial["sampler"],
                    trial["scene"],
                    trial["seed"],
                )
            )
            in fast
        ]
        delta, delta_se, samples = paired_quality_delta(summary["suite"], pairs)
        summary["paired_auc_delta_vs_fast"] = delta
        summary["paired_auc_delta_vs_fast_se"] = delta_se
        summary["paired_auc_samples"] = samples


def main(raw_path: str, output_path: str) -> None:
    records = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line]
    # The original benchmark runner always used PROSAC but did not serialize it.
    # Preserve those historic artifacts as a separately labelled baseline.
    for record in records:
        record.setdefault("sampler", "prosac")
    summaries = summarize(
        records, ("suite", "estimator", "scoring_mode", "sampler", "profile", "scene")
    )
    dataset_summaries = summarize(
        records, ("suite", "estimator", "scoring_mode", "sampler", "profile")
    )
    add_paired_deltas(summaries, records)
    add_paired_deltas(dataset_summaries, records)

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
