#!/usr/bin/env python3
"""Aggregate raw benchmark JSONL into a stable, dashboard-friendly summary."""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


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


def main(raw_path: str, output_path: str) -> None:
    records = [json.loads(line) for line in Path(raw_path).read_text().splitlines() if line]
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for record in records:
        key = tuple(record[field] for field in ("estimator", "scoring_mode", "profile", "scene"))
        groups[key].append(record)

    summaries = []
    for key, trials in sorted(groups.items()):
        estimator, scoring_mode, profile, scene = key
        successes = [trial["success"] for trial in trials]
        summaries.append(
            {
                "estimator": estimator,
                "scoring_mode": scoring_mode,
                "profile": profile,
                "scene": scene,
                "trials": len(trials),
                "success_rate": sum(successes) / len(successes),
                "median_runtime_ms": median([trial["runtime_ms"] for trial in trials]),
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
            }
        )

    frontiers = {}
    for estimator in sorted({row["estimator"] for row in summaries}):
        for scene in sorted({row["scene"] for row in summaries}):
            rows = [row for row in summaries if row["estimator"] == estimator and row["scene"] == scene]
            frontiers[f"{estimator}/{scene}"] = pareto(rows)

    result = {"schema_version": 1, "groups": summaries, "pareto_frontiers": frontiers}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
