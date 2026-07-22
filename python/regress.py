#!/usr/bin/env python3
"""Gate fixed benchmark groups against a baseline summary."""

import json
import sys
from pathlib import Path

SUCCESS_DROP = 0.05
ITERATION_INCREASE = 0.20
PAIRED_MIN_SAMPLES = 8
PAIRED_REGRESSION_SIGMAS = 2.5


def paired_quality_regressed(candidate: dict, previous: dict) -> bool:
    """Detect a significant loss of quality relative to the same fast trials."""
    if candidate.get("profile") == "fast":
        return False
    current_delta = candidate.get("paired_auc_delta_vs_fast")
    previous_delta = previous.get("paired_auc_delta_vs_fast")
    current_samples = candidate.get("paired_auc_samples", 0)
    previous_samples = previous.get("paired_auc_samples", 0)
    if (
        current_delta is None
        or previous_delta is None
        or current_samples < PAIRED_MIN_SAMPLES
        or previous_samples < PAIRED_MIN_SAMPLES
    ):
        return False
    current_se = candidate.get("paired_auc_delta_vs_fast_se") or 0.0
    previous_se = previous.get("paired_auc_delta_vs_fast_se") or 0.0
    tolerance = PAIRED_REGRESSION_SIGMAS * (current_se**2 + previous_se**2) ** 0.5
    return current_delta < previous_delta - tolerance


def index(summary: dict) -> dict[tuple[str, ...], dict]:
    def key(row: dict) -> tuple[str, ...]:
        return (
            row.get("suite", "public-api"),
            row["estimator"],
            row["scoring_mode"],
            row.get("sampler", "prosac"),
            row.get("variant", "default"),
            row.get("threshold_scale", 1.0),
            row["profile"],
            row["scene"],
        )

    return {
        key(row): row
        for row in summary["groups"]
    }


def main(current_path: str, baseline_path: str) -> None:
    if not Path(baseline_path).exists():
        print("No baseline available; reporting only.")
        return
    current = index(json.loads(Path(current_path).read_text()))
    baseline = index(json.loads(Path(baseline_path).read_text()))
    failures = []
    for key, previous in baseline.items():
        candidate = current.get(key)
        if candidate is None:
            continue
        quality_improved = candidate["success_rate"] > previous["success_rate"]
        if candidate["success_rate"] < previous["success_rate"] - SUCCESS_DROP:
            failures.append(f"{key}: success rate regressed")
        if (
            previous["median_iterations"] > 0
            and candidate["median_iterations"]
            > previous["median_iterations"] * (1 + ITERATION_INCREASE)
            and not quality_improved
        ):
            failures.append(f"{key}: median iterations regressed")
        if paired_quality_regressed(candidate, previous):
            failures.append(f"{key}: paired quality versus fast regressed")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Benchmark regression gate passed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
