#!/usr/bin/env python3
"""Gate fixed benchmark groups against a baseline summary."""

import json
import sys
from pathlib import Path

SUCCESS_DROP = 0.05
ITERATION_INCREASE = 0.20


def index(summary: dict) -> dict[tuple[str, ...], dict]:
    def key(row: dict) -> tuple[str, ...]:
        return (
            row.get("suite", "public-api"),
            row["estimator"],
            row["scoring_mode"],
            row.get("sampler", "prosac"),
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
    if failures:
        raise SystemExit("\n".join(failures))
    print("Benchmark regression gate passed.")


if __name__ == "__main__":
    main(*sys.argv[1:])
