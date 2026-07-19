#!/usr/bin/env python3
"""Compatibility tests for sampler-aware benchmark aggregation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import aggregate
import regress


def trial(sampler: str | None) -> dict:
    result = {
        "suite": "phototourism-val",
        "estimator": "fundamental",
        "scoring_mode": "ransac",
        "profile": "balanced",
        "scene": "scene/pair",
        "success": True,
        "runtime_ms": 10.0,
        "iterations": 1000,
        "normalized_model_error": 0.2,
        "inlier_classification_error": 0.1,
        "inlier_precision": 0.95,
        "inlier_recall": 0.95,
        "pose_error_deg": 2.0,
        "homography_auc_3": None,
        "diagnostics": {
            "sampling_attempts": 1000,
            "rejected_samples": 2,
            "model_estimation_failures": 3,
            "candidate_models": 997,
            "rejected_models": 4,
            "scored_models": 993,
            "local_optimization_runs": 5,
            "final_optimization_runs": 1,
            "inlier_ratio": 0.7,
        },
    }
    if sampler is not None:
        result["sampler"] = sampler
    return result


class AggregateTests(unittest.TestCase):
    def test_sampler_is_a_group_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.jsonl"
            output_path = Path(directory) / "summary.json"
            raw_path.write_text(
                "\n".join(json.dumps(record) for record in [trial("uniform"), trial("prosac")])
            )
            aggregate.main(str(raw_path), str(output_path))
            groups = json.loads(output_path.read_text())["groups"]

        self.assertEqual({group["sampler"] for group in groups}, {"uniform", "prosac"})
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["mean_sampling_attempts"], 1000.0)
        self.assertEqual(groups[0]["mean_inlier_ratio"], 0.7)

    def test_legacy_group_defaults_to_prosac(self) -> None:
        summary = {"groups": [trial(None)]}
        key = next(iter(regress.index(summary)))
        self.assertEqual(key[3], "prosac")


if __name__ == "__main__":
    unittest.main()
