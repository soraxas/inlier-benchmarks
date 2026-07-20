"""Tests for benchmark regression decisions."""

from __future__ import annotations

import unittest

from python.regress import paired_quality_regressed


def row(delta: float, error: float, samples: int = 16) -> dict:
    return {
        "profile": "balanced",
        "paired_auc_delta_vs_fast": delta,
        "paired_auc_delta_vs_fast_se": error,
        "paired_auc_samples": samples,
    }


class RegressionTests(unittest.TestCase):
    def test_significant_paired_quality_loss_is_rejected(self) -> None:
        self.assertTrue(paired_quality_regressed(row(-0.10, 0.01), row(0.10, 0.01)))

    def test_overlapping_paired_uncertainty_is_not_rejected(self) -> None:
        self.assertFalse(paired_quality_regressed(row(0.02, 0.04), row(0.08, 0.04)))

    def test_small_smoke_sample_is_not_used_for_paired_gate(self) -> None:
        self.assertFalse(paired_quality_regressed(row(-0.10, 0.01, 1), row(0.10, 0.01, 1)))


if __name__ == "__main__":
    unittest.main()
