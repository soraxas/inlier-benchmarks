"""Regression tests for PhotoTourism pose evaluation and AUC aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from python.aggregate import pose_auc_at_10
from python.evaluate_phototourism import pose_error_degrees


def calibrated_translation_pair() -> dict:
    points_3d = np.array(
        [
            [-0.5, -0.3, 3.0],
            [0.2, -0.4, 4.0],
            [0.7, 0.1, 5.0],
            [-0.3, 0.5, 3.5],
            [0.4, 0.6, 4.5],
        ],
        dtype=np.float64,
    )
    translation = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    points1 = points_3d[:, :2] / points_3d[:, 2:]
    points2_3d = points_3d + translation
    points2 = points2_3d[:, :2] / points2_3d[:, 2:]
    essential = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    return {
        "points1": points1.tolist(),
        "points2": points2.tolist(),
        "intrinsics1": np.eye(3).tolist(),
        "intrinsics2": np.eye(3).tolist(),
        "relative_rotation": np.eye(3).tolist(),
        "relative_translation": translation.tolist(),
        "fundamental": essential.tolist(),
        "essential": essential.tolist(),
    }


class PhotoTourismEvaluationTests(unittest.TestCase):
    def test_ground_truth_fundamental_and_essential_have_zero_pose_error(self) -> None:
        pair = calibrated_translation_pair()
        indices = list(range(len(pair["points1"])))
        for estimator in ("fundamental", "essential"):
            error = pose_error_degrees(
                pair,
                {
                    "estimator": estimator,
                    "epipolar_matrix": pair[estimator],
                    "inlier_indices": indices,
                },
            )
            self.assertLess(error, 1e-8)

    def test_missing_pose_payload_is_a_complete_failure(self) -> None:
        self.assertEqual(pose_error_degrees(calibrated_translation_pair(), {}), 180.0)

    def test_auc_at_ten_uses_the_reference_cutoff_convention(self) -> None:
        self.assertAlmostEqual(pose_auc_at_10([0.0, 10.0, 20.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
