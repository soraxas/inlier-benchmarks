"""Tests for the independent OpenCV USAC/MAGSAC reference adapter."""

from __future__ import annotations

import unittest

import numpy as np

from python.run_opencv_reference import run_essential, run_fundamental, run_homography


def fundamental_pair() -> dict:
    # USAC's exact-minimal behavior differs between OpenCV builds. Use a
    # comfortably non-minimal, non-planar scene to test our adapter instead.
    points_3d = np.array(
        [
            (x * 0.23, y * 0.19, 3.0 + 0.13 * (x * x + y * y) + 0.07 * x * y)
            for x in range(-2, 3)
            for y in range(-2, 3)
        ],
        dtype=np.float64,
    )
    translation = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    points1 = points_3d[:, :2] / points_3d[:, 2:]
    shifted = points_3d + translation
    points2 = shifted[:, :2] / shifted[:, 2:]
    essential = [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    return {
        "scene": "synthetic",
        "pair": "pair",
        "points1": points1.tolist(),
        "points2": points2.tolist(),
        "fundamental": essential,
        "essential": essential,
        "intrinsics1": np.eye(3).tolist(),
        "intrinsics2": np.eye(3).tolist(),
    }


def homography_pair() -> dict:
    points1 = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    homography = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])
    points2 = points1 + np.array([3.0, -2.0])
    return {
        "dataset": "synthetic",
        "pair": "pair",
        "points1": points1.tolist(),
        "points2": points2.tolist(),
        "homography": homography.tolist(),
    }


class OpenCvReferenceTests(unittest.TestCase):
    def test_fundamental_reference_emits_pose_payload(self) -> None:
        trial = run_fundamental(fundamental_pair(), 1e-3, "balanced", 7)
        self.assertTrue(trial["success"])
        self.assertEqual(trial["estimator"], "fundamental")
        self.assertGreaterEqual(len(trial["inlier_indices"]), 5)
        self.assertEqual(np.asarray(trial["epipolar_matrix"]).shape, (3, 3))

    def test_homography_reference_uses_the_shared_quality_metric(self) -> None:
        trial = run_homography(homography_pair(), 1e-4, "balanced", 7)
        self.assertTrue(trial["success"])
        self.assertGreater(trial["homography_auc_3"], 0.99)
        self.assertEqual(trial["scoring_mode"], "opencv_usac_magsac")

    def test_essential_reference_emits_pose_payload(self) -> None:
        trial = run_essential(fundamental_pair(), 1e-3, "balanced", 7)
        self.assertTrue(trial["success"])
        self.assertEqual(trial["estimator"], "essential")
        self.assertGreaterEqual(len(trial["inlier_indices"]), 5)


if __name__ == "__main__":
    unittest.main()
