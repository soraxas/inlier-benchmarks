import unittest

from python.run_opencv_pnp_reference import trial


class OpenCvPnpReferenceTests(unittest.TestCase):
    def test_reference_emits_pose_payload(self) -> None:
        pair = {
            "scene": "unit",
            "points3d": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "points2d": [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        }
        result = trial(pair, "balanced", 42, 1e-6)
        self.assertEqual(result["scoring_mode"], "opencv_ransac")
        self.assertEqual(result["sampler"], "opencv")
        self.assertIsNotNone(result["absolute_pose"])


if __name__ == "__main__":
    unittest.main()
