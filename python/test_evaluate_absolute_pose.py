import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from python.evaluate_absolute_pose import pose_error_degrees
from python.prepare_epos_pnp import parse_case


class EvaluateAbsolutePoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = {
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation": [0.0, 0.0, 2.0],
        }

    def test_ground_truth_pose_has_zero_error(self) -> None:
        trial = {"absolute_pose": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]]}
        self.assertAlmostEqual(pose_error_degrees(self.pair, trial), 0.0)

    def test_missing_pose_is_complete_failure(self) -> None:
        self.assertEqual(pose_error_degrees(self.pair, {}), 180.0)

    def test_parser_accepts_epos_extended_confidence_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "case.txt"
            path.write_text(
                "2 1 1\n"
                "500 0 320\n0 500 240\n0 0 1\n"
                "1\n"
                "1 0 0 0\n0 1 0 0\n0 0 1 2\n"
                "3\n"
                "320 240 0 0 1 1 2 .9 .8 .7\n"
                "321 240 .1 0 1 1 2 .9 .8 .7 .6 .5\n"
                "319 240 -.1 0 1 1 2 .9 .8 .7\n"
                "0\n"
            )
            parsed = parse_case(path)
        self.assertEqual(parsed["rows"].shape, (3, 10))


if __name__ == "__main__":
    unittest.main()
