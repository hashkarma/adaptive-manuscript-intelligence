import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.main import (
    load_execution_profile_config,
    normalize_execution_profile,
    stage4_parameters_for_execution_profile,
)


class PlatformExecutionProfileTest(unittest.TestCase):
    def test_generic_default_preserves_layer_defaults(self):
        self.assertEqual(
            stage4_parameters_for_execution_profile("generic_default"),
            {},
        )

    def test_demo_balanced_loads_validated_stage4_parameters(self):
        expected = {
            "profile_smooth_window": 5,
            "peak_threshold_ratio": 0.22,
            "peak_min_distance_ratio": 0.03,
            "physical_peak_merge_distance_ratio": 0.04,
            "shallow_valley_ratio": 0.38,
            "minimum_line_height_ratio": 0.025,
        }
        self.assertEqual(
            stage4_parameters_for_execution_profile("demo_balanced"),
            expected,
        )

    def test_demo_balanced_source_file_matches(self):
        profile = load_execution_profile_config("demo_balanced")
        self.assertEqual(profile.get("profile_id"), "demo_balanced")
        self.assertEqual(
            profile["stage4"]["expected_golden_line_count"],
            11,
        )

    def test_invalid_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_execution_profile("not-a-real-profile")


if __name__ == "__main__":
    unittest.main(verbosity=2)
