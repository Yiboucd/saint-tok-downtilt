"""CPU-only checks against the frozen data used by the current paper table."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tables" / "make_table1.py"
SPEC = importlib.util.spec_from_file_location("make_table1", SCRIPT)
table = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = table
SPEC.loader.exec_module(table)


class MainTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = table.build_rows()

    def test_current_paper_rows_and_rounded_metrics(self):
        expected = [
            ("Random", (-0.004, -0.024, 0.021)),
            ("Always -1deg", (0.146, 0.136, 0.159)),
            ("Always +1deg", (-0.178, -0.171, -0.186)),
            ("DQN", (0.184, 0.238, 0.117)),
            ("Identity-token PPO", (0.262, 0.232, 0.300)),
            ("Identity-token QMIX", (0.342, 0.354, 0.327)),
            ("BDQN", (0.363, 0.355, 0.372)),
            ("Identity-token double-Q", (0.336, 0.381, 0.280)),
            ("Identity-token SAC", (0.376, 0.416, 0.326)),
            ("SAINT-Tok (SAC)", (0.464, 0.467, 0.459)),
        ]
        actual = [(row.label, tuple(round(value, 3) for value in row.metrics))
                  for row in self.rows]
        self.assertEqual(actual, expected)
        self.assertTrue(all(row.per_case.shape == (90,) for row in self.rows))
        self.assertEqual([row.training_seeds for row in self.rows], [0, 0, 0] + [3] * 7)

    def test_missing_or_extra_training_seed_is_rejected(self):
        source = table.load(ROOT / "results" / "v12_se_flat_eval.json")
        missing = copy.deepcopy(source)
        del missing["flat_s2"]
        with self.assertRaises(AssertionError):
            table.learned_row("DQN", missing, "flat_s")
        extra = copy.deepcopy(source)
        extra["flat_s3"] = copy.deepcopy(extra["flat_s0"])
        with self.assertRaises(AssertionError):
            table.learned_row("DQN", extra, "flat_s")

    def test_wrong_case_count_or_split_summary_is_rejected(self):
        record = table.load(ROOT / "results" / "v12_se_flat_eval.json")["flat_s0"]
        shortened = copy.deepcopy(record)
        shortened["per_case_dse"].pop()
        with self.assertRaises(AssertionError):
            table.checked_record(shortened, "DQN")
        wrong_split = copy.deepcopy(record)
        wrong_split["dSE_ood"] += 0.01
        with self.assertRaises(AssertionError):
            table.checked_record(wrong_split, "DQN")

    def test_fixed_actions_use_replayed_per_case_gains(self):
        replay = table.load(ROOT / "data" / table.FIXED_ACTION_FILE)
        np.testing.assert_array_equal(self.rows[1].per_case, replay["per_case"]["minus1_gain"])
        np.testing.assert_array_equal(self.rows[2].per_case, replay["per_case"]["plus1_gain"])
        wrong_order = copy.deepcopy(replay)
        wrong_order["cases"][50]["group"] = "seen"
        with self.assertRaises(AssertionError):
            table.fixed_action_rows(wrong_order)
        wrong_gain = copy.deepcopy(replay)
        wrong_gain["per_case"]["plus1_gain"][0] += 0.01
        with self.assertRaises(AssertionError):
            table.fixed_action_rows(wrong_gain)

    def test_paired_difference_and_case_win_rate(self):
        mean, low, high, wins = table.paired_bootstrap(
            self.rows[-1].per_case, self.rows[-2].per_case
        )
        self.assertAlmostEqual(mean, 0.0875762639222322)
        self.assertAlmostEqual(wins, 100 * 71 / 90)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_cli_works_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as other_directory:
            result = subprocess.run(
                [sys.executable, str(SCRIPT)], cwd=other_directory,
                check=True, capture_output=True, text=True,
            )
        self.assertIn("Always +1deg", result.stdout)
        self.assertIn("SAINT-Tok (SAC)", result.stdout)
        for obsolete in ("Best constant", "gamma=0", "tailored", "Tok3 + double-Q", "retained"):
            self.assertNotIn(obsolete, result.stdout)


if __name__ == "__main__":
    unittest.main()
