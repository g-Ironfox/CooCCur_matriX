"""Tests for streaming weighted word aggregation and CLI reports."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import weighted_word_stats as stats


class WeightedWordStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "weighted.csv"
        self.source.write_text(
            "vertex,a,b,c\na,5,2,1\nb,0,4,3\nc,0,0,0\n", encoding="utf-8")

    def test_direction_and_diagonal_policy(self):
        rows = stats.summarize(self.source)
        self.assertEqual([r["word"] for r in rows], ["c", "b", "a"])
        by_word = {row["word"]: row for row in rows}
        self.assertEqual(by_word["c"]["incoming_score"], 4)
        self.assertEqual(by_word["c"]["outgoing_score"], 0)
        self.assertEqual(by_word["c"]["incoming_reachable"], 2)
        self.assertEqual(by_word["a"]["outgoing_score"], 3)
        self.assertEqual(by_word["a"]["self_score"], 5)
        included = {r["word"]: r for r in stats.summarize(self.source, True)}
        self.assertEqual(included["a"]["incoming_score"], 5)
        self.assertEqual(included["a"]["outgoing_score"], 8)
        self.assertEqual(included["a"]["outgoing_reachable"], 3)
        self.assertEqual([r["word"] for r in stats.summarize(self.source, sort_by="outgoing")],
                         ["a", "b", "c"])

    def test_cli_top_only_limits_plot_and_keeps_all_csv_words(self):
        output = self.root / "out"
        result = subprocess.run(
            [sys.executable, str(Path(stats.__file__).resolve()), str(self.source),
             "--outdir", str(output), "--top", "2", "--sort-by", "min"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "weighted_word_stats.csv").open(encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["word"], "b")
        self.assertEqual(rows[0]["min_score"], "2")
        self.assertGreater((output / "weighted_word_stats.png").stat().st_size, 0)

    def test_minimum_uses_row_and_column_totals_and_diagonal_policy(self):
        rows = stats.summarize(self.source, sort_by="min")
        self.assertEqual([r["word"] for r in rows], ["b", "a", "c"])
        self.assertEqual([r["min_score"] for r in rows], [2, 0, 0])
        included = stats.summarize(self.source, True, sort_by="min")
        self.assertEqual([r["word"] for r in included], ["b", "a", "c"])
        self.assertEqual([r["min_score"] for r in included], [6, 5, 0])

    def test_invalid_matrices_are_rejected(self):
        for content in ["vertex,a,b\na,0,1\n", "vertex,a,b\nb,0,1\na,0,0\n",
                        "vertex,a\na,-1\n", "vertex,a\na,nan\n"]:
            with self.subTest(content=content):
                self.source.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    stats.summarize(self.source)

    def test_all_zero_matrix_is_valid(self):
        self.source.write_text("vertex,a\na,0\n", encoding="utf-8")
        rows = stats.run(self.source, self.root / "zero", top=0)
        self.assertEqual(rows[0]["incoming_score"], 0)
        self.assertEqual(rows[0]["min_score"], 0)


if __name__ == "__main__":
    unittest.main()
