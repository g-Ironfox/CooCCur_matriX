"""Tests for direct in/out scatter plots from weighted matrices."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import weighted_inout_scatter as scatter


class WeightedInoutScatterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "weighted_004.csv"
        self.source.write_text(
            "vertex,a,b,c,d\na,5,10,0,0\nb,0,0,6,0\n"
            "c,0,0,0,4\nd,0,0,0,0\n", encoding="utf-8")

    def test_sums_use_matrix_weights_and_exclude_diagonal_by_default(self):
        rows = {row["word"]: row for row in scatter.summarize_matrix(self.source)}
        self.assertEqual(rows["a"]["outgoing_score"], 10)
        self.assertEqual(rows["a"]["incoming_score"], 0)
        self.assertEqual(rows["a"]["max_score"], 10)
        self.assertEqual(rows["a"]["directional_imbalance"], -1)
        self.assertEqual(rows["b"]["incoming_score"], 10)
        self.assertEqual(rows["b"]["directional_imbalance"], 0.25)
        included = {row["word"]: row for row in
                    scatter.summarize_matrix(self.source, include_diagonal=True)}
        self.assertEqual(included["a"]["incoming_score"], 5)
        self.assertEqual(included["a"]["outgoing_score"], 15)

    def test_fractional_weights_are_supported(self):
        self.source.write_text(
            "vertex,a,b,c,d\na,0,0.5,0,0\nb,0,0,0.25,0\n"
            "c,0,0,0,0\nd,0,0,0,0\n", encoding="utf-8")
        rows = {row["word"]: row for row in scatter.summarize_matrix(self.source)}
        self.assertEqual(rows["a"]["outgoing_score"], 0.5)
        self.assertEqual(rows["b"]["incoming_score"], 0.5)
        self.assertEqual(rows["b"]["outgoing_score"], 0.25)

    def test_sort_by_modes(self):
        self.assertEqual(
            [row["word"] for row in scatter.summarize_matrix(self.source, sort_by="max")],
            ["a", "b", "c", "d"])
        self.assertEqual(
            [row["word"] for row in scatter.summarize_matrix(self.source, sort_by="min")],
            ["b", "c", "a", "d"])
        self.assertEqual(
            [row["word"] for row in scatter.summarize_matrix(self.source, sort_by="total")],
            ["b", "a", "c", "d"])

    def test_cli_writes_full_csv_and_scatter(self):
        output = self.root / "out"
        result = subprocess.run(
            [sys.executable, str(Path(scatter.__file__).resolve()), str(self.source),
             "--outdir", str(output), "--top", "2", "--sort-by", "total"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "weighted_004_inout_stats.csv").open(encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["word"], "b")
        self.assertIn("directional_imbalance", rows[0])
        self.assertTrue((output / "weighted_004_inout_scatter.png").is_file())
        self.assertGreater((output / "weighted_004_inout_scatter.png").stat().st_size, 0)

    def test_page_size_defaults_to_one_image_and_can_split(self):
        default_output = self.root / "default_pages"
        scatter.run(self.source, default_output, top=0)
        self.assertEqual(len(list(default_output.glob("*_inout_scatter*.png"))), 1)

        paged_output = self.root / "paged"
        scatter.run(self.source, paged_output, top=0, page_size=2)
        self.assertEqual(len(list(paged_output.glob("*_inout_scatter*.png"))), 2)

    def test_negative_page_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "--page-size"):
            scatter.run(self.source, self.root / "invalid", page_size=-1)


if __name__ == "__main__":
    unittest.main()
