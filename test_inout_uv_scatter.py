"""Tests for rotated in/out score visualizations."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import inout_uv_scatter as uv


class InoutUVScatterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "weighted_002.csv"
        self.source.write_text(
            "vertex,a,b,c\na,0,1,1\nb,3,0,0\nc,0,0,0\n", encoding="utf-8")

    def test_rotation_coordinates_and_projection_order(self):
        rows = uv.run(self.source, self.root / "out", top=0)
        by_word = {row["word"]: row for row in rows}
        self.assertAlmostEqual(by_word["a"]["incoming_score"], 3)
        self.assertAlmostEqual(by_word["a"]["outgoing_score"], 2)
        self.assertAlmostEqual(by_word["a"]["u_score"], 2.5)
        self.assertAlmostEqual(by_word["a"]["v_score"], 0.5)
        self.assertEqual([row["word"] for row in rows], ["a", "b", "c"])

    def test_cli_writes_stats_and_plot(self):
        output = self.root / "out"
        result = subprocess.run(
            [sys.executable, str(Path(uv.__file__).resolve()), str(self.source),
             "--outdir", str(output), "--top", "2", "--sort-by", "projection"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "weighted_002_uv_stats.csv").open(encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 3)
        self.assertIn("u_score", rows[0])
        self.assertIn("v_score", rows[0])
        self.assertEqual(rows[0]["word"], "a")
        image = output / "weighted_002_uv_scatter.png"
        self.assertTrue(image.is_file())
        self.assertGreater(image.stat().st_size, 0)

    def test_invalid_sort_and_negative_page_size_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sort_by"):
            uv.transform_rows([], "total")
        with self.assertRaisesRegex(ValueError, "--page-size"):
            uv.run(self.source, self.root / "invalid", page_size=-1)


if __name__ == "__main__":
    unittest.main()
