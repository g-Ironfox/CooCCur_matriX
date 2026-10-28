"""Tests for weighted TextRank and its command-line outputs."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import textrank


class TextRankTests(unittest.TestCase):
    def test_weighted_direction_and_score_normalization(self):
        scores, _ = textrank.textrank(np.array([[0, 3, 0], [0, 0, 1], [0, 0, 0]], dtype=float))
        self.assertAlmostEqual(float(scores.sum()), 1.0)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[1], scores[0])

    def test_symmetric_graph_favors_higher_weighted_degree(self):
        scores, _ = textrank.textrank(np.array([[0, 2, 0], [2, 0, 1], [0, 1, 0]], dtype=float))
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[0], scores[2])

    def test_invalid_parameters_and_weights(self):
        for kwargs in ({"damping": 1}, {"tolerance": 0}, {"max_iterations": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                textrank.textrank(np.zeros((2, 2)), **kwargs)
        with self.assertRaises(ValueError):
            textrank.textrank(np.array([[0, -1], [0, 0]]))

    def test_cli_outputs_full_ranked_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "graph.csv"
            source.write_text("word,a,b,c\na,0,1,0\nb,0,0,1\nc,0,0,0\n", encoding="utf-8")
            output = root / "out"
            result = subprocess.run(
                [sys.executable, str(Path(textrank.__file__).resolve()), str(source),
                 "--outdir", str(output), "--top", "2"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "graph_textrank.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["word"], "c")
            self.assertGreater((output / "graph_textrank.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
