"""Tests for the three-method ranking comparison."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

import compare_rankings as comparison


class CompareRankingsTests(unittest.TestCase):
    def test_three_methods_return_full_rankings(self):
        labels = ["a", "b", "c", "d"]
        graph = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=float)
        methods, beta = comparison.compare(labels, graph)
        self.assertAlmostEqual(beta, 0.425)
        self.assertEqual(set(methods), {"linear_4hop", "katz_4hop", "textrank"})
        for values in methods.values():
            self.assertEqual(sorted(values["rank"].tolist()), [1, 2, 3, 4])
            self.assertTrue(np.isfinite(values["score"]).all())

    def test_hop_parameter_changes_boolean_linear_method(self):
        labels = ["a", "b", "c"]
        graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        methods, _ = comparison.compare(labels, graph, hop=2)
        self.assertIn("linear_2hop", methods)
        np.testing.assert_array_equal(methods["linear_2hop"]["score"], [3, 4, 3])
        with self.assertRaisesRegex(ValueError, "hop must be >= 1"):
            comparison.compare(labels, graph, hop=0)

    def test_matrix_validation_rejects_non_boolean_and_diagonal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("word,a,b\na,0,2\nb,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                comparison.read_boolean_matrix(path)
            path.write_text("word,a,b\na,1,0\nb,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "diagonal"):
                comparison.read_boolean_matrix(path)

    def test_run_writes_all_words_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "graph.csv", root / "out"
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["word", "a", "b", "c"])
                writer.writerow(["a", 0, 1, 0])
                writer.writerow(["b", 1, 0, 1])
                writer.writerow(["c", 0, 1, 0])
            comparison.run(source, output, top=2)
            self.assertTrue((output / "graph_comparison.png").is_file())
            with (output / "graph_comparison.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertIn("linear_4hop_rank", rows[0])
            self.assertIn("katz_4hop_rank", rows[0])
            self.assertIn("textrank_rank", rows[0])


if __name__ == "__main__":
    unittest.main()
