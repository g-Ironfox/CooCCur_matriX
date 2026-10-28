"""Tests for iterative Katz weighted-path scoring."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import katz_stability as katz


class KatzTests(unittest.TestCase):
    def test_iteration_matches_closed_form_for_two_cycle(self):
        adjacency = np.array([[0, 1], [1, 0]], dtype=float)
        beta = 0.1
        final = None
        for _, partial, converged in katz.iter_katz(adjacency, beta, 1000):
            final = partial.copy()
            if converged:
                break
        expected = np.array([[beta**2, beta], [beta, beta**2]]) / (1 - beta**2)
        np.testing.assert_allclose(final, expected, rtol=1e-7, atol=1e-10)

    def test_weighted_adjacency_is_preserved_and_nilpotent_graph_converges(self):
        adjacency = np.array([[0, 2], [0, 0]], dtype=float)
        values = list(katz.iter_katz(adjacency, 0.2, 10))
        self.assertEqual(values[-1][0], 2)
        self.assertTrue(values[-1][2])
        np.testing.assert_allclose(values[-1][1], [[0, 0.4], [0, 0]])

    def test_rejects_negative_beta_and_respects_iteration_cap(self):
        with self.assertRaisesRegex(ValueError, "beta"):
            list(katz.iter_katz(np.zeros((2, 2)), -0.1, 3))
        iterator = katz.iter_katz(np.array([[0, 1], [1, 0]], dtype=float), 0.8, 1)
        self.assertFalse(list(iterator)[-1][2])

    def test_cli_writes_weighted_and_stats_outputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "graph.csv", root / "out"
            source.write_text("vertex,a,b\na,0,2\nb,0,0\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(katz.__file__).resolve()), str(source),
                 "--beta", "0.2", "--outdir", str(output), "--top", "2",
                 "--sort-by", "min"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Converged at iteration: 2", result.stdout)
            self.assertTrue((output / "katz_002.csv").is_file())
            self.assertTrue((output / "katz_002.png").is_file())
            self.assertTrue((output / "katz_002_word_stats.csv").is_file())
            self.assertTrue((output / "katz_002_word_stats.png").is_file())
            self.assertTrue((output / "katz_001_word_stats.png").stat().st_size > 0)
            self.assertFalse(list(output.glob("reachability_*")))
            with (output / "katz_001.csv").open(encoding="utf-8-sig") as file:
                matrix = list(csv.reader(file))
            self.assertEqual(matrix[1], ["a", "0", "0.4"])
            with (output / "katz_001_word_stats.csv").open(encoding="utf-8-sig") as file:
                stats = list(csv.DictReader(file))
            self.assertEqual(stats[0]["word"], "b")
            self.assertEqual(stats[0]["incoming_score"], "0.4")

    def test_run_rejects_beta_at_convergence_boundary_before_writing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "graph.csv", root / "out"
            source.write_text(
                "vertex,a,b,c\na,0,1,1\nb,1,0,1\nc,1,1,0\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not converge"):
                katz.run(source, output, beta=0.5)
            self.assertFalse(output.exists())

    def test_reader_rejects_negative_edge_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative.csv"
            path.write_text("vertex,a,b\na,0,-1\nb,0,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-negative"):
                katz.read_matrix(path)


if __name__ == "__main__":
    unittest.main()
