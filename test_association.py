"""Regression tests for PPMI/NPMI matrix analysis."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import association


class AssociationTests(unittest.TestCase):
    def test_scores_exclude_diagonal_by_default(self):
        counts = np.array([[9, 2, 0], [1, 8, 1], [1, 0, 7]], dtype=float)
        original = counts.copy()
        ppmi, npmi = association.compute_scores(counts)
        np.testing.assert_array_equal(counts, original)
        self.assertEqual(ppmi.shape, counts.shape)
        self.assertEqual(ppmi[0, 0], 0)
        self.assertEqual(npmi[0, 0], 0)
        self.assertTrue(np.isfinite(ppmi).all())
        self.assertTrue(np.isfinite(npmi).all())
        self.assertTrue((ppmi >= 0).all())
        self.assertTrue(((npmi >= -1) & (npmi <= 1)).all())

    def test_single_observed_event_npmi_is_one(self):
        ppmi, npmi = association.compute_scores(np.array([[0, 4], [0, 0]]))
        self.assertEqual(ppmi[0, 1], 0)
        self.assertEqual(npmi[0, 1], 1)

    def test_cli_writes_association_outputs_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "counts.csv", root / "out"
            association.write_matrix(source, ["alpha", "beta"], np.array([[0, 2], [1, 0]]))
            result = subprocess.run(
                [sys.executable, str(Path(association.__file__).resolve()), str(source),
                 "--outdir", str(output), "--plot-top-words", "0", "--figure-size", "4"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual({path.name for path in output.iterdir()}, {
                "counts_ppmi.csv", "counts_npmi.csv", "counts_ppmi_heatmap.png",
                "counts_npmi_heatmap.png", "counts_ppmi_npmi_comparison.png"})
            self.assertIn("PPMI matrix:", result.stdout)

    def test_invalid_matrix_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("word,alpha,beta\nbeta,1,0\nalpha,0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row labels"):
                association.read_matrix(path)


if __name__ == "__main__":
    unittest.main()
