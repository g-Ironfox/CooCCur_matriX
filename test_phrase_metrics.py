"""Tests for the aligned four-metric phrase visualization."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import association
import phrase_metrics as metrics


class PhraseMetricsTests(unittest.TestCase):
    def test_metrics_share_axes_and_match_existing_definitions(self):
        markdown = ("outside alpha beta apple. outside alpha beta berry. "
                    "outside alpha beta berry. first gamma delta last. lonely.")
        words, counts, left, right, ppmi, npmi, joint, phrases = metrics.compute_metrics(markdown)
        self.assertEqual(words, sorted({"outside", "alpha", "beta", "apple", "berry",
                                        "first", "gamma", "delta", "last", "lonely"}))
        for matrix in (counts, left, right, ppmi, npmi, joint):
            self.assertEqual(matrix.shape, (len(words), len(words)))
        expected_ppmi, expected_npmi = association.compute_scores(counts)
        np.testing.assert_allclose(ppmi, expected_ppmi)
        np.testing.assert_allclose(npmi, expected_npmi)
        self.assertAlmostEqual(left[words.index("alpha"), words.index("beta")], 0)
        self.assertAlmostEqual(right[words.index("alpha"), words.index("beta")],
                               -(1 / 3 * np.log(1 / 3) + 2 / 3 * np.log(2 / 3)))
        self.assertTrue(np.isfinite(joint[words.index("alpha"), words.index("beta")]))
        self.assertIn(("alpha", "beta"), phrases)
        self.assertTrue(np.isnan(left[words.index("lonely"), words.index("alpha")]))

    def test_plot_frequency_threshold_only_selects_visual_axes(self):
        markdown = "alpha beta. alpha beta. alpha beta. gamma delta. lonely."
        words, counts, *_ = metrics.compute_metrics(markdown)
        indices = metrics.select_indices(counts, 0, min_phrase_frequency=2)
        self.assertEqual([words[index] for index in indices], ["alpha", "beta"])
        self.assertEqual(counts.sum(), 4)
        words, counts, *_ = metrics.compute_metrics(markdown, candidates={("beta", "alpha")})
        self.assertEqual(words, sorted({"alpha", "beta", "delta", "gamma", "lonely"}))
        self.assertEqual(counts.sum(), 0)

    def test_plot_selection_is_only_a_view(self):
        counts = np.array([[0, 5, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(metrics.select_indices(counts, 1), [0])
        np.testing.assert_array_equal(metrics.select_indices(counts, 0), [0, 1, 2])

    def test_left_and_right_entropy_share_their_absolute_nats_scale(self):
        left = np.array([[0.2, np.nan], [1.1, 0.0]])
        right = np.array([[2.4, 0.5], [np.nan, 0.0]])
        self.assertEqual(metrics.shared_entropy_max(left, right), 2.4)

    def test_cli_writes_aligned_matrices_and_four_quadrant_png(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("outside alpha beta apple. outside alpha beta berry. "
                              "outside alpha beta berry. lonely.", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(metrics.__file__).resolve()), str(source),
                 "--outdir", str(output), "--plot-top-words", "0",
                 "--plot-min-phrase-frequency", "2", "--figure-size", "4"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {"source_phrase_counts.csv", "source_left_entropy.csv",
                        "source_right_entropy.csv", "source_joint_entropy.csv",
                        "source_left_minus_right_entropy.csv", "source_ppmi.csv", "source_npmi.csv",
                        "source_phrase_boundaries.csv", "source_four_metrics_heatmap.png",
                        "source_left_right_entropy_heatmap.png", "source_joint_entropy_heatmap.png",
                        "source_left_minus_right_entropy_heatmap.png", "source_ppmi_heatmap.png",
                        "source_npmi_heatmap.png"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            with (output / "source_left_entropy.csv").open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[0][0], "word")
            self.assertEqual(len(rows), len(rows[0]))
            matrix_names = ("phrase_counts", "left_entropy", "right_entropy", "joint_entropy",
                            "left_minus_right_entropy", "ppmi", "npmi")
            for name in matrix_names:
                with (output / f"source_{name}.csv").open(
                        encoding="utf-8-sig", newline="") as file:
                    matrix_rows = list(csv.reader(file))
                self.assertIn("lonely", matrix_rows[0])
                self.assertEqual(len(matrix_rows), len(matrix_rows[0]))
                if name == "phrase_counts":
                    self.assertEqual(sum(int(value) for row in matrix_rows[1:]
                                         for value in row[1:] if value), 9)
            self.assertEqual((output / "source_four_metrics_heatmap.png").read_bytes()[:8],
                             b"\x89PNG\r\n\x1a\n")
            for name in ("left_right_entropy", "joint_entropy", "left_minus_right_entropy",
                         "ppmi", "npmi"):
                self.assertEqual((output / f"source_{name}_heatmap.png").read_bytes()[:8],
                                 b"\x89PNG\r\n\x1a\n")

    def test_no_pair_events_writes_blank_association_scores_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("lonely. alpha and beta.", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(metrics.__file__).resolve()), str(source),
                 "--outdir", str(output), "--figure-size", "3"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("ppmi", "npmi"):
                with (output / f"source_{name}.csv").open(encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.reader(file))
                self.assertTrue(all(cell == "" for row in rows[1:] for cell in row[1:]))
            self.assertTrue((output / "source_four_metrics_heatmap.png").is_file())
            for name in ("left_right_entropy", "joint_entropy", "left_minus_right_entropy",
                         "ppmi", "npmi"):
                self.assertTrue((output / f"source_{name}_heatmap.png").is_file())

    def test_empty_vocabulary_still_writes_all_heatmaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(metrics.__file__).resolve()), str(source),
                 "--outdir", str(output), "--figure-size", "3"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("four_metrics", "left_right_entropy", "joint_entropy",
                         "left_minus_right_entropy", "ppmi", "npmi"):
                self.assertTrue((output / f"source_{name}_heatmap.png").is_file())


if __name__ == "__main__":
    unittest.main()
