"""Tests for exact directed phrase boundary entropy from Markdown."""
import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import phrase_entropy as pe


class PhraseEntropyTests(unittest.TestCase):
    def test_asymmetric_left_right_entropy_and_reverse_pair(self):
        phrases = pe.collect_contexts(
            "first alpha beta apple. second alpha beta berry. third alpha beta berry. "
            "fourth beta alpha peach. fifth beta alpha plum. fifth beta alpha plum.", min_length=3)
        ab, ba = phrases[("alpha", "beta")], phrases[("beta", "alpha")]
        self.assertEqual(ab.count, 3)
        self.assertAlmostEqual(pe.entropy(ab.left), math.log(3))
        self.assertAlmostEqual(pe.entropy(ab.right), -((1 / 3) * math.log(1 / 3) + (2 / 3) * math.log(2 / 3)))
        self.assertAlmostEqual(pe.entropy(ab.joint), math.log(3))
        self.assertAlmostEqual(pe.entropy(ba.left), -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)))
        self.assertAlmostEqual(pe.entropy(ba.right), -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)))

    def test_stopwords_punctuation_and_paragraph_boundaries_are_not_bridged(self):
        phrases = pe.collect_contexts(
            "outside alpha and beta tail.\n\n"
            "left alpha beta right.\n\nleft alpha beta right.", min_length=3)
        self.assertEqual(phrases[("alpha", "beta")].count, 1)
        self.assertEqual(sum(phrases[("alpha", "beta")].left.values()), 1)
        self.assertEqual(len(phrases[("alpha", "beta")].paragraphs), 1)

    def test_missing_context_is_nan_and_deterministic_context_is_zero(self):
        phrases = pe.collect_contexts("alpha beta. left gamma delta right.", min_length=3)
        words, counts, left, right, joint = pe.build_matrices(phrases)
        ab = (words.index("alpha"), words.index("beta"))
        gd = (words.index("gamma"), words.index("delta"))
        self.assertTrue(math.isnan(left[ab]))
        self.assertTrue(math.isnan(right[ab]))
        self.assertEqual(left[gd], 0)
        self.assertEqual(right[gd], 0)
        self.assertEqual(joint[ab], 0)
        self.assertEqual(joint[gd], 0)
        self.assertEqual(counts[ab], 1)

    def test_base_plot_selection_keeps_empty_axes_and_nonempty_selection_prunes_them(self):
        counts = np.zeros((6, 6), dtype=int)
        counts[0, 1], counts[0, 2] = 10, 10
        counts[3, 4], counts[3, 5] = 8, 7
        indices = pe.select_plot_indices(counts, 4)
        self.assertEqual(set(indices), {0, 1, 2, 3})
        nonempty = pe.select_nonempty_indices(counts, indices)
        self.assertEqual(set(nonempty), {0, 1, 2})
        self.assertEqual(len(pe.select_plot_indices(counts, 5)), 5)

    def test_nonempty_selection_keeps_single_neighbor_nodes_and_removes_isolated_nodes(self):
        counts = np.zeros((4, 4), dtype=int)
        counts[0, 1], counts[1, 2] = 2, 1
        original = counts.copy()
        indices = pe.select_plot_indices(counts, 0)
        self.assertEqual(list(indices), [0, 1, 2, 3])
        self.assertEqual(list(pe.select_nonempty_indices(counts, indices)), [0, 1, 2])
        np.testing.assert_array_equal(counts, original)

    def test_nonempty_selection_does_not_introduce_words_outside_selected_axes(self):
        counts = np.zeros((4, 4), dtype=int)
        counts[0, 1], counts[2, 3] = 9, 8
        indices = np.array([1, 2])
        self.assertEqual(pe.select_nonempty_indices(counts, indices).size, 0)

    def test_joint_entropy_includes_segment_boundary_markers(self):
        phrase = pe.collect_contexts("alpha beta. left alpha beta right.")[ ("alpha", "beta")]
        self.assertEqual(sum(phrase.joint.values()), 2)
        self.assertEqual(phrase.joint[("<BOS>", "<EOS>")], 1)
        self.assertEqual(phrase.joint[("left", "right")], 1)
        self.assertAlmostEqual(pe.entropy(phrase.joint), math.log(2))

    def test_candidate_filter_and_minimum_frequency(self):
        candidates = {("alpha", "beta")}
        phrases = pe.collect_contexts("x alpha beta y. x gamma delta y.", 3, candidates)
        self.assertEqual(set(phrases), candidates)
        self.assertEqual(pe.collect_contexts("alpha beta.", 3)[("alpha", "beta")].count, 1)

    def test_cli_writes_combined_and_joint_heatmaps_with_square_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("left alpha beta right.\n\nstart alpha beta finish.\n\n"
                              "first gamma delta last.\n\nsecond gamma delta end.", encoding="utf-8")
            result = subprocess.run([sys.executable, str(Path(pe.__file__).resolve()), str(source),
                                     "--outdir", str(output), "--min-phrase-frequency", "2",
                                     "--plot-top-words", "0", "--figure-size", "4"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {"source_phrase_boundaries.csv", "source_phrase_counts.csv",
                        "source_left_entropy.csv", "source_right_entropy.csv",
                        "source_joint_entropy.csv", "source_left_entropy_heatmap.png",
                        "source_right_entropy_heatmap.png", "source_left_right_entropy_heatmap.png",
                        "source_joint_entropy_heatmap.png",
                        "source_nonempty_left_right_entropy_heatmap.png",
                        "source_nonempty_joint_entropy_heatmap.png"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            for name in ("left", "right", "left_right", "joint",
                         "nonempty_left_right", "nonempty_joint"):
                image = output / f"source_{name}_entropy_heatmap.png"
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            for side in ("left", "right", "joint"):
                matrix = output / f"source_{side}_entropy.csv"
                self.assertTrue(matrix.read_text(encoding="utf-8-sig").startswith("word,"))
            self.assertNotIn("undirected", result.stdout.lower())

    def test_full_vocabulary_keeps_singletons_and_words_without_phrase_edges(self):
        markdown = "alpha beta. lonely. gamma and delta. alpha alpha."
        phrases = pe.collect_contexts(markdown)
        vocabulary = pe.collect_vocabulary(markdown)
        self.assertEqual(vocabulary, {"alpha", "beta", "lonely", "gamma", "delta"})
        words, counts, left, right, joint = pe.build_matrices(phrases, vocabulary)
        self.assertEqual(words, sorted(vocabulary))
        self.assertEqual(counts.sum(), 1)
        for word in ("lonely", "gamma", "delta"):
            index = words.index(word)
            self.assertFalse(counts[index].any())
            self.assertFalse(counts[:, index].any())
            for matrix in (left, right, joint):
                self.assertTrue(np.isnan(matrix[index]).all())
                self.assertTrue(np.isnan(matrix[:, index]).all())
        indices = pe.select_nonempty_indices(counts, pe.select_plot_indices(counts, 0))
        self.assertEqual([words[i] for i in indices], ["alpha", "beta"])

    def test_six_plots_use_four_full_and_two_pruned_views_without_changing_values(self):
        markdown = "start alpha beta finish. lonely. gamma and delta."
        words, counts, left, right, joint = pe.build_matrices(
            pe.collect_contexts(markdown), pe.collect_vocabulary(markdown))
        snapshots = [matrix.copy() for matrix in (counts, left, right, joint)]
        indices = pe.select_nonempty_indices(counts, np.arange(len(words)))
        subset = np.ix_(indices, indices)
        with patch.object(pe, "plot_entropy") as single, \
                patch.object(pe, "plot_left_right_heatmap") as combined:
            paths = pe.plot_heatmaps(Path("unused"), "sample", words, counts,
                                     left, right, joint, 0)
        self.assertEqual(len(paths), 6)
        self.assertEqual(single.call_count, 4)
        self.assertEqual(combined.call_count, 2)
        for call, matrix in zip(single.call_args_list[:3], (left, right, joint)):
            self.assertEqual(call.args[1], words)
            np.testing.assert_equal(call.args[2], matrix)
        self.assertEqual(combined.call_args_list[0].args[1], words)
        np.testing.assert_equal(combined.call_args_list[0].args[2], left)
        np.testing.assert_equal(combined.call_args_list[0].args[3], right)
        reduced_words = [words[i] for i in indices]
        self.assertNotIn("lonely", reduced_words)
        self.assertNotIn("gamma", reduced_words)
        self.assertNotIn("delta", reduced_words)
        self.assertEqual(single.call_args_list[3].args[1], reduced_words)
        np.testing.assert_equal(single.call_args_list[3].args[2], joint[subset])
        self.assertEqual(combined.call_args_list[1].args[1], reduced_words)
        np.testing.assert_equal(combined.call_args_list[1].args[2], left[subset])
        np.testing.assert_equal(combined.call_args_list[1].args[3], right[subset])
        for matrix, original in zip((counts, left, right, joint), snapshots):
            np.testing.assert_equal(matrix, original)

    def test_cli_defaults_keep_all_vocabulary_and_single_occurrences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("alpha beta. lonely. gamma and delta.", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(pe.__file__).resolve()), str(source),
                 "--outdir", str(output), "--figure-size", "3"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Original heatmap axes: 5; nonempty heatmap axes: 2", result.stdout)
            self.assertEqual(len(list(output.glob("*.png"))), 6)
            with (output / "source_phrase_counts.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows[0][1:], ["alpha", "beta", "delta", "gamma", "lonely"])
            self.assertEqual([r[0] for r in rows[1:]], rows[0][1:])
            self.assertEqual(sum(int(v) for row in rows[1:] for v in row[1:]), 1)
            with (output / "source_phrase_boundaries.csv").open(encoding="utf-8-sig", newline="") as f:
                report = list(csv.DictReader(f))
            self.assertEqual([(row["phrase"], row["count"]) for row in report], [("alpha beta", "1")])

    def test_cli_without_phrase_edges_still_produces_six_plots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.md", root / "out"
            source.write_text("lonely. alpha and beta.", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(pe.__file__).resolve()), str(source),
                 "--outdir", str(output), "--figure-size", "3"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Original heatmap axes: 3; nonempty heatmap axes: 0", result.stdout)
            self.assertEqual(len(list(output.glob("*.png"))), 6)

    def test_frequency_zero_preserves_axes_and_positive_thresholds_prune_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.md"
            source.write_text("alpha beta. alpha beta. alpha beta. gamma delta. lonely.",
                              encoding="utf-8")
            for threshold, expected_words, expected_phrases, nonempty_count in (
                    (None, ["alpha", "beta", "delta", "gamma", "lonely"], 2, 4),
                    (0, ["alpha", "beta", "delta", "gamma", "lonely"], 2, 4),
                    (1, ["alpha", "beta", "delta", "gamma"], 2, 4),
                    (2, ["alpha", "beta"], 1, 2),
                    (3, ["alpha", "beta"], 1, 2),
                    (9, [], 0, 0)):
                with self.subTest(threshold=threshold):
                    output = root / f"out_{threshold}"
                    argv = ["phrase_entropy.py", str(source), "--outdir", str(output)]
                    if threshold is not None:
                        argv.extend(["--min-phrase-frequency", str(threshold)])
                    with patch.object(sys, "argv", argv), \
                            patch.object(pe, "plot_heatmaps", return_value={}) as plots, \
                            redirect_stdout(io.StringIO()) as stdout:
                        pe.main()
                    for name in ("phrase_counts", "left_entropy", "right_entropy", "joint_entropy"):
                        with (output / f"source_{name}.csv").open(
                                encoding="utf-8-sig", newline="") as f:
                            rows = list(csv.reader(f))
                        self.assertEqual(rows[0], ["word", *expected_words])
                        self.assertEqual([row[0] for row in rows[1:]], expected_words)
                        self.assertTrue(all(len(row) == len(expected_words) + 1 for row in rows))
                    with (output / "source_phrase_boundaries.csv").open(
                            encoding="utf-8-sig", newline="") as f:
                        report = list(csv.DictReader(f))
                    self.assertEqual(len(report), expected_phrases)
                    self.assertTrue(all(int(row["count"]) >= (threshold or 0) for row in report))
                    args = plots.call_args.args
                    self.assertEqual(args[2], expected_words)
                    self.assertEqual(args[3].shape, (len(expected_words), len(expected_words)))
                    self.assertEqual(args[7], 0)
                    self.assertIn(f"Original heatmap axes: {len(expected_words)}; "
                                  f"nonempty heatmap axes: {nonempty_count}", stdout.getvalue())
                    if expected_phrases:
                        words, counts, left, right, joint = pe.build_matrices(
                            {pair: stats for pair, stats in pe.collect_contexts(
                                source.read_text(encoding="utf-8")).items()
                             if stats.count >= (threshold or 0)},
                            pe.collect_vocabulary(source.read_text(encoding="utf-8"))
                            if not threshold else None)
                        for actual, expected in zip(args[3:7], (counts, left, right, joint)):
                            np.testing.assert_equal(actual, expected)

    def test_negative_frequency_is_rejected_before_output(self):
        with patch.object(sys, "argv", ["phrase_entropy.py", "--min-phrase-frequency", "-1"]), \
                redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                pe.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--min-phrase-frequency must be >= 0", stderr.getvalue())

    def test_bad_candidate_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            path.write_text("phrase\nalpha beta\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "left and right"):
                pe.read_candidates(path)


if __name__ == "__main__":
    unittest.main()
