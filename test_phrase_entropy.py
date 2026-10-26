"""Tests for exact directed phrase boundary entropy from Markdown."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
                        "source_joint_entropy_heatmap.png"}
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            for name in ("left", "right", "left_right", "joint"):
                image = output / f"source_{name}_entropy_heatmap.png"
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            for side in ("left", "right", "joint"):
                matrix = output / f"source_{side}_entropy.csv"
                self.assertTrue(matrix.read_text(encoding="utf-8-sig").startswith("word,"))
            self.assertNotIn("undirected", result.stdout.lower())

    def test_bad_candidate_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            path.write_text("phrase\nalpha beta\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "left and right"):
                pe.read_candidates(path)


if __name__ == "__main__":
    unittest.main()
