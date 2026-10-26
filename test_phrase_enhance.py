"""Regression checks for phrase matching and exact matrix reconstruction."""
import csv
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import cooccurrence as co
import phrase_enhance as phrase


class PhraseEnhanceTests(unittest.TestCase):
    def test_raw_boundaries_deduplication_and_stopwords(self):
        segments = phrase.raw_segments(
            "Machine and learning.\n\nMachine and learning.\n\nMachine, learning."
        )
        self.assertEqual(segments, [["machine", "and", "learning"], ["machine"], ["learning"]])
        candidates, selected = phrase.mine_phrases(segments, 1, -1, 3)
        self.assertEqual(candidates, [])
        self.assertEqual(selected, [])

    def test_npmi_uses_directed_event_marginals(self):
        candidates, _ = phrase.mine_phrases(
            [["alpha", "beta"], ["alpha", "beta"], ["alpha", "gamma"], ["delta", "beta"]],
            1, -1, 3,
        )
        item = next(item for item in candidates if item.label == "alpha beta")
        self.assertEqual(item.count, 2)
        self.assertAlmostEqual(item.npmi, math.log(8 / 9) / math.log(2))

    def test_overlap_consumes_each_token_once(self):
        selected = [phrase.Candidate("alpha", "beta", 1, 1),
                    phrase.Candidate("beta", "gamma", 1, 1)]
        merged, replacements = phrase.merge_phrases([["alpha", "beta", "gamma"]], selected, 3)
        self.assertEqual(merged, [["alpha beta", "gamma"]])
        self.assertEqual(replacements["alpha beta"], 1)
        self.assertEqual(replacements["beta gamma"], 0)

    def test_exact_context_and_segment_counts_after_merge(self):
        selected = [phrase.Candidate("alpha", "beta", 3, 1)]
        merged, replacements = phrase.merge_phrases(
            [["left", "alpha", "beta", "right", "alpha", "beta", "end"],
             ["alpha", "alone"]], selected, 3,
        )
        terms, directed, undirected, frequencies = co.build_matrices(merged, 1, 0)
        at = {word: index for index, word in enumerate(terms)}
        self.assertNotIn("beta", at)
        self.assertIn("alpha", at)
        for left, right in [("left", "alpha beta"), ("alpha beta", "right"),
                            ("right", "alpha beta"), ("alpha beta", "end"), ("alpha", "alone")]:
            self.assertEqual(directed[at[left], at[right]], 1)
        self.assertEqual(int(directed.sum()), 5)
        self.assertEqual(undirected[at["alpha beta"], at["right"]], 2)
        self.assertEqual(replacements["alpha beta"], 2)
        self.assertEqual(frequencies["alpha beta"], 1)

    def test_no_selection_matches_original_pipeline(self):
        raw = "Alpha and beta gamma.\n\nDelta beta."
        merged, _ = phrase.merge_phrases(phrase.raw_segments(raw), [], 3)
        original = co.tokenize_segments(co.unique_paragraphs(co.markdown_to_text(raw)), 3)
        self.assertEqual(merged, original)

    def test_phrase_to_phrase_edges(self):
        selected = [phrase.Candidate("alpha", "beta", 1, 1),
                    phrase.Candidate("gamma", "delta", 1, 1)]
        merged, _ = phrase.merge_phrases([["alpha", "beta", "gamma", "delta"]], selected, 3)
        terms, directed, _, _ = co.build_matrices(merged, 1, 0)
        self.assertEqual(directed[terms.index("alpha beta"), terms.index("gamma delta")], 1)

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(Path(phrase.__file__).resolve()), *map(str, args)],
                              capture_output=True, text=True)

    def test_cli_markdown_outputs_six_integer_matrices_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, outdir = root / "input.md", root / "out"
            raw = "Left alpha beta right.\n\nStart alpha beta end."
            source.write_text(raw, encoding="utf-8")
            result = self.run_cli(source, "--outdir", outdir, "--min-frequency", 1,
                                  "--top-words", 0, "--min-phrase-frequency", 2, "--min-npmi", -1,
                                  "--plot-top-words", 2, "--figure-size", 4)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_text(encoding="utf-8"), raw)
            self.assertEqual(len(list(outdir.glob("phrase_enhanced_*.csv"))), 6)
            images = list(outdir.glob("phrase_enhanced_*_heatmap.png"))
            self.assertEqual(len(images), 6)
            for image in images:
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with (outdir / "phrase_enhanced_directed_diag_count.csv").open(encoding="utf-8-sig") as file:
                rows = list(csv.reader(file))
            words = rows[0][1:]
            counts = np.array([[int(value) for value in row[1:]] for row in rows[1:]])
            self.assertEqual([row[0] for row in rows[1:]], words)
            self.assertEqual(counts[words.index("alpha beta"), words.index("alpha beta")], 2)
            with (outdir / "phrase_candidates.csv").open(encoding="utf-8-sig") as file:
                report = list(csv.DictReader(file))
            self.assertEqual(report[0]["phrase"], "alpha beta")
            self.assertEqual(report[0]["merged_count"], "2")
            self.assertEqual(report[0]["included_in_matrix"], "True")

    def test_cli_no_candidates_still_writes_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.md"
            source.write_text("Alpha beta.", encoding="utf-8")
            result = self.run_cli(source, "--outdir", root / "out", "--min-frequency", 1, "--no-plots")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("No phrases passed", result.stdout)
            self.assertEqual(list((root / "out").glob("*.png")), [])

    def test_cli_invalid_input_and_threshold(self):
        for args in [("matrix.csv",), ("missing.md",), ("--min-npmi", "nan"),
                     ("--min-phrase-frequency", "0"), ("--top-words", "-1"),
                     ("--plot-top-words", "-1"), ("--figure-size", "0"),
                     ("--figure-size", "nan")]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
