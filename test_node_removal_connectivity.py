"""Tests for per-node connectivity impact analysis."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import node_removal_connectivity as impact


class NodeRemovalConnectivityTests(unittest.TestCase):
    def test_undirected_path_middle_node_splits_component(self):
        words = ["a", "b", "c"]
        graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        rows = {row["removed_node"]: row
                for row in impact.analyze_removals(words, graph)}
        self.assertEqual(rows["b"]["components_before"], 1)
        self.assertEqual(rows["b"]["components_after"], 2)
        self.assertEqual(rows["b"]["component_delta"], 1)
        self.assertEqual(rows["b"]["connected_pairs_lost"], 1)
        self.assertEqual(rows["a"]["connected_pairs_lost"], 0)

    def test_directed_weak_and_strong_connectivity_differ(self):
        words = ["a", "b", "c"]
        graph = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
        weak = {row["removed_node"]: row
                for row in impact.analyze_removals(words, graph, "weak")}
        strong = {row["removed_node"]: row
                  for row in impact.analyze_removals(words, graph, "strong")}
        self.assertEqual(weak["b"]["components_before"], 1)
        self.assertEqual(weak["b"]["components_after"], 2)
        self.assertEqual(weak["b"]["connected_pairs_lost"], 1)
        self.assertEqual(strong["b"]["components_before"], 3)
        self.assertEqual(strong["b"]["components_after"], 2)
        self.assertEqual(strong["b"]["connected_pairs_lost"], 0)

    def test_cli_writes_csv_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, outdir = root / "path.csv", root / "out"
            source.write_text(
                "vertex,a,b,c\na,0,1,0\nb,1,0,1\nc,0,1,0\n",
                encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(Path(impact.__file__).resolve()), str(source),
                 "--outdir", str(outdir), "--top", "2"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((outdir / "path_node_removal_connectivity.png").is_file())
            with (outdir / "path_node_removal_connectivity.csv").open(
                    encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["removed_node"], "b")


if __name__ == "__main__":
    unittest.main()
