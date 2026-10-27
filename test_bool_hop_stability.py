"""Tests for Boolean n-hop stabilization and visual output."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import matplotlib.image as mpimg
import numpy as np

import bool_hop_stability as hops


class BooleanHopTests(unittest.TestCase):
    def test_stabilization_matches_diameter_and_round_trip(self):
        path_graph = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=bool)
        self.assertEqual(hops.stabilization_hops(path_graph), 3)

        complete = np.ones((4, 4), dtype=bool)
        np.fill_diagonal(complete, False)
        self.assertEqual(hops.stabilization_hops(complete), 2)

    def test_diameter_when_first_vertex_is_in_the_middle(self):
        graph = np.zeros((5, 5), dtype=bool)
        for left, right in [(0, 1), (1, 2), (0, 3), (3, 4)]:
            graph[left, right] = graph[right, left] = True
        self.assertEqual(hops.stabilization_hops(graph), 4)

    def test_each_hop_updates_all_coefficients_with_boolean_powers(self):
        graph = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        expected = [
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[1, 2, 1], [2, 1, 2], [1, 2, 1]],
            [[2, 4, 2], [4, 2, 4], [2, 4, 2]],
        ]
        for hop, reachability, weighted in hops.iter_weighted_hops(graph, 3):
            np.testing.assert_array_equal(weighted, expected[hop - 1])
            np.testing.assert_array_equal(reachability, weighted > 0)

    def test_weighted_scores_do_not_wrap_at_uint8(self):
        graph = np.array([[0, 1], [1, 0]], dtype=bool)
        for hop, _, weighted in hops.iter_weighted_hops(graph, 33):
            if hop == 33:
                np.testing.assert_array_equal(weighted, [[272, 289], [289, 272]])

    def test_disconnected_and_edgeless_graphs(self):
        graph = np.zeros((5, 5), dtype=bool)
        graph[0, 1] = graph[1, 0] = True
        graph[1, 2] = graph[2, 1] = True
        self.assertEqual(hops.stabilization_hops(graph), 2)
        self.assertEqual(hops.stabilization_hops(np.zeros((3, 3), dtype=bool)), 1)

    def test_cli_outputs_each_step_matrix_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "graph.csv", root / "out"
            with source.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["vertex", "a", "b", "c"])
                writer.writerow(["a", 0, 1, 0])
                writer.writerow(["b", 1, 0, 1])
                writer.writerow(["c", 0, 1, 0])
            stable = hops.run(source, output)
            self.assertEqual(stable, 2)
            self.assertEqual(len(list(output.glob("reachability_*.csv"))), 2)
            self.assertEqual(len(list(output.glob("reachability_*.png"))), 2)
            self.assertEqual(len(list(output.glob("weighted_[0-9][0-9][0-9].csv"))), 2)
            self.assertEqual(len(list(output.glob("weighted_[0-9][0-9][0-9].png"))), 2)
            for hop in (1, 2):
                self.assertTrue((output / f"weighted_{hop:03d}_word_stats.csv").is_file())
                self.assertTrue((output / f"weighted_{hop:03d}_word_stats.png").is_file())
            with (output / "weighted_001_word_stats.csv").open(encoding="utf-8-sig") as file:
                first_stats = list(csv.DictReader(file))
            self.assertEqual(first_stats[0]["word"], "b")
            self.assertEqual(first_stats[0]["incoming_score"], "2")
            with (output / "weighted_002_word_stats.csv").open(encoding="utf-8-sig") as file:
                stats = list(csv.DictReader(file))
            self.assertEqual(stats[0]["word"], "b")
            self.assertEqual(stats[0]["incoming_score"], "4")
            self.assertEqual(stats[0]["outgoing_score"], "4")
            with (output / "weighted_002.csv").open(encoding="utf-8-sig") as file:
                scores = list(csv.reader(file))
            self.assertEqual(scores[0], ["vertex", "a", "b", "c"])
            self.assertEqual(scores[1][1:], ["1", "2", "1"])
            self.assertEqual(scores[2][1:], ["2", "1", "2"])
            with (output / "reachability_002.csv").open(encoding="utf-8-sig") as file:
                result = list(csv.reader(file))
            self.assertEqual(result[0], ["vertex", "a", "b", "c"])
            self.assertEqual(result[1][1:], ["1", "1", "1"])

    def test_cell_size_scales_plot_and_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "graph.csv"
            source.write_text(
                "vertex,a,b\na,0,1\nb,1,0\n", encoding="utf-8")
            small, large = root / "small", root / "large"
            hops.run(source, small, cell_size=0.2)
            hops.run(source, large, cell_size=1.5)
            small_shape = mpimg.imread(small / "reachability_002.png").shape
            large_shape = mpimg.imread(large / "reachability_002.png").shape
            self.assertGreater(large_shape[0], small_shape[0])
            with self.assertRaisesRegex(ValueError, "greater than zero"):
                hops.run(source, root / "invalid", cell_size=0)

    def test_directed_chain_keeps_edge_direction(self):
        graph = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ], dtype=bool)
        self.assertEqual(hops.stabilization_hops(graph), 3)
        for hop, reachable, weighted in hops.iter_weighted_hops(graph, 3):
            if hop == 3:
                np.testing.assert_array_equal(weighted, [
                    [0, 3, 2, 1], [0, 0, 3, 2],
                    [0, 0, 0, 3], [0, 0, 0, 0],
                ])
                self.assertTrue(reachable[0, 3])
                self.assertFalse(reachable[3, 0])
                self.assertFalse(reachable.diagonal().any())

    def test_directed_cycle_includes_positive_length_return_paths(self):
        graph = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
        # Distinct pairs are reachable at hop 2, but diagonals need hop 3.
        self.assertEqual(hops.stabilization_hops(graph), 3)
        for hop, reachable, weighted in hops.iter_weighted_hops(graph, 3):
            if hop == 3:
                self.assertTrue(reachable.all())
                np.testing.assert_array_equal(weighted, [
                    [1, 3, 2], [2, 1, 3], [3, 2, 1],
                ])

    def test_directed_disconnected_graph_and_single_edge(self):
        graph = np.zeros((6, 6), dtype=bool)
        for left, right in [(0, 1), (1, 2), (2, 0), (3, 4)]:
            graph[left, right] = True
        self.assertEqual(hops.stabilization_hops(graph), 3)
        self.assertEqual(hops.stabilization_hops(
            np.array([[0, 1], [0, 0]], dtype=bool)), 1)

    def test_cli_accepts_asymmetric_input_and_writes_weighted_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "directed.csv", root / "out"
            source.write_text(
                "vertex,a,b,c\na,0,1,0\nb,0,0,1\nc,0,0,0\n", encoding="utf-8")
            labels, adjacency = hops.read_matrix(source)
            self.assertEqual(labels, ["a", "b", "c"])
            self.assertTrue(adjacency[0, 1])
            self.assertFalse(adjacency[1, 0])
            result = subprocess.run(
                [sys.executable, str(Path(hops.__file__).resolve()), str(source),
                 "--outdir", str(output), "--top", "1", "--sort-by", "min",
                 "--include-diagonal"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Stabilization hop: 2", result.stdout)
            self.assertEqual(len(list(output.iterdir())), 12)
            with (output / "weighted_002_word_stats.csv").open(encoding="utf-8-sig") as file:
                stats = list(csv.DictReader(file))
            self.assertEqual(len(stats), 3)
            self.assertEqual(stats[0]["word"], "b")
            self.assertEqual(stats[0]["outgoing_score"], "2")
            self.assertEqual(stats[0]["incoming_score"], "2")
            self.assertEqual(stats[0]["min_score"], "2")
            with (output / "weighted_001_word_stats.csv").open(encoding="utf-8-sig") as file:
                first_stats = list(csv.DictReader(file))
            self.assertEqual(first_stats[0]["word"], "b")
            self.assertEqual(first_stats[0]["min_score"], "1")
            with (output / "weighted_002.csv").open(encoding="utf-8-sig") as file:
                scores = list(csv.reader(file))
            self.assertEqual(scores[1], ["a", "0", "2", "1"])
            self.assertEqual(scores[3], ["c", "0", "0", "0"])

    def test_self_loops_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "loop.csv"
            source.write_text("vertex,a\na,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "self-loops"):
                hops.read_matrix(source)


if __name__ == "__main__":
    unittest.main()
