"""Tests for adjacency-matrix block ordering."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import block_diagonal_reorder as reorder


class BlockDiagonalReorderTests(unittest.TestCase):
    def test_leiden_groups_dense_blocks_and_preserves_matrix(self):
        matrix = np.array([
            [0, 8, 7, 0, 0, 0], [8, 0, 6, 0, 0, 0], [7, 6, 0, 0, 0, 0],
            [0, 0, 0, 0, 9, 8], [0, 0, 0, 9, 0, 7], [0, 0, 0, 8, 7, 0],
        ], dtype=float)
        shuffled = matrix[np.ix_([0, 3, 1, 4, 2, 5], [0, 3, 1, 4, 2, 5])]
        original = shuffled.copy()
        order, communities = reorder.reorder_matrix(shuffled, method="leiden", seed=12)
        arranged = shuffled[np.ix_(order, order)]
        arranged_communities = communities[order]
        self.assertEqual(len(set(arranged_communities)), 2)
        self.assertEqual(len(set(arranged_communities[:3])), 1)
        self.assertEqual(len(set(arranged_communities[3:])), 1)
        self.assertNotEqual(arranged_communities[0], arranged_communities[3])
        self.assertFalse(arranged[:3, 3:].any())
        self.assertFalse(arranged[3:, :3].any())
        self.assertTrue(np.array_equal(shuffled, original))

    def test_louvain_and_seed_are_supported(self):
        matrix = np.array([[0, 4, 3, 0], [4, 0, 3, 0],
                           [3, 3, 0, 0], [0, 0, 0, 0]], dtype=float)
        first = reorder.reorder_matrix(matrix, method="louvain", seed=5)
        second = reorder.reorder_matrix(matrix, method="louvain", seed=5)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))

    def test_zero_and_singleton_matrices(self):
        for n in (1, 4):
            order, labels = reorder.reorder_matrix(np.zeros((n, n)), method="leiden")
            self.assertEqual(sorted(order.tolist()), list(range(n)))
            self.assertEqual(len(set(labels)), n)

    def test_invalid_weights_shapes_and_parameters(self):
        for matrix in (np.array(1), np.zeros(3), np.zeros((2, 3)), np.zeros((0, 0)),
                       np.array([[np.nan]]), np.array([[-1.0]]), np.array([[np.inf]])):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                reorder.reorder_matrix(matrix)
        for kwargs in ({"method": "bad"}, {"resolution": 0}, {"iterations": 0},
                       {"iterations": -2}, {"seed": 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reorder.reorder_matrix(np.zeros((2, 2)), **kwargs)

    def test_cli_writes_reordered_matrix_order_and_heatmap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "graph.csv"
            source.write_text(
                "vertex,a,b,c,d\na,0,1,0,0\nb,1,0,0,0\nc,0,0,0,1\nd,0,0,1,0\n",
                encoding="utf-8")
            output = root / "out"
            result = subprocess.run(
                [sys.executable, str(Path(reorder.__file__).resolve()), str(source),
                 "--method", "leiden", "--outdir", str(output)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater((output / "graph_block_diagonal.png").stat().st_size, 0)
            with (output / "graph_order.csv").open(encoding="utf-8-sig", newline="") as file:
                order_rows = list(csv.DictReader(file))
            self.assertEqual(len(order_rows), 4)
            self.assertEqual(len({row["cluster"] for row in order_rows}), 2)
            self.assertEqual({row["vertex"] for row in order_rows}, {"a", "b", "c", "d"})
            with (output / "graph_reordered.csv").open(encoding="utf-8-sig", newline="") as file:
                matrix_rows = list(csv.reader(file))
            self.assertEqual(len(matrix_rows), 5)
            names, original = reorder.read_matrix(source)
            new_names, actual = reorder.read_matrix(output / "graph_reordered.csv")
            permutation = [names.index(name) for name in new_names]
            self.assertTrue(np.array_equal(actual, original[np.ix_(permutation, permutation)]))
            self.assertEqual([row["vertex"] for row in order_rows], new_names)
            self.assertEqual([int(row["original_position"]) for row in order_rows],
                             [index + 1 for index in permutation])


if __name__ == "__main__":
    unittest.main()
