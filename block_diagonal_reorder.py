#!/usr/bin/env python3
"""Reorder a labeled adjacency matrix to expose block/community structure."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("CSV must contain a header and at least one vertex")
    labels = rows[0][1:]
    if len(rows) != len(labels) + 1:
        raise ValueError("matrix must have one data row per vertex")
    row_labels, values = [], []
    for row in rows[1:]:
        if len(row) != len(labels) + 1:
            raise ValueError("matrix rows have inconsistent column counts")
        row_labels.append(row[0])
        try:
            values.append([float(value) for value in row[1:]])
        except ValueError as error:
            raise ValueError("matrix values must be numeric") from error
    if row_labels != labels:
        raise ValueError("row labels must match column labels in the same order")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("vertex labels must be unique and non-empty")
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("matrix weights must be finite and non-negative")
    return labels, matrix


def reorder_matrix(matrix: np.ndarray, method: str = "leiden", resolution: float = 1.0,
                   seed: int = 42, iterations: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Detect weighted communities and return their contiguous vertex ordering."""
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] == 0 or weights.shape[0] != weights.shape[1]:
        raise ValueError("matrix must be a non-empty square array")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("matrix weights must be finite and non-negative")
    if method not in ("leiden", "louvain"):
        raise ValueError("method must be 'leiden' or 'louvain'")
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be finite and greater than zero")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if not isinstance(iterations, (int, np.integer)) or (iterations == 0 or iterations < -1):
        raise ValueError("iterations must be -1 (until stable) or a positive integer")

    try:
        import igraph as ig
    except ImportError as error:
        raise RuntimeError(
            "community detection requires python-igraph; install it with "
            "'.venv/Scripts/python.exe -m pip install -r requirements-community.txt'"
        ) from error

    # Community detection is undirected here: for directed input, either edge
    # direction contributes affinity. Self-loops do not define communities.
    affinity = np.maximum(weights, weights.T)
    np.fill_diagonal(affinity, 0.0)
    edges = [(int(i), int(j)) for i, j in zip(*np.triu(affinity > 0, k=1).nonzero())]
    n = weights.shape[0]
    if not edges:
        membership = list(range(n))
        edge_weights: list[float] = []
    else:
        edge_weights = [float(affinity[i, j]) for i, j in edges]
        graph = ig.Graph(n=n, edges=edges, directed=False)
        old_rng = ig.set_random_number_generator(random.Random(int(seed)))
        try:
            if method == "leiden":
                partition = graph.community_leiden(
                    objective_function="modularity", weights=edge_weights,
                    resolution=float(resolution), n_iterations=int(iterations))
            else:
                partition = graph.community_multilevel(
                    weights=edge_weights, resolution=float(resolution))
            membership = partition.membership
        finally:
            ig.set_random_number_generator(old_rng)

    groups: dict[int, list[int]] = {}
    for vertex, community in enumerate(membership):
        groups.setdefault(int(community), []).append(vertex)
    # Put denser/larger blocks first, then order vertices by within-community strength.
    def community_key(item: tuple[int, list[int]]) -> tuple[float, int]:
        _, members = item
        total = float(affinity[np.ix_(members, members)].sum()) / 2
        return (-total, min(members))

    sorted_groups = sorted(groups.items(), key=community_key)
    order: list[int] = []
    community_ids = np.zeros(n, dtype=int)
    for new_community_id, (_, members) in enumerate(sorted_groups, 1):
        member_set = np.asarray(members, dtype=int)
        within_strength = affinity[np.ix_(member_set, member_set)].sum(axis=1)
        sorted_members = member_set[np.argsort(-within_strength, kind="stable")]
        order.extend(int(vertex) for vertex in sorted_members)
        community_ids[member_set] = new_community_id
    return np.asarray(order, dtype=int), community_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect graph communities and reorder a labeled adjacency matrix.")
    parser.add_argument("input", nargs="?", default="output/undirected_diag_zero.csv",
                        help="Labeled adjacency-matrix CSV")
    parser.add_argument("--method", choices=("leiden", "louvain"), default="leiden",
                        help="Community detection algorithm (default: leiden)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Community granularity; higher values usually yield smaller communities (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--iterations", type=int, default=-1,
                        help="Leiden iterations; -1 runs until stable (default: -1; ignored by Louvain)")
    parser.add_argument("--outdir", default="output/block_diagonal",
                        help="Output directory (default: output/block_diagonal)")
    parser.add_argument("--prefix", help="Output filename prefix (default: input filename)")
    args = parser.parse_args()
    if args.prefix and (Path(args.prefix).name != args.prefix or "/" in args.prefix or "\\" in args.prefix
                        or args.prefix in (".", "..")):
        parser.error("--prefix must be a filename prefix, not a path")
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"input matrix not found: {source}")
    try:
        labels, matrix = read_matrix(source)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    n = len(labels)
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("--resolution must be finite and greater than zero")
    if args.iterations == 0 or args.iterations < -1:
        parser.error("--iterations must be -1 or a positive integer")
    try:
        order, cluster_ids = reorder_matrix(
            matrix, method=args.method, resolution=args.resolution,
            seed=args.seed, iterations=args.iterations)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    clusters = int(cluster_ids.max())
    ordered_labels = [labels[i] for i in order]
    reordered = matrix[np.ix_(order, order)]
    ordered_clusters = cluster_ids[order]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or source.stem
    matrix_path = outdir / f"{prefix}_reordered.csv"
    order_path = outdir / f"{prefix}_order.csv"
    plot_path = outdir / f"{prefix}_block_diagonal.png"
    with matrix_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["vertex", *ordered_labels])
        writer.writerows([[label, *row] for label, row in zip(ordered_labels, reordered)])
    with order_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["position", "original_position", "vertex", "cluster"])
        writer.writerows((position, int(index) + 1, labels[index], int(cluster_ids[index]))
                         for position, index in enumerate(order, 1))

    fig_size = max(7, min(18, 4 + n * 0.16))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    image = ax.imshow(reordered, interpolation="nearest", aspect="equal", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Edge weight")
    boundaries = np.flatnonzero(ordered_clusters[1:] != ordered_clusters[:-1]) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=1.2)
        ax.axvline(boundary, color="white", linewidth=1.2)
    tick_step = max(1, int(np.ceil(n / 40)))
    ticks = np.arange(0, n, tick_step)
    ax.set_xticks(ticks, [ordered_labels[i] for i in ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks, [ordered_labels[i] for i in ticks], fontsize=7)
    ax.set_title(f"{args.method.title()} communities ({clusters} found; resolution={args.resolution:g})")
    ax.set_xlabel("Target vertex")
    ax.set_ylabel("Source vertex")
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(f"Vertices: {n}")
    print(f"Method: {args.method}; resolution: {args.resolution:g}; seed: {args.seed}")
    print(f"Communities found: {clusters}")
    print(f"Reordered matrix: {matrix_path}")
    print(f"Vertex order: {order_path}")
    print(f"Heatmap: {plot_path}")


if __name__ == "__main__":
    main()
