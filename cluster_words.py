#!/usr/bin/env python3
"""Cluster words from a co-occurrence adjacency-matrix CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import cut_tree, dendrogram, linkage
from scipy.spatial.distance import squareform

CLUSTER_COLORS = (
    "#1F5A75",
    "#4C8C9A",
    "#6FAE9B",
    "#C59A52",
    "#C77763",
    "#7B6F8F",
)


def read_matrix(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("CSV must contain a header and at least one word row")
    words = rows[0][1:]
    if len(rows) != len(words) + 1:
        raise ValueError("matrix must have one data row per word")
    row_words, values = [], []
    for row in rows[1:]:
        if len(row) != len(words) + 1:
            raise ValueError("matrix rows have inconsistent column counts")
        row_words.append(row[0])
        values.append([float(value) for value in row[1:]])
    if row_words != words:
        raise ValueError("row labels must match column labels in the same order")
    if len(set(words)) != len(words) or any(not word.strip() for word in words):
        raise ValueError("word labels must be unique and non-empty")
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if (matrix < 0).any():
        raise ValueError("co-occurrence weights must be non-negative")
    return words, matrix


def cosine_distances(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1)
    normalized = np.divide(features, norms[:, None], out=np.zeros_like(features), where=norms[:, None] > 0)
    similarity = normalized @ normalized.T
    distances = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchically cluster words by adjacency-pattern similarity.")
    parser.add_argument("input", nargs="?", default="output/undirected_diag_zero.csv",
                        help="Adjacency matrix CSV (default: output/undirected_diag_zero.csv)")
    parser.add_argument("--clusters", type=int, default=6, help="Number of clusters (default: 6)")
    parser.add_argument("--direction", choices=("auto", "directed", "undirected"), default="auto",
                        help="Feature interpretation; auto detects matrix asymmetry (default: auto)")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--prefix", help="Output filename prefix (default: input filename)")
    parser.add_argument("--figure-size", type=float, help="Figure width in inches (default: based on word count)")
    parser.add_argument("--keep-diagonal", action="store_true", help="Include diagonal counts in clustering features (default: ignore)")
    args = parser.parse_args()

    if args.figure_size is not None and (not np.isfinite(args.figure_size) or args.figure_size <= 0):
        parser.error("--figure-size must be finite and greater than 0")
    if args.prefix and (Path(args.prefix).name != args.prefix or "/" in args.prefix or "\\" in args.prefix):
        parser.error("--prefix must be a filename prefix, not a path")
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input matrix not found: {input_path}")
    try:
        words, matrix = read_matrix(input_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    n = len(words)
    if not 1 <= args.clusters <= n:
        parser.error(f"--clusters must be between 1 and number of words ({n})")

    if not args.keep_diagonal:
        np.fill_diagonal(matrix, 0.0)
    is_directed = not np.allclose(matrix, matrix.T)
    directed = is_directed if args.direction == "auto" else args.direction == "directed"
    features = np.concatenate((matrix, matrix.T), axis=1) if directed else (matrix + matrix.T) / 2
    isolated = np.linalg.norm(features, axis=1) == 0
    if isolated.any():
        print("Unassigned (cluster 0, no connections): " + ", ".join(word for word, flag in zip(words, isolated) if flag))
    active = ~isolated
    active_words = [word for word, flag in zip(words, active) if flag]
    active_count = len(active_words)
    if active_count and args.clusters > active_count:
        parser.error(f"--clusters cannot exceed connected-word count ({active_count})")
    labels = np.zeros(n, dtype=int)
    if active_count > 1:
        distances = cosine_distances(features[active])
        tree = linkage(squareform(distances, checks=False), method="average")
        labels[active] = cut_tree(tree, n_clusters=[args.clusters]).ravel() + 1
    else:
        tree = None
        labels[active] = 1
    cluster_count = len(set(labels.tolist()) - {0})

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    csv_path = outdir / f"{prefix}_clusters.csv"
    plot_path = outdir / f"{prefix}_dendrogram.png"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "cluster"])
        writer.writerows(sorted(zip(words, labels.tolist()), key=lambda item: (item[1], item[0])))

    fig_width = args.figure_size or max(10, min(24, n * 0.32))
    fig, ax = plt.subplots(figsize=(fig_width, 8), constrained_layout=True)
    if tree is not None:
        leaf_clusters = labels[active]
        node_clusters = {index: int(cluster) for index, cluster in enumerate(leaf_clusters)}
        for merge_index, (left, right, _, _) in enumerate(tree):
            left_cluster = node_clusters[int(left)]
            right_cluster = node_clusters[int(right)]
            node_clusters[active_count + merge_index] = (
                left_cluster if left_cluster == right_cluster else 0
            )

        def branch_color(node_id: int) -> str:
            cluster = node_clusters[node_id]
            if cluster == 0:
                return "#7A858B"
            return CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)]

        dendrogram(tree, labels=active_words, leaf_rotation=90, leaf_font_size=8, ax=ax,
                   color_threshold=0, above_threshold_color="#7A858B",
                   link_color_func=branch_color)
        for line in ax.get_lines():
            line.set_linewidth(1.8)
        ax.set_ylabel("Cosine distance (average linkage)")
    else:
        ax.text(0.5, 0.5, active_words[0] if active_words else "No connected words to cluster", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    label_by_word = dict(zip(words, labels.tolist()))
    for tick in ax.get_xticklabels():
        cluster = label_by_word[tick.get_text()]
        tick.set_color("#7A858B" if cluster == 0 else CLUSTER_COLORS[(cluster - 1) % len(CLUSTER_COLORS)])
        tick.set_fontweight("bold")
    if isolated.any():
        fig.suptitle(f"{int(isolated.sum())} isolated words excluded; see cluster 0 in CSV", fontsize=10)
    ax.set_title(f"Word Clusters ({args.clusters} requested, {cluster_count} produced; {'directed' if directed else 'undirected'} features)")
    ax.set_xlabel("Word")
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)

    print(f"Words: {n}")
    print(f"Features: {'outgoing + incoming' if directed else 'undirected neighbors'}")
    print(f"Clusters produced: {cluster_count}")
    print(f"Cluster assignments: {csv_path}")
    print(f"Dendrogram: {plot_path}")


if __name__ == "__main__":
    main()
