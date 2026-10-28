#!/usr/bin/env python3
"""Measure and plot connectivity changes caused by removing each vertex."""
from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


FIELDS = [
    "rank", "removed_node", "components_before", "components_after",
    "component_delta", "largest_component_before", "largest_component_after",
    "largest_component_delta", "connected_pairs_lost", "component_sizes_after",
]


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
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if (matrix < 0).any():
        raise ValueError("matrix weights must be non-negative")
    return labels, matrix > 0


def component_sizes(adjacency: np.ndarray, mode: str) -> list[int]:
    if adjacency.shape[0] == 0:
        return []
    graph = csr_matrix(adjacency)
    directed = mode == "strong"
    _, labels = connected_components(
        graph, directed=directed, connection=mode, return_labels=True)
    sizes = np.bincount(labels).tolist()
    return sorted(sizes, reverse=True)


def analyze_removals(words: list[str], adjacency: np.ndarray,
                     mode: str = "weak") -> list[dict]:
    if mode not in ("weak", "strong"):
        raise ValueError("mode must be weak or strong")
    graph = np.asarray(adjacency, dtype=bool).copy()
    if graph.shape != (len(words), len(words)):
        raise ValueError("adjacency dimensions must match the vertex labels")
    np.fill_diagonal(graph, False)

    directed = mode == "strong"
    before_count, original_labels = connected_components(
        csr_matrix(graph), directed=directed, connection=mode,
        return_labels=True)
    before_sizes = sorted(np.bincount(original_labels).tolist(), reverse=True)
    before_largest = max(before_sizes, default=0)
    results = []
    for removed, word in enumerate(words):
        keep = np.arange(len(words)) != removed
        remaining = graph[np.ix_(keep, keep)]
        after_sizes = component_sizes(remaining, mode)
        original_pair_groups = np.bincount(original_labels[keep])
        before_pair_count = sum(comb(int(size), 2) for size in original_pair_groups)
        after_pair_count = sum(comb(size, 2) for size in after_sizes)
        largest_after = max(after_sizes, default=0)
        results.append({
            "removed_node": word,
            "components_before": before_count,
            "components_after": len(after_sizes),
            "component_delta": len(after_sizes) - before_count,
            "largest_component_before": before_largest,
            "largest_component_after": largest_after,
            "largest_component_delta": largest_after - before_largest,
            "connected_pairs_lost": before_pair_count - after_pair_count,
            "component_sizes_after": ";".join(map(str, after_sizes)),
        })

    results.sort(key=lambda row: (
        -row["connected_pairs_lost"],
        -row["component_delta"],
        -row["largest_component_before"] + row["largest_component_after"],
        row["removed_node"],
    ))
    for rank, row in enumerate(results, 1):
        row["rank"] = rank
    return results


def plot_impact(path: Path, rows: list[dict], mode: str) -> None:
    height = max(5, min(24, 2.2 + len(rows) * 0.32))
    fig, axes = plt.subplots(1, 2, figsize=(14, height), sharey=True,
                             constrained_layout=True)
    positions = np.arange(len(rows))
    components_after = [row["components_after"] for row in rows]
    pairs_lost = [row["connected_pairs_lost"] for row in rows]

    axes[0].barh(positions, components_after, color="#2F6F73")
    axes[0].set_xlabel("Connected components after removal")
    axes[0].set_title("Component count")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(positions, pairs_lost, color="#C77763")
    axes[1].set_xlabel("Previously connected pairs disconnected")
    axes[1].set_title("Connectivity loss")
    axes[1].grid(axis="x", alpha=0.25)

    axes[0].set_yticks(positions, [row["removed_node"] for row in rows], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"Connectivity impact by node removal ({mode} connectivity)")
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def run(input_path: Path, outdir: Path, top: int = 30,
        mode: str = "weak") -> list[dict]:
    if top < 0:
        raise ValueError("--top must be >= 0")
    words, adjacency = read_matrix(input_path)
    rows = analyze_removals(words, adjacency, mode)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_node_removal_connectivity.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    shown = rows[:top] if top else rows
    image_path = outdir / f"{prefix}_node_removal_connectivity.png"
    plot_impact(image_path, shown, mode)
    print(f"Plot: {image_path}")
    print(f"Vertices: {len(words)}; connectivity mode: {mode}")
    print(f"Results: {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Labeled square adjacency-matrix CSV")
    parser.add_argument("--outdir", default="output/node_removal_connectivity",
                        help="Output directory")
    parser.add_argument("--top", type=int, default=30,
                        help="Nodes shown in the plot; 0 shows all (default: 30)")
    parser.add_argument("--mode", choices=("weak", "strong"), default="weak",
                        help="Connectivity for directed graphs (default: weak)")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.top, args.mode)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
