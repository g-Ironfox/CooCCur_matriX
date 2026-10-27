#!/usr/bin/env python3
"""Plot Boolean reachability and changing-hop weighted sums at every step."""
from __future__ import annotations

import argparse
import csv
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
        values.append([float(value) for value in row[1:]])
    if row_labels != labels:
        raise ValueError("row labels must match column labels in the same order")
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("vertex labels must be unique and non-empty")

    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    adjacency = matrix > 0
    if np.diag(adjacency).any():
        raise ValueError("self-loops are not supported; diagonal entries must be zero")
    return labels, adjacency


def stabilization_hops(adjacency: np.ndarray) -> int:
    """Find the first k with R_(k+1) == R_k, respecting edge directions.

    Use Boolean multiplication in R_(k+1) = R_k OR (R_k @ B).
    In particular, a directed cycle can add diagonal reachability after
    all distinct reachable vertex pairs have already been discovered.
    """
    base = np.asarray(adjacency, dtype=bool)
    if base.shape[0] == 0:
        return 0
    reachability = base.copy()
    hop = 1
    while True:
        following = reachability | (reachability @ base)
        if np.array_equal(following, reachability):
            return hop
        reachability = following
        hop += 1


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["vertex", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *map(int, row.tolist())])


def iter_weighted_hops(adjacency: np.ndarray, max_hops: int):
    """Yield (k, R_k, S_k); yielded arrays are updated by the next step.

    Powers use Boolean multiplication. S_k uses ordinary integer addition:
    C_k = C_(k-1) + B^k and S_k = S_(k-1) + C_k.
    Thus every earlier power gains one unit of weight at every new hop.
    """
    if max_hops < 1:
        return
    dtype = np.min_scalar_type(max_hops * (max_hops + 1) // 2)
    base = np.asarray(adjacency, dtype=bool)
    power = base.copy()
    reachability = power.copy()
    cumulative = np.zeros(power.shape, dtype=dtype)
    weighted = np.zeros(power.shape, dtype=dtype)
    for hop in range(1, max_hops + 1):
        if hop > 1:
            power = power @ base
            reachability |= power
        cumulative += power
        weighted += cumulative
        yield hop, reachability, weighted


def plot_reachability(path: Path, labels: list[str], matrix: np.ndarray,
                      hop: int, stable: bool, cell_size: float,
                      weighted: bool = False) -> None:
    size = max(5, min(18, len(labels) * cell_size + 4))
    label_size = max(3, min(8, cell_size * 18))
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    vmax = max(1, int(matrix.max())) if weighted else 1
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
    if weighted:
        fig.colorbar(image, ax=ax, label="Weighted Boolean score (darker = larger)")
    stride = max(1, (len(labels) + 59) // 60)
    indices = list(range(0, len(labels), stride))
    tick_labels = [str(index) for index in indices]
    ax.set_xticks(indices, labels=tick_labels, rotation=70, ha="right", fontsize=label_size)
    ax.set_yticks(indices, labels=tick_labels, fontsize=label_size)
    if weighted:
        suffix = " (reachability stable)" if stable else ""
        ax.set_title(f"Weighted Boolean powers S_{hop}{suffix}")
    else:
        suffix = " (stable)" if stable else ""
        ax.set_title(f"Boolean reachability R_{hop}{suffix}")
    ax.set_xlabel("Destination vertex (0-based index)")
    ax.set_ylabel("Source vertex (0-based index)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(input_path: Path, outdir: Path, cell_size: float = 0.35) -> int:
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell size must be greater than zero")
    labels, adjacency = read_matrix(input_path)
    if not labels:
        raise ValueError("input graph must contain at least one vertex")

    stable_hop = stabilization_hops(adjacency)
    outdir.mkdir(parents=True, exist_ok=True)
    for hop, reachability, weighted in iter_weighted_hops(adjacency, stable_hop):
        stable = hop == stable_hop
        write_matrix(outdir / f"reachability_{hop:03d}.csv", labels, reachability)
        plot_reachability(outdir / f"reachability_{hop:03d}.png", labels,
                          reachability, hop, stable, cell_size)
        write_matrix(outdir / f"weighted_{hop:03d}.csv", labels, weighted)
        plot_reachability(outdir / f"weighted_{hop:03d}.png", labels,
                          weighted, hop, stable, cell_size, weighted=True)

    print(f"Vertices: {len(labels)}")
    print(f"Stabilization hop: {stable_hop}")
    print(f"Steps visualized: {stable_hop}")
    print(f"Output directory: {outdir}")
    return stable_hop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find when directed or undirected Boolean n-hop reachability stabilizes."
    )
    parser.add_argument("input", help="Labeled square adjacency-matrix CSV; row = source, column = destination")
    parser.add_argument("--outdir", default="output/bool_hop_stability",
                        help="Directory for per-hop matrices and heatmaps")
    parser.add_argument("--cell-size", type=float, default=0.35,
                        help="Approximate heatmap cell size in inches (default: 0.35)")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.cell_size)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
