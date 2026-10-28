#!/usr/bin/env python3
"""Rank words in a labeled weighted adjacency matrix with TextRank."""
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
    if len(set(labels)) != len(labels) or any(not label.strip() for label in labels):
        raise ValueError("vertex labels must be unique and non-empty")
    values = []
    for i, row in enumerate(rows[1:]):
        if len(row) != len(labels) + 1:
            raise ValueError("matrix rows have inconsistent column counts")
        if row[0] != labels[i]:
            raise ValueError("row labels must match column labels in the same order")
        try:
            values.append([float(value) for value in row[1:]])
        except ValueError as error:
            raise ValueError(f"matrix contains a non-numeric value in row {i + 1}") from error
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if (matrix < 0).any():
        raise ValueError("edge weights must be non-negative")
    if np.diag(matrix).any():
        raise ValueError("self-loops are not supported; diagonal entries must be zero")
    return labels, matrix


def textrank(matrix: np.ndarray, damping: float = 0.85,
             tolerance: float = 1e-10, max_iterations: int = 1000) -> tuple[np.ndarray, int]:
    """Return weighted PageRank scores; rows are sources, columns destinations."""
    weights = np.asarray(matrix, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1] or weights.shape[0] == 0:
        raise ValueError("matrix must be a non-empty square array")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("matrix weights must be finite and non-negative")
    if not 0 <= damping < 1:
        raise ValueError("damping must be in [0, 1)")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    size = len(weights)
    row_sums = weights.sum(axis=1)
    transition = np.divide(weights, row_sums[:, None], out=np.zeros_like(weights),
                           where=row_sums[:, None] != 0)
    dangling = row_sums == 0
    scores = np.full(size, 1.0 / size)
    teleport = (1.0 - damping) / size
    for iteration in range(1, max_iterations + 1):
        following = teleport + damping * (
            scores @ transition + scores[dangling].sum() / size
        )
        if np.abs(following - scores).sum() <= tolerance:
            return following / following.sum(), iteration
        scores = following
    return scores / scores.sum(), max_iterations


def write_results(input_path: Path, outdir: Path, top: int = 30,
                  damping: float = 0.85, tolerance: float = 1e-10,
                  max_iterations: int = 1000) -> list[dict]:
    if top < 0:
        raise ValueError("top must be >= 0")
    labels, matrix = read_matrix(input_path)
    scores, iterations = textrank(matrix, damping, tolerance, max_iterations)
    rows = [{"rank": 0, "word": word, "textrank_score": float(scores[i])}
            for i, word in enumerate(labels)]
    rows.sort(key=lambda row: (-row["textrank_score"], row["word"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_textrank.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["rank", "word", "textrank_score"])
        writer.writeheader()
        writer.writerows(rows)

    shown = rows[:top] if top else rows
    fig, ax = plt.subplots(figsize=(10, max(4, len(shown) * 0.32 + 1.5)),
                           constrained_layout=True)
    positions = np.arange(len(shown))
    values = [row["textrank_score"] for row in shown]
    bars = ax.barh(positions, values, color="#2F6F73")
    ax.bar_label(bars, padding=3, fontsize=8, fmt="%.4f")
    ax.set_yticks(positions, labels=[row["word"] for row in shown])
    ax.invert_yaxis()
    ax.set_xlabel("TextRank score")
    ax.set_title(f"TextRank (d={damping:g}, iterations={iterations})")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(outdir / f"{prefix}_textrank.png", dpi=160)
    plt.close(fig)
    print(f"Vertices: {len(labels)}; iterations: {iterations}")
    print(f"Scores: {csv_path}")
    print(f"Plot: {outdir / f'{prefix}_textrank.png'}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Labeled weighted adjacency-matrix CSV; row = source, column = destination")
    parser.add_argument("--outdir", default="output/textrank", help="Output directory")
    parser.add_argument("--top", type=int, default=30, help="Words to plot; 0 plots all")
    parser.add_argument("--damping", type=float, default=0.85, help="Damping factor (default: 0.85)")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="L1 convergence tolerance")
    parser.add_argument("--max-iterations", type=int, default=1000, help="Iteration limit")
    args = parser.parse_args()
    try:
        write_results(Path(args.input), Path(args.outdir), args.top,
                      args.damping, args.tolerance, args.max_iterations)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
