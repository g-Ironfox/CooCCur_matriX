#!/usr/bin/env python3
"""Project directed word profiles (outgoing + incoming) to two dimensions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("CSV must contain a header and at least one word row")
    words = rows[0][1:]
    if len(rows) != len(words) + 1:
        raise ValueError("matrix must have one data row per word")
    values = []
    for row in rows[1:]:
        if len(row) != len(words) + 1:
            raise ValueError("matrix rows have inconsistent column counts")
        values.append([float(value) for value in row[1:]])
    if [row[0] for row in rows[1:]] != words:
        raise ValueError("row labels must match column labels in the same order")
    if len(set(words)) != len(words) or any(not word.strip() for word in words):
        raise ValueError("word labels must be unique and non-empty")
    matrix = np.asarray(values, dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("weights must be finite and non-negative")
    return words, matrix


def project_profiles(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA coordinates and explained-variance ratios of the first two PCs."""
    # Keep source and target roles distinct while compressing large weights.
    features = np.concatenate((np.sqrt(matrix), np.sqrt(matrix.T)), axis=1)
    centered = features - features.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    coordinates = left[:, :2] * singular_values[:2]
    variance = singular_values ** 2
    ratios = variance[:2] / variance.sum() if variance.sum() else np.zeros(2)
    return coordinates, ratios


def plot_embedding(path: Path, words: list[str], matrix: np.ndarray,
                   coordinates: np.ndarray, label_top: int) -> None:
    strength = matrix.sum(axis=0) + matrix.sum(axis=1)
    positive = strength[strength > 0]
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    if positive.size:
        lower = max(float(positive.min()), np.finfo(float).tiny)
        points = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=np.maximum(strength, lower),
                            norm=LogNorm(vmin=lower, vmax=max(float(positive.max()), lower * 1.001)),
                            cmap="viridis", s=30, alpha=0.82,
                            edgecolors="white", linewidths=0.35)
        fig.colorbar(points, ax=ax, label="Total incoming + outgoing edge weight (log scale)")
    else:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], color="#357C78", s=30, alpha=0.82)

    labeled = np.argsort(-strength, kind="stable")[:min(label_top, len(words))]
    for i in labeled:
        ax.annotate(words[i], coordinates[i], xytext=(4, 4), textcoords="offset points",
                    fontsize=7)
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    ax.set_title(f"Directed word profiles | PCA of row + column features\n"
                 f"{len(words)} words; max directional difference = {asymmetry:.4g}")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.22)
    ax.set_axisbelow(True)
    try:
        fig.savefig(path, dpi=180)
    finally:
        plt.close(fig)


def run(input_path: Path, outdir: Path, label_top: int = 15) -> tuple[Path, Path]:
    if label_top < 0:
        raise ValueError("--label-top must be >= 0")
    words, matrix = read_matrix(input_path)
    coordinates, explained = project_profiles(matrix)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_pca_coordinates.csv"
    plot_path = outdir / f"{prefix}_pca.png"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word", "pc1", "pc2", "incoming_weight", "outgoing_weight"])
        for word, point, incoming, outgoing in zip(
                words, coordinates, matrix.sum(axis=0), matrix.sum(axis=1)):
            writer.writerow([word, f"{point[0]:.10g}", f"{point[1]:.10g}",
                             f"{incoming:.10g}", f"{outgoing:.10g}"])
    plot_embedding(plot_path, words, matrix, coordinates, label_top)
    print(f"Words: {len(words)}; features per word: {matrix.shape[1] * 2}")
    print(f"Explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")
    print(f"Coordinates: {csv_path}")
    print(f"Plot: {plot_path}")
    return csv_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Directed weighted square matrix CSV")
    parser.add_argument("--outdir", default="output/adjacency_embedding_viz")
    parser.add_argument("--label-top", type=int, default=15,
                        help="Number of highest-total-weight words to label (default: 15)")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.label_top)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
