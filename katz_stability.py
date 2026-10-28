#!/usr/bin/env python3
"""Compute Katz scores by iteratively summing weighted adjacency powers."""
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
    if any(len(row) != len(labels) + 1 for row in rows[1:]):
        raise ValueError("matrix rows have inconsistent column counts")
    if [row[0] for row in rows[1:]] != labels:
        raise ValueError("row labels must match column labels in the same order")
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if (matrix < 0).any():
        raise ValueError("Katz adjacency weights must be non-negative")
    if np.diag(matrix).any():
        raise ValueError("self-loops are not supported; diagonal entries must be zero")
    return labels, matrix


def spectral_radius(adjacency: np.ndarray, max_iterations: int = 500,
                    tolerance: float = 1e-8) -> float:
    """Estimate the Perron root of a non-negative matrix by power iteration."""
    matrix = np.asarray(adjacency, dtype=float)
    vector = np.ones(matrix.shape[0], dtype=float)
    estimates = []
    for _ in range(max_iterations):
        following = matrix @ vector
        scale = float(np.max(following, initial=0.0))
        if scale == 0:
            return 0.0
        estimates.append(scale)
        vector = following / scale
        if len(estimates) >= 8:
            recent = estimates[-8:]
            average = sum(recent) / len(recent)
            if max(recent) - min(recent) <= tolerance * max(average, 1.0):
                return max(recent)
    return max(estimates[-8:], default=0.0)


def iter_katz(adjacency: np.ndarray, beta: float, max_iterations: int,
              atol: float = 1e-10, rtol: float = 1e-8):
    """Yield (k, partial_sum, converged) for K_k = sum_(i=1..k) beta^i A^i."""
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency matrix must be square")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("adjacency weights must be finite and non-negative")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and >= 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("tolerances must be finite and >= 0")
    if atol == 0 and rtol == 0:
        raise ValueError("at least one tolerance must be greater than zero")

    term = beta * matrix
    total = np.zeros_like(matrix)
    for hop in range(1, max_iterations + 1):
        total += term
        if not np.isfinite(total).all():
            raise ValueError("Katz iteration overflowed; reduce beta or check the input")
        converged = np.max(np.abs(term), initial=0.0) <= (
            atol + rtol * np.max(np.abs(total), initial=0.0)
        )
        yield hop, total, converged
        if converged:
            return
        term = beta * (matrix @ term)
        if not np.isfinite(term).all():
            raise ValueError("Katz iteration overflowed; reduce beta or check the input")


def write_matrix(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["vertex", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *(format(float(value), ".12g") for value in row)])


def plot_matrix(path: Path, labels: list[str], matrix: np.ndarray,
                hop: int, cell_size: float) -> None:
    size = max(5, min(18, len(labels) * cell_size + 4))
    label_size = max(3, min(8, cell_size * 18))
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues", vmin=0,
                      vmax=max(float(np.max(matrix, initial=0.0)), 1e-15),
                      interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Katz score")
    stride = max(1, (len(labels) + 59) // 60)
    indices = list(range(0, len(labels), stride))
    ax.set_xticks(indices, labels=[str(i) for i in indices], rotation=70, ha="right", fontsize=label_size)
    ax.set_yticks(indices, labels=[str(i) for i in indices], fontsize=label_size)
    ax.set_title(f"Katz weighted matrix (iteration {hop})")
    ax.set_xlabel("Destination vertex (0-based index)")
    ax.set_ylabel("Source vertex (0-based index)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(matrix: np.ndarray, labels: list[str], include_diagonal: bool,
              sort_by: str) -> list[dict]:
    if sort_by not in ("incoming", "outgoing", "min"):
        raise ValueError("sort_by must be incoming, outgoing or min")
    scores = np.asarray(matrix, dtype=float).copy()
    self_scores = np.diag(scores).copy()
    if not include_diagonal:
        np.fill_diagonal(scores, 0)
    incoming, outgoing = scores.sum(axis=0), scores.sum(axis=1)
    minimum = np.minimum(incoming, outgoing)
    primary = {"incoming": incoming, "outgoing": outgoing, "min": minimum}[sort_by]
    order = sorted(range(len(labels)), key=lambda i: (-primary[i], -incoming[i], -outgoing[i], i))
    return [{"rank": rank, "word": labels[i], "incoming_score": incoming[i],
             "outgoing_score": outgoing[i], "min_score": minimum[i],
             "self_score": self_scores[i]} for rank, i in enumerate(order, 1)]


def write_word_stats(path: Path, labels: list[str], matrix: np.ndarray, top: int,
                     include_diagonal: bool, sort_by: str) -> None:
    rows = summarize(matrix, labels, include_diagonal, sort_by)
    fields = ["rank", "word", "incoming_score", "outgoing_score", "min_score", "self_score"]
    with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: (format(value, ".12g") if isinstance(value, (float, np.floating)) else value)
                          for key, value in row.items()} for row in rows)
    shown = rows[:top] if top else rows
    if not shown:
        return
    positions = np.arange(len(shown))
    charts = [
        ("incoming_score", "Incoming Katz score", "#2878b5"),
        ("outgoing_score", "Outgoing Katz score", "#e07a2d"),
    ]
    if sort_by == "min":
        charts.insert(0, ("min_score", "Minimum Katz score", "#4a8f5b"))
    fig, axes = plt.subplots(1, len(charts),
                             figsize=(5 * len(charts), max(4, len(shown) * 0.25)),
                             sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    for ax, (field, title, color) in zip(axes, charts):
        values = [row[field] for row in shown]
        bars = ax.barh(positions, values, color=color)
        ax.bar_label(bars, padding=3, fontsize=7, fmt="%.3g")
        ax.set_xlim(0, max(values, default=0) * 1.22 or 1)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, labels=[row["word"] for row in shown], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Word")
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def run(input_path: Path, outdir: Path, beta: float, max_iterations: int = 1000,
        atol: float = 1e-10, rtol: float = 1e-8, cell_size: float = 0.35,
        top: int = 30, sort_by: str = "incoming", include_diagonal: bool = False) -> int:
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell size must be greater than zero")
    if top < 0:
        raise ValueError("--top must be >= 0")
    labels, adjacency = read_matrix(input_path)
    if not labels:
        raise ValueError("input graph must contain at least one vertex")
    radius = spectral_radius(adjacency)
    if beta * radius >= 1:
        limit = (1 / radius) if radius else float("inf")
        raise ValueError(
            f"Katz series does not converge: beta * spectral_radius = {beta * radius:.6g} "
            f"(spectral_radius={radius:.6g}); choose beta < {limit:.6g}"
        )
    print(f"Spectral radius: {radius:.6g}; beta * radius: {beta * radius:.6g}")
    outdir.mkdir(parents=True, exist_ok=True)
    final_hop, converged = 0, False
    for hop, total, converged in iter_katz(adjacency, beta, max_iterations, atol, rtol):
        final_hop = hop
        write_matrix(outdir / f"katz_{hop:03d}.csv", labels, total)
        plot_matrix(outdir / f"katz_{hop:03d}.png", labels, total, hop, cell_size)
        write_word_stats(outdir / f"katz_{hop:03d}_word_stats.csv", labels,
                         total, top, include_diagonal, sort_by)
    if not converged:
        raise ValueError(f"Katz iteration did not converge within {max_iterations} steps; "
                         "reduce --beta or increase --max-iterations")
    print(f"Vertices: {len(labels)}")
    print(f"Beta: {beta}")
    print(f"Converged at iteration: {final_hop}")
    print(f"Output directory: {outdir}")
    return final_hop


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Katz scores by iterative path summation.")
    parser.add_argument("input", help="Labeled square adjacency-matrix CSV; row = source, column = destination")
    parser.add_argument("--outdir", default="output/katz_stability",
                        help="Directory for weighted matrices, heatmaps and word statistics")
    parser.add_argument("--beta", type=float, required=True,
                        help="Attenuation factor; choose beta below the reciprocal spectral radius")
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--cell-size", type=float, default=0.35)
    parser.add_argument("--top", type=int, default=30, help="Words per statistics plot; 0 means all")
    parser.add_argument("--sort-by", choices=["incoming", "outgoing", "min"], default="incoming")
    parser.add_argument("--include-diagonal", action="store_true")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.beta, args.max_iterations,
            args.atol, args.rtol, args.cell_size, args.top, args.sort_by,
            args.include_diagonal)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
