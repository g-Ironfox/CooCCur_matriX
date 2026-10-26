#!/usr/bin/env python3
"""Compute PPMI and NPMI from a co-occurrence matrix CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


PPMI_CMAP = LinearSegmentedColormap.from_list(
    "ppmi", ["#F5F4F3", "#C99A8D", "#6F332B"]
)
NPMI_CMAP = "coolwarm"


def read_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.reader(file))
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("CSV must contain a header and at least one word")

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
        raise ValueError("co-occurrence counts must be non-negative")
    return words, matrix


def compute_scores(counts: np.ndarray, include_diagonal: bool = False):
    """Use row/column event marginals; unseen pairs have score zero in outputs."""
    counts = np.asarray(counts, dtype=float).copy()
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("co-occurrence matrix must be square")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must be finite and non-negative")
    if not include_diagonal:
        np.fill_diagonal(counts, 0.0)

    total = counts.sum()
    if total <= 0:
        raise ValueError("matrix has no off-diagonal co-occurrences to score")

    row_totals = counts.sum(axis=1)
    col_totals = counts.sum(axis=0)
    expected_denominator = row_totals[:, None] * col_totals[None, :]
    observed = counts > 0
    pmi = np.zeros_like(counts)
    pmi[observed] = np.log(
        counts[observed] * total / expected_denominator[observed]
    )
    ppmi = np.maximum(pmi, 0.0)

    joint_probability = counts[observed] / total
    npmi = np.zeros_like(counts)
    normalization = -np.log(joint_probability)
    npmi_values = np.ones_like(joint_probability)
    np.divide(pmi[observed], normalization, out=npmi_values, where=normalization > 0)
    npmi[observed] = np.clip(npmi_values, -1.0, 1.0)
    return ppmi, npmi


def write_matrix(path: Path, words: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["word", *words])
        for word, row in zip(words, matrix):
            writer.writerow([word, *[f"{value:.8g}" for value in row]])


def heatmap(path: Path, words: list[str], matrix: np.ndarray, title: str,
            cmap, vmin: float, vmax: float, colorbar_label: str,
            figure_size: float) -> None:
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.88), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(words)), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)), labels=words, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Word on right of pair")
    ax.set_ylabel("Word on left of pair")
    fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot PPMI/NPMI from a co-occurrence count matrix."
    )
    parser.add_argument("input", help="Count matrix CSV from cooccurrence.py")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--prefix", help="Output filename prefix (default: input filename)")
    parser.add_argument("--include-diagonal", action="store_true",
                        help="Include diagonal counts in probability estimates (default: exclude)")
    parser.add_argument("--plot-top-words", type=int, default=40,
                        help="Words shown in heatmaps by total co-occurrence; use 0 for all (default: 40)")
    parser.add_argument("--figure-size", type=float,
                        help="Figure width in inches (default: based on word count)")
    args = parser.parse_args()

    if args.plot_top_words < 0:
        parser.error("--plot-top-words must be >= 0")
    if args.figure_size is not None and (not np.isfinite(args.figure_size) or args.figure_size <= 0):
        parser.error("--figure-size must be finite and greater than 0")
    if args.prefix and (Path(args.prefix).name != args.prefix or "/" in args.prefix or "\\" in args.prefix):
        parser.error("--prefix must be a filename prefix, not a path")
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input matrix not found: {input_path}")
    try:
        words, counts = read_matrix(input_path)
        ppmi, npmi = compute_scores(counts, include_diagonal=args.include_diagonal)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    ppmi_csv, npmi_csv = outdir / f"{prefix}_ppmi.csv", outdir / f"{prefix}_npmi.csv"
    ppmi_png, npmi_png = outdir / f"{prefix}_ppmi_heatmap.png", outdir / f"{prefix}_npmi_heatmap.png"
    comparison_png = outdir / f"{prefix}_ppmi_npmi_comparison.png"

    write_matrix(ppmi_csv, words, ppmi)
    write_matrix(npmi_csv, words, npmi)
    if args.plot_top_words and args.plot_top_words < len(words):
        strength = counts.sum(axis=0) + counts.sum(axis=1)
        plot_indices = np.argsort(-strength, kind="stable")[:args.plot_top_words]
        plot_words = [words[index] for index in plot_indices]
        plot_ppmi = ppmi[np.ix_(plot_indices, plot_indices)]
        plot_npmi = npmi[np.ix_(plot_indices, plot_indices)]
    else:
        plot_words, plot_ppmi, plot_npmi = words, ppmi, npmi
    plot_note = f" (top {len(plot_words)} by co-occurrence)" if len(plot_words) < len(words) else ""
    plot_size = args.figure_size or max(10, min(28, len(plot_words) * 0.55))

    heatmap(ppmi_png, plot_words, plot_ppmi, f"PPMI{plot_note}", PPMI_CMAP, 0,
            max(float(ppmi.max()), 1e-12), "PPMI (natural log)", plot_size)
    heatmap(npmi_png, plot_words, plot_npmi, f"NPMI{plot_note}", NPMI_CMAP, -1, 1,
            "NPMI", plot_size)

    fig, axes = plt.subplots(1, 2, figsize=(plot_size * 2, plot_size * 0.88), constrained_layout=True)
    for ax, matrix, title, scale_label, cmap, vmin, vmax in (
        (axes[0], plot_ppmi, f"PPMI{plot_note}", "PPMI", PPMI_CMAP, 0, max(float(ppmi.max()), 1e-12)),
        (axes[1], plot_npmi, f"NPMI{plot_note}", "NPMI", NPMI_CMAP, -1, 1),
    ):
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(plot_words)), labels=plot_words, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(plot_words)), labels=plot_words, fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Word on right of pair")
        fig.colorbar(image, ax=ax, shrink=0.8, label=scale_label)
    axes[0].set_ylabel("Word on left of pair")
    fig.savefig(comparison_png, dpi=220)
    plt.close(fig)

    print(f"Words: {len(words)}")
    print(f"PPMI matrix: {ppmi_csv}")
    print(f"NPMI matrix: {npmi_csv}")
    print(f"PPMI heatmap: {ppmi_png}")
    print(f"NPMI heatmap: {npmi_png}")
    print(f"Comparison: {comparison_png}")


if __name__ == "__main__":
    main()
