#!/usr/bin/env python3
"""Plot left/right entropy, PPMI, and NPMI in one four-quadrant heatmap."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

import association
import phrase_entropy as entropy


SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "muted_vermilion", ["#F3E6E2", "#D69588", "#A64036"]
)
NPMI_CMAP = LinearSegmentedColormap.from_list(
    "teal_neutral_vermilion", ["#2F6F73", "#E9E4E1", "#A64036"]
)
MISSING_COLOR = "#FFFFFF"
GRID_COLOR = "#D3CDCA"
QUADRANT_COLOR = "#8F8580"


def write_matrix(path: Path, words: list[str], matrix: np.ndarray) -> None:
    """Write a square metric matrix, leaving non-finite cells blank."""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word", *words])
        for word, row in zip(words, matrix):
            writer.writerow([
                word,
                *[f"{value:.8g}" if math.isfinite(float(value)) else "" for value in row],
            ])


def compute_metrics(markdown: str, min_length: int = 3,
                    candidates: set[tuple[str, str]] | None = None):
    """Build one aligned vocabulary and all phrase and association metrics."""
    vocabulary = entropy.collect_vocabulary(markdown, min_length)
    phrases = entropy.collect_contexts(markdown, min_length, candidates)
    words, counts, left, right, joint = entropy.build_matrices(phrases, vocabulary)

    if counts.size and counts.sum() > 0:
        ppmi, npmi = association.compute_scores(counts)
    else:
        ppmi = np.full(counts.shape, np.nan)
        npmi = np.full(counts.shape, np.nan)
    return words, counts, left, right, ppmi, npmi, joint, phrases


def select_indices(counts: np.ndarray, top_words: int,
                   min_phrase_frequency: int = 0) -> np.ndarray:
    """Select plot axes only; saved CSV matrices always retain the full vocabulary."""
    if min_phrase_frequency > 0:
        eligible = counts >= min_phrase_frequency
        rows, columns = np.nonzero(eligible)
        indices = np.unique(np.concatenate((rows, columns))) if len(rows) else np.array([], dtype=int)
        plot_counts = np.where(eligible, counts, 0)
    else:
        indices = np.arange(counts.shape[0])
        plot_counts = counts
    if top_words and top_words < len(indices):
        strength = plot_counts.sum(axis=0) + plot_counts.sum(axis=1)
        ranked = indices[np.argsort(-strength[indices], kind="stable")[:top_words]]
        indices = np.sort(ranked)
    return indices


def _finite_max(matrix: np.ndarray, default: float = 1.0) -> float:
    values = matrix[np.isfinite(matrix)]
    return max(float(values.max()), default) if values.size else default


def shared_entropy_max(left: np.ndarray, right: np.ndarray) -> float:
    """Return one absolute scale for comparing left and right entropy in nats."""
    return max(_finite_max(left), _finite_max(right))


def _quadrant_polygons(rows: np.ndarray, columns: np.ndarray, quadrant: str):
    """Return one rectangle polygon per selected cell quadrant."""
    x0, x1 = columns - 0.5, columns + 0.5
    y0, y1 = rows - 0.5, rows + 0.5
    xc, yc = columns, rows
    corners = {
        "top-left": np.stack((
            np.column_stack((x0, y0)), np.column_stack((xc, y0)),
            np.column_stack((xc, yc)), np.column_stack((x0, yc))), axis=1),
        "top-right": np.stack((
            np.column_stack((xc, y0)), np.column_stack((x1, y0)),
            np.column_stack((x1, yc)), np.column_stack((xc, yc))), axis=1),
        "bottom-left": np.stack((
            np.column_stack((x0, yc)), np.column_stack((xc, yc)),
            np.column_stack((xc, y1)), np.column_stack((x0, y1))), axis=1),
        "bottom-right": np.stack((
            np.column_stack((xc, yc)), np.column_stack((x1, yc)),
            np.column_stack((x1, y1)), np.column_stack((xc, y1))), axis=1),
    }
    return corners[quadrant]


def four_metric_heatmap(path: Path, words: list[str], counts: np.ndarray,
                        left: np.ndarray, right: np.ndarray,
                        ppmi: np.ndarray, npmi: np.ndarray,
                        figure_size: float) -> None:
    """Draw four aligned metrics as four quadrants inside each matrix cell."""
    fig = plt.figure(figsize=(figure_size, figure_size + 0.4))
    ax = fig.add_axes([0.09, 0.11, 0.68, 0.82])
    ax.set_facecolor(MISSING_COLOR)

    entropy_vmax = shared_entropy_max(left, right)
    specs = (
        ("Left entropy", left, "top-left", SEQUENTIAL_CMAP, 0, entropy_vmax),
        ("Right entropy", right, "top-right", SEQUENTIAL_CMAP, 0, entropy_vmax),
        ("PPMI", ppmi, "bottom-left", SEQUENTIAL_CMAP, 0, _finite_max(ppmi)),
        ("NPMI", npmi, "bottom-right", NPMI_CMAP, -1, 1),
    )
    scalar_mappables = []
    for name, matrix, quadrant, cmap, vmin, vmax in specs:
        norm = Normalize(vmin=vmin, vmax=vmax)
        valid = np.isfinite(matrix)
        if name in {"PPMI", "NPMI"}:
            valid &= counts > 0
        rows, columns = np.nonzero(valid)
        if len(rows):
            ax.add_collection(PolyCollection(
                _quadrant_polygons(rows, columns, quadrant),
                facecolors=[plt.get_cmap(cmap)(norm(value)) for value in matrix[rows, columns]],
                edgecolors="none", antialiaseds=False, zorder=2,
            ))
        if name == "Left entropy":
            scalar_mappables.append((ScalarMappable(norm=norm, cmap=cmap),
                                     "Boundary entropy (nats)"))
        elif name != "Right entropy":
            scalar_mappables.append((ScalarMappable(norm=norm, cmap=cmap), name))

    extent = len(words) - 0.5 if words else 0.5
    ax.set_xlim(-0.5, extent)
    ax.set_ylim(extent, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(words)), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)), labels=words, fontsize=8)
    ax.set_xlabel("Second word B (ordered pair A -> B)")
    ax.set_ylabel("First word A")
    ax.set_title(
        "Four metrics per directed word pair\n"
        "Top-left: left entropy | Top-right: right entropy | Bottom-left: PPMI | Bottom-right: NPMI",
        fontsize=10,
        pad=9,
    )
    if words:
        cell_edges = np.arange(-0.5, len(words) + 0.5, 1)
        centers = np.arange(len(words))
        cell_lines = ([[(x, -0.5), (x, extent)] for x in cell_edges]
                      + [[(-0.5, y), (extent, y)] for y in cell_edges])
        quadrant_lines = ([[(x, -0.5), (x, extent)] for x in centers]
                          + [[(-0.5, y), (extent, y)] for y in centers])
        ax.add_collection(LineCollection(cell_lines, colors=GRID_COLOR,
                                         linewidths=0.35, zorder=3))
        ax.add_collection(LineCollection(quadrant_lines, colors=QUADRANT_COLOR,
                                         linewidths=0.45, zorder=4))
        for spine in ax.spines.values():
            spine.set_color(QUADRANT_COLOR)
            spine.set_linewidth(0.7)

    fig.canvas.draw()
    plot_bounds = ax.get_position()
    bar_gap = 0.05
    bar_height = (plot_bounds.height - (len(scalar_mappables) - 1) * bar_gap) / len(scalar_mappables)
    bar_width = 0.013
    bar_x = plot_bounds.x1 + 0.04
    for index, (mappable, label) in enumerate(scalar_mappables):
        y = plot_bounds.y1 - bar_height - index * (bar_height + bar_gap)
        cax = fig.add_axes([bar_x, y, bar_width, bar_height])
        ticks = [-1, 0, 1] if label == "NPMI" else np.linspace(
            mappable.norm.vmin, mappable.norm.vmax, 3)
        colorbar = fig.colorbar(mappable, cax=cax, orientation="vertical", ticks=ticks)
        colorbar.ax.set_title(label, fontsize=8, pad=4)
        colorbar.outline.set_linewidth(0.45)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def metric_heatmap(path: Path, words: list[str], values: np.ndarray, title: str,
                   cmap, vmin: float, vmax: float, figure_size: float,
                   valid: np.ndarray | None = None, ticks=None) -> None:
    """Write one square metric heatmap with white missing cells."""
    masked = np.ma.masked_invalid(values)
    if valid is not None:
        masked = np.ma.masked_where(~valid, masked)
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad(MISSING_COLOR)

    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
    ax.set_facecolor(MISSING_COLOR)
    size = len(words)
    if size:
        image = ax.imshow(masked, cmap=color_map, vmin=vmin, vmax=vmax,
                          interpolation="nearest", aspect="equal")
    else:
        image = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=color_map)
        ax.text(0.5, 0.5, "No phrase axes", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xticks(range(size), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(size), labels=words, fontsize=8)
    ax.set_xlabel("Second word B (ordered pair A -> B)")
    ax.set_ylabel("First word A")
    ax.set_title(title)
    if size:
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color=GRID_COLOR, linewidth=0.35)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_color(QUADRANT_COLOR)
            spine.set_linewidth(0.7)
    fig.colorbar(image, ax=ax, label=title, shrink=0.82, ticks=ticks)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def left_right_entropy_heatmap(path: Path, words: list[str], left: np.ndarray,
                               right: np.ndarray, figure_size: float,
                               vmax: float) -> None:
    """Draw left and right entropy in the upper-left and lower-right cell halves."""
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
    ax.set_facecolor(MISSING_COLOR)
    norm = Normalize(vmin=0, vmax=vmax)
    for matrix, upper_left in ((left, True), (right, False)):
        rows, columns = np.nonzero(np.isfinite(matrix))
        if not len(rows):
            continue
        x0, x1 = columns - 0.5, columns + 0.5
        y0, y1 = rows - 0.5, rows + 0.5
        if upper_left:
            triangles = np.stack((np.column_stack((x0, y0)),
                                  np.column_stack((x1, y0)),
                                  np.column_stack((x0, y1))), axis=1)
        else:
            triangles = np.stack((np.column_stack((x1, y0)),
                                  np.column_stack((x1, y1)),
                                  np.column_stack((x0, y1))), axis=1)
        ax.add_collection(PolyCollection(
            triangles, facecolors=SEQUENTIAL_CMAP(norm(matrix[rows, columns])),
            edgecolors=MISSING_COLOR, linewidths=0.25, antialiaseds=False,
        ))
    size = len(words)
    extent = size - 0.5 if size else 0.5
    ax.set_xlim(-0.5, extent)
    ax.set_ylim(extent, -0.5)
    ax.set_aspect("equal")
    if not size:
        ax.text(0.5, 0.5, "No phrase axes", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xticks(range(size), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(size), labels=words, fontsize=8)
    ax.set_xlabel("Second word B (ordered pair A -> B)")
    ax.set_ylabel("First word A")
    ax.set_title("Left/right boundary entropy (nats)\nUpper-left: left; lower-right: right")
    if size:
        edges = np.arange(-0.5, size + 0.5, 1)
        ax.add_collection(LineCollection(
            [[(x, -0.5), (x, extent)] for x in edges]
            + [[(-0.5, y), (extent, y)] for y in edges],
            colors=GRID_COLOR, linewidths=0.35,
        ))
        for spine in ax.spines.values():
            spine.set_color(QUADRANT_COLOR)
            spine.set_linewidth(0.7)
    fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP), ax=ax,
                 label="Boundary entropy (nats)", shrink=0.82)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot left/right entropy, PPMI, and NPMI in one heatmap."
    )
    parser.add_argument("input", nargs="?", default="1.md", help="Source Markdown (default: 1.md)")
    parser.add_argument("--outdir", default="output/phrase_metrics", help="Output directory")
    parser.add_argument("--prefix", help="Output filename prefix (default: input filename stem)")
    parser.add_argument("--phrase-candidates", help="Optional candidate CSV with left and right columns")
    parser.add_argument("--plot-min-phrase-frequency", type=int, default=0,
                        help="Filter plotted pairs and axes by phrase count; CSV data is unchanged")
    parser.add_argument("--min-length", type=int, default=3,
                        help="Minimum token length for phrase words (default: 3)")
    parser.add_argument("--plot-top-words", type=int, default=0,
                        help="Displayed axes by pair strength; 0 displays all (default: 0)")
    parser.add_argument("--figure-size", type=float,
                        help="Figure width and height in inches (default: based on word count)")
    args = parser.parse_args()
    if args.plot_min_phrase_frequency < 0:
        parser.error("--plot-min-phrase-frequency must be >= 0")
    if args.min_length < 1:
        parser.error("--min-length must be >= 1")
    if args.plot_top_words < 0:
        parser.error("--plot-top-words must be >= 0")
    if args.figure_size is not None and (not math.isfinite(args.figure_size) or args.figure_size <= 0):
        parser.error("--figure-size must be finite and greater than 0")
    if args.prefix is not None and (Path(args.prefix).name != args.prefix
                                    or any(char in args.prefix for char in '/\\:<>"|?*')):
        parser.error("--prefix must be a filename prefix, not a path")

    input_path = Path(args.input)
    try:
        markdown = input_path.read_text(encoding="utf-8-sig")
        candidates = (entropy.read_candidates(Path(args.phrase_candidates))
                      if args.phrase_candidates else None)
        words, counts, left, right, ppmi, npmi, joint, phrases = compute_metrics(
            markdown, args.min_length, candidates)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    both_sides = np.isfinite(left) & np.isfinite(right)
    left_minus_right = np.full(left.shape, np.nan)
    left_minus_right[both_sides] = left[both_sides] - right[both_sides]
    matrices = {
        "phrase_counts": counts,
        "left_entropy": left,
        "right_entropy": right,
        "joint_entropy": joint,
        "left_minus_right_entropy": left_minus_right,
        "ppmi": ppmi,
        "npmi": npmi,
    }
    for name, matrix in matrices.items():
        output = outdir / f"{prefix}_{name}.csv"
        write_matrix(output, words, matrix)
        print(f"{name}: {output}")
    report = outdir / f"{prefix}_phrase_boundaries.csv"
    entropy.write_report(report, phrases)

    indices = select_indices(counts, args.plot_top_words, args.plot_min_phrase_frequency)
    plot_words = [words[index] for index in indices]
    subset = np.ix_(indices, indices)
    visible_pairs = (counts >= args.plot_min_phrase_frequency
                     if args.plot_min_phrase_frequency > 0 else counts > 0)
    plot_counts = np.where(visible_pairs, counts, 0)
    plot_left = np.where(visible_pairs, left, np.nan)
    plot_right = np.where(visible_pairs, right, np.nan)
    plot_joint = np.where(visible_pairs, joint, np.nan)
    plot_ppmi = np.where(visible_pairs, ppmi, np.nan)
    plot_npmi = np.where(visible_pairs, npmi, np.nan)
    plot_difference = np.where(visible_pairs, left_minus_right, np.nan)
    plot_both_sides = np.isfinite(plot_left) & np.isfinite(plot_right)
    size = args.figure_size or max(10, min(28, len(plot_words) * 0.55))
    image = outdir / f"{prefix}_four_metrics_heatmap.png"
    four_metric_heatmap(image, plot_words, plot_counts[subset], plot_left[subset], plot_right[subset],
                        plot_ppmi[subset], plot_npmi[subset], size)
    image_specs = (
        ("left_right_entropy_heatmap", "Left/right boundary entropy", None,
         SEQUENTIAL_CMAP, 0, shared_entropy_max(plot_left, plot_right), None, None),
        ("joint_entropy_heatmap", "Joint boundary entropy", plot_joint,
         SEQUENTIAL_CMAP, 0, _finite_max(plot_joint), None, None),
        ("left_minus_right_entropy_heatmap", "Left entropy minus right entropy (nats)",
         plot_difference, NPMI_CMAP, -_finite_max(np.abs(plot_difference)),
         _finite_max(np.abs(plot_difference)), plot_both_sides, None),
        ("ppmi_heatmap", "PPMI", plot_ppmi, SEQUENTIAL_CMAP, 0, _finite_max(plot_ppmi),
         plot_counts > 0, None),
        ("npmi_heatmap", "NPMI", plot_npmi, NPMI_CMAP, -1, 1, plot_counts > 0, [-1, 0, 1]),
    )
    for suffix, title, matrix, cmap, vmin, vmax, valid, ticks in image_specs:
        path = outdir / f"{prefix}_{suffix}.png"
        if suffix == "left_right_entropy_heatmap":
            left_right_entropy_heatmap(
                path, plot_words, plot_left[subset], plot_right[subset], size,
                shared_entropy_max(plot_left, plot_right))
        else:
            metric_heatmap(path, plot_words, matrix[subset], title, cmap,
                           vmin, vmax, size, valid[subset] if valid is not None else None, ticks)
        print(f"{suffix}: {path}")
    print(f"Words: {len(words)}; displayed axes: {len(plot_words)}")
    print(f"Phrase boundary report: {report}")
    print(f"Four-metric heatmap: {image}")


if __name__ == "__main__":
    main()
