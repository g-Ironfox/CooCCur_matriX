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
            if np.issubdtype(np.asarray(matrix).dtype, np.floating):
                values = [format(float(value), ".12g") for value in row]
            else:
                values = list(map(int, row.tolist()))
            writer.writerow([label, *values])


def iter_weighted_hops(adjacency: np.ndarray, max_hops: int,
                       decay: str = "linear", alpha: float | None = None):
    """Yield (k, R_k, S_k); yielded arrays are updated by the next step.

    Powers use Boolean multiplication. Linear mode preserves the original
    cumulative weighting; exponential mode sums alpha**k * B^k.
    """
    if decay not in ("linear", "exponential"):
        raise ValueError("decay must be linear or exponential")
    if decay == "exponential":
        if alpha is None or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("exponential decay requires alpha with 0 < alpha < 1")
    elif alpha is not None:
        raise ValueError("alpha can only be used with exponential decay")
    if max_hops < 1:
        return
    dtype = (float if decay == "exponential"
             else np.min_scalar_type(max_hops * (max_hops + 1) // 2))
    base = np.asarray(adjacency, dtype=bool)
    power = base.copy()
    reachability = power.copy()
    cumulative = np.zeros(power.shape, dtype=dtype)
    weighted = np.zeros(power.shape, dtype=dtype)
    for hop in range(1, max_hops + 1):
        if hop > 1:
            power = power @ base
            reachability |= power
        if decay == "linear":
            cumulative += power
            weighted += cumulative
        else:
            weighted += (alpha ** hop) * power
        yield hop, reachability, weighted


def plot_reachability(path: Path, labels: list[str], matrix: np.ndarray,
                      hop: int, stable: bool, cell_size: float,
                      weighted: bool = False) -> None:
    size = max(5, min(18, len(labels) * cell_size + 4))
    label_size = max(3, min(8, cell_size * 18))
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    vmax = max(1e-15, float(matrix.max())) if weighted else 1
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


WORD_STATS_FIELDS = ["rank", "word", "incoming_score", "outgoing_score", "min_score",
          "incoming_reachable", "outgoing_reachable", "self_score"]


def summarize_weighted_words(path: Path, include_diagonal: bool = False,
              sort_by: str = "incoming") -> list[dict]:
    """Row = source, column = destination; scores sum integer matrix entries."""
    if sort_by not in ("incoming", "outgoing", "min"):
        raise ValueError("sort_by must be incoming, outgoing or min")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        labels = header[1:]
        if not labels or len(set(labels)) != len(labels) or any(not w.strip() for w in labels):
            raise ValueError("CSV must contain unique, non-empty word labels")
        size = len(labels)
        incoming, outgoing = [0] * size, [0] * size
        in_count, out_count, diagonal = [0] * size, [0] * size, [0] * size
        row_count = 0
        for i, row in enumerate(reader):
            if i >= size or len(row) != size + 1:
                raise ValueError("matrix must be square with one data row per word")
            if row[0] != labels[i]:
                raise ValueError("row labels must match column labels in the same order")
            for j, value in enumerate(row[1:]):
                try:
                    score = float(value)
                except ValueError as error:
                    raise ValueError(f"invalid numeric weight at row {i + 1}, column {j + 1}") from error
                if not np.isfinite(score) or score < 0:
                    raise ValueError("weights must be finite and non-negative")
                if score.is_integer():
                    score = int(score)
                if i == j:
                    diagonal[i] = score
                    if not include_diagonal:
                        continue
                outgoing[i] += score
                incoming[j] += score
                if score > 0:
                    out_count[i] += 1
                    in_count[j] += 1
            row_count += 1
        if row_count != size:
            raise ValueError("matrix must be square with one data row per word")

    rows = [dict(word=word, incoming_score=incoming[i], outgoing_score=outgoing[i],
                 min_score=min(incoming[i], outgoing[i]),
                 incoming_reachable=in_count[i], outgoing_reachable=out_count[i],
                 self_score=diagonal[i]) for i, word in enumerate(labels)]
    rows.sort(key=lambda row: (-row[f"{sort_by}_score"], row["word"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def plot_word_stats(path: Path, rows: list[dict], include_diagonal: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, max(4, len(rows) * 0.3 + 1.8)),
                             sharey=True, constrained_layout=True)
    positions = list(range(len(rows)))
    for ax, field, title, color in [
        (axes[0], "min_score", "Min score (row sum, column sum)", "#7563A0"),
        (axes[1], "incoming_score", "Incoming score (column sum)", "#2F6F73"),
        (axes[2], "outgoing_score", "Outgoing score (row sum)", "#A85F50"),
    ]:
        values = [row[field] for row in rows]
        bars = ax.barh(positions, values, color=color)
        ax.bar_label(bars, padding=3, fontsize=7, fmt="%.3g")
        ax.set_xlim(0, max(values, default=0) * 1.22 or 1)
        ax.set_title(title)
        ax.set_xlabel("Sum of weighted Boolean scores")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_yticks(positions, labels=[row["word"] for row in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Word")
    diagonal_note = "included" if include_diagonal else "excluded"
    fig.suptitle(f"Weighted hop scores by word — diagonal {diagonal_note}")
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def write_word_stats(input_path: Path, outdir: Path, top: int = 30,
        include_diagonal: bool = False, sort_by: str = "incoming") -> list[dict]:
    if top < 0:
        raise ValueError("--top must be >= 0")
    rows = summarize_weighted_words(input_path, include_diagonal, sort_by)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_word_stats.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=WORD_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    shown = rows[:top] if top else rows
    # Keep complete word labels readable even when --top 0 selects all words.
    page_size = 50
    pages = [shown[start:start + page_size] for start in range(0, len(shown), page_size)]
    for page, subset in enumerate(pages, 1):
        suffix = f"_{page:03d}" if len(pages) > 1 else ""
        png_path = outdir / f"{prefix}_word_stats{suffix}.png"
        plot_word_stats(png_path, subset, include_diagonal)
        print(f"Plot: {png_path}")
    print(f"Words: {len(rows)}; sorted by: {sort_by} score")
    print(f"Statistics: {csv_path}")
    return rows



def run(input_path: Path, outdir: Path, cell_size: float = 0.35,
        top: int = 30, sort_by: str = "incoming",
        include_diagonal: bool = False, decay: str = "linear",
        alpha: float | None = None) -> int:
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell size must be greater than zero")
    if top < 0:
        raise ValueError("--top must be >= 0")
    if sort_by not in ("incoming", "outgoing", "min"):
        raise ValueError("sort_by must be incoming, outgoing or min")
    labels, adjacency = read_matrix(input_path)
    if not labels:
        raise ValueError("input graph must contain at least one vertex")

    if decay == "exponential":
        if alpha is None or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("exponential decay requires alpha with 0 < alpha < 1")
    elif decay != "linear":
        raise ValueError("decay must be linear or exponential")
    elif alpha is not None:
        raise ValueError("--alpha can only be used with exponential decay")

    stable_hop = stabilization_hops(adjacency)
    outdir.mkdir(parents=True, exist_ok=True)
    for hop, reachability, weighted in iter_weighted_hops(
            adjacency, stable_hop, decay, alpha):
        stable = hop == stable_hop
        write_matrix(outdir / f"reachability_{hop:03d}.csv", labels, reachability)
        plot_reachability(outdir / f"reachability_{hop:03d}.png", labels,
                          reachability, hop, stable, cell_size)
        write_matrix(outdir / f"weighted_{hop:03d}.csv", labels, weighted)
        plot_reachability(outdir / f"weighted_{hop:03d}.png", labels,
                          weighted, hop, stable, cell_size, weighted=True)
        write_word_stats(outdir / f"weighted_{hop:03d}.csv", outdir,
                         top, include_diagonal, sort_by)

    print(f"Vertices: {len(labels)}")
    print(f"Stabilization hop: {stable_hop}")
    print(f"Weighting: {decay}" + (f" (alpha={alpha:g})" if alpha is not None else ""))
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
    parser.add_argument("--top", type=int, default=30,
                        help="Words to plot at each hop; 0 plots all, 50 per image (default: 30)")
    parser.add_argument("--sort-by", choices=["incoming", "outgoing", "min"], default="incoming",
                        help="Rank each hop by column sum, row sum or their minimum (default: incoming)")
    parser.add_argument("--include-diagonal", action="store_true",
                        help="Include self-return values in word statistics; hop matrices are unchanged")
    parser.add_argument("--decay", choices=["linear", "exponential"], default="linear",
                        help="Hop weighting mode (default: linear)")
    parser.add_argument("--alpha", type=float,
                        help="Exponential attenuation factor; required with --decay exponential (0 < alpha < 1)")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.cell_size,
            args.top, args.sort_by, args.include_diagonal, args.decay, args.alpha)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
