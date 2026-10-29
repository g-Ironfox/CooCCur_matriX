#!/usr/bin/env python3
"""Plot incoming and outgoing scores directly from a weighted matrix CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


FIELDS = ["rank", "word", "incoming_score", "outgoing_score", "min_score",
          "max_score", "directional_imbalance"]


def summarize_matrix(path: Path, include_diagonal: bool = False,
                     sort_by: str = "max") -> list[dict]:
    if sort_by not in ("max", "min", "total"):
        raise ValueError("sort_by must be max, min or total")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        labels = header[1:]
        if not labels or len(set(labels)) != len(labels) or any(not word.strip() for word in labels):
            raise ValueError("CSV must contain unique, non-empty word labels")

        size = len(labels)
        incoming = [0] * size
        outgoing = [0] * size
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
                if i == j and not include_diagonal:
                    continue
                outgoing[i] += score
                incoming[j] += score
            row_count += 1
        if row_count != size:
            raise ValueError("matrix must be square with one data row per word")

    rows = [
        {
            "word": word,
            "incoming_score": incoming[i],
            "outgoing_score": outgoing[i],
            "min_score": min(incoming[i], outgoing[i]),
            "max_score": max(incoming[i], outgoing[i]),
            "directional_imbalance": (
                (incoming[i] - outgoing[i]) / (incoming[i] + outgoing[i])
                if incoming[i] + outgoing[i] else 0.0
            ),
        }
        for i, word in enumerate(labels)
    ]

    def key(row: dict) -> tuple[int, str]:
        incoming_score = row["incoming_score"]
        outgoing_score = row["outgoing_score"]
        if sort_by == "max":
            score = max(incoming_score, outgoing_score)
        elif sort_by == "min":
            score = min(incoming_score, outgoing_score)
        else:
            score = incoming_score + outgoing_score
        return -score, row["word"]

    rows.sort(key=key)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def plot_scatter(path: Path, rows: list[dict], sort_by: str,
                 label_words: set[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    incoming = np.asarray([row["incoming_score"] for row in rows], dtype=float)
    outgoing = np.asarray([row["outgoing_score"] for row in rows], dtype=float)
    balanced = np.asarray([row["min_score"] for row in rows], dtype=float)
    imbalance = np.asarray([row["directional_imbalance"] for row in rows], dtype=float)
    peak = float(balanced.max(initial=0.0))
    sizes = 22 + 78 * np.sqrt(balanced / peak) if peak else np.full(len(rows), 22.0)
    points = ax.scatter(outgoing, incoming, s=sizes, c=imbalance, cmap="RdBu_r",
                        norm=Normalize(-1, 1), alpha=0.8, edgecolors="white",
                        linewidths=0.45, zorder=2)
    labels = label_words if label_words is not None else {row["word"] for row in rows}
    for row in rows:
        if row["word"] in labels:
            ax.annotate(row["word"], (row["outgoing_score"], row["incoming_score"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
    limit = max(float(incoming.max(initial=0)), float(outgoing.max(initial=0)))
    limit = limit * 1.12 or 1
    ax.plot([0, limit], [0, limit], color="#555555", linewidth=1,
            linestyle="--", alpha=0.6, zorder=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Outgoing score (row sum)")
    ax.set_ylabel("Incoming score (column sum)")
    ax.set_title(f"Incoming vs. outgoing scores\nsize = min(in, out); ranked by {sort_by}")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.colorbar(points, ax=ax,
                 label="Directional imbalance (incoming - outgoing) / (incoming + outgoing)")
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def run(input_path: Path, outdir: Path, top: int = 30,
        sort_by: str = "max", include_diagonal: bool = False,
        page_size: int = 0) -> list[dict]:
    if top < 0:
        raise ValueError("--top must be >= 0")
    if page_size < 0:
        raise ValueError("--page-size must be >= 0")
    rows = summarize_matrix(input_path, include_diagonal, sort_by)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_inout_stats.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    primary = rows[:top] if top else rows
    label_words = {row["word"] for row in primary}
    if top:
        label_words.update(row["word"] for row in
                           sorted(rows, key=lambda row: (-row["incoming_score"], row["word"]))[:8])
        label_words.update(row["word"] for row in
                           sorted(rows, key=lambda row: (-row["outgoing_score"], row["word"]))[:8])
    ordered_labels = [row["word"] for row in rows if row["word"] in label_words]
    pages = ([ordered_labels[start:start + page_size]
              for start in range(0, len(ordered_labels), page_size)]
             if page_size else [ordered_labels])
    for page, page_labels in enumerate(pages, 1):
        suffix = f"_{page:03d}" if len(pages) > 1 else ""
        image_path = outdir / f"{prefix}_inout_scatter{suffix}.png"
        plot_scatter(image_path, rows, sort_by, set(page_labels))
        print(f"Plot: {image_path}")
    print(f"Words: {len(rows)}; labels: {len(label_words)}; sorted by: {sort_by}")
    print(f"Statistics: {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Weighted square matrix CSV; rows are sources and columns are destinations")
    parser.add_argument("--outdir", default="output/weighted_inout_scatter",
                        help="Output directory")
    parser.add_argument("--top", type=int, default=30,
                        help="Primary ranked words to label; 0 labels all (default: 30)")
    parser.add_argument("--page-size", type=int, default=0,
                        help="Labels per image; all points remain in every plot (default: 0)")
    parser.add_argument("--sort-by", choices=["max", "min", "total"], default="max",
                        help="Rank by max(incoming, outgoing), min(incoming, outgoing), or their sum")
    parser.add_argument("--include-diagonal", action="store_true",
                        help="Include self-return values in incoming and outgoing scores")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.top, args.sort_by,
            args.include_diagonal, args.page_size)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
