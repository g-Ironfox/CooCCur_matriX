#!/usr/bin/env python3
"""Plot incoming and outgoing scores directly from a weighted matrix CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = ["rank", "word", "incoming_score", "outgoing_score", "min_score"]


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
                    score = int(value)
                except ValueError as error:
                    raise ValueError(f"invalid integer weight at row {i + 1}, column {j + 1}") from error
                if score < 0:
                    raise ValueError("weights must be non-negative integers")
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


def plot_scatter(path: Path, rows: list[dict], sort_by: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    incoming = [row["incoming_score"] for row in rows]
    outgoing = [row["outgoing_score"] for row in rows]
    ax.scatter(outgoing, incoming, s=38, alpha=0.78, color="#2F6F73",
               edgecolors="white", linewidths=0.6)
    for row in rows:
        ax.annotate(row["word"], (row["outgoing_score"], row["incoming_score"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)
    limit = max([*incoming, *outgoing], default=0)
    ax.set_xlim(0, limit * 1.12 or 1)
    ax.set_ylim(0, limit * 1.12 or 1)
    ax.set_xlabel("Outgoing score (row sum)")
    ax.set_ylabel("Incoming score (column sum)")
    ax.set_title(f"Incoming vs. outgoing scores (ranked by {sort_by})")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
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

    shown = rows[:top] if top else rows
    pages = ([shown[start:start + page_size]
              for start in range(0, len(shown), page_size)]
             if page_size else [shown])
    for page, subset in enumerate(pages, 1):
        suffix = f"_{page:03d}" if len(pages) > 1 else ""
        image_path = outdir / f"{prefix}_inout_scatter{suffix}.png"
        plot_scatter(image_path, subset, sort_by)
        print(f"Plot: {image_path}")
    print(f"Words: {len(rows)}; sorted by: {sort_by}")
    print(f"Statistics: {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Weighted square matrix CSV; rows are sources and columns are destinations")
    parser.add_argument("--outdir", default="output/weighted_inout_scatter",
                        help="Output directory")
    parser.add_argument("--top", type=int, default=30,
                        help="Words to plot; 0 plots all (default: 30)")
    parser.add_argument("--page-size", type=int, default=0,
                        help="Words per image; 0 disables pagination (default: 0)")
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
