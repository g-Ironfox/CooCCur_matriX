#!/usr/bin/env python3
"""Plot in/out scores in mean-difference (u, v) coordinates."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

from weighted_inout_scatter import summarize_matrix


FIELDS = ["rank", "word", "incoming_score", "outgoing_score", "u_score",
          "v_score", "min_score", "directional_imbalance"]


def transform_rows(rows: list[dict], sort_by: str = "projection") -> list[dict]:
    if sort_by not in ("projection", "min", "max"):
        raise ValueError("sort_by must be projection, min or max")
    transformed = []
    for row in rows:
        incoming = float(row["incoming_score"])
        outgoing = float(row["outgoing_score"])
        transformed.append({
            "word": row["word"],
            "incoming_score": incoming,
            "outgoing_score": outgoing,
            "u_score": (incoming + outgoing) / 2,
            "v_score": (incoming - outgoing) / 2,
            "min_score": float(row["min_score"]),
            "directional_imbalance": float(row["directional_imbalance"]),
        })
    score_key = {
        "projection": "u_score",
        "min": "min_score",
        "max": None,
    }[sort_by]
    if score_key:
        transformed.sort(key=lambda row: (-row[score_key], row["word"]))
    else:
        transformed.sort(key=lambda row: (
            -max(row["incoming_score"], row["outgoing_score"]), row["word"]))
    for rank, row in enumerate(transformed, 1):
        row["rank"] = rank
    return transformed


def plot_scatter(path: Path, rows: list[dict], sort_by: str,
                 label_words: set[str] | None = None) -> None:
    u_values = np.asarray([row["u_score"] for row in rows], dtype=float)
    v_values = np.asarray([row["v_score"] for row in rows], dtype=float)
    balanced = np.asarray([row["min_score"] for row in rows], dtype=float)
    imbalance = np.asarray([row["directional_imbalance"] for row in rows], dtype=float)
    peak = float(balanced.max(initial=0.0))
    sizes = 22 + 78 * np.sqrt(balanced / peak) if peak else np.full(len(rows), 22.0)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    points = ax.scatter(u_values, v_values, s=sizes, c=imbalance, cmap="RdBu_r",
                        norm=Normalize(-1, 1), alpha=0.8, edgecolors="white",
                        linewidths=0.45, zorder=2)
    labels = label_words if label_words is not None else {row["word"] for row in rows}
    for row in rows:
        if row["word"] in labels:
            vertical_offset = 4 if row["rank"] % 2 else -7
            ax.annotate(row["word"], (row["u_score"], row["v_score"]),
                        xytext=(4, vertical_offset), textcoords="offset points",
                        fontsize=7)

    u_limit = float(u_values.max(initial=0)) * 1.12 or 1
    v_limit = float(np.abs(v_values).max(initial=0)) * 1.12 or 1
    boundary = min(u_limit, v_limit)
    ax.plot([0, u_limit], [0, 0], color="#444444", linewidth=1,
            linestyle="--", alpha=0.65, zorder=1)
    ax.plot([0, boundary], [0, boundary], color="#888888", linewidth=0.9,
            linestyle=":", alpha=0.65, zorder=1)
    ax.plot([0, boundary], [0, -boundary], color="#888888", linewidth=0.9,
            linestyle=":", alpha=0.65, zorder=1)
    ax.set_xlim(0, u_limit)
    ax.set_ylim(-v_limit, v_limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u = (incoming + outgoing) / 2 (mean)")
    ax.set_ylabel("v = (incoming - outgoing) / 2 (half-difference)")
    ax.set_title(f"In/out scores in mean-difference coordinates\nsize = min(in, out); ranked by {sort_by}")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.colorbar(points, ax=ax,
                 label="Relative imbalance (incoming - outgoing) / (incoming + outgoing)")
    try:
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)


def run(input_path: Path, outdir: Path, top: int = 30,
        sort_by: str = "projection", include_diagonal: bool = False,
        page_size: int = 0) -> list[dict]:
    if top < 0:
        raise ValueError("--top must be >= 0")
    if page_size < 0:
        raise ValueError("--page-size must be >= 0")
    source_rows = summarize_matrix(input_path, include_diagonal, "total")
    rows = transform_rows(source_rows, sort_by)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_uv_stats.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    primary = rows[:top] if top else rows
    effective_page_size = page_size or (40 if top == 0 and len(primary) > 40 else 0)
    pages = ([primary[start:start + effective_page_size]
              for start in range(0, len(primary), effective_page_size)]
             if effective_page_size else [primary])
    for page, page_rows in enumerate(pages, 1):
        suffix = f"_{page:03d}" if len(pages) > 1 else ""
        image_path = outdir / f"{prefix}_uv_scatter{suffix}.png"
        plot_scatter(image_path, rows, sort_by,
                     {row["word"] for row in page_rows})
        print(f"Plot: {image_path}")
    print(f"Words: {len(rows)}; labels: {len(primary)}; sorted by: {sort_by}")
    print(f"Statistics: {csv_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Weighted square matrix CSV; rows are sources and columns are destinations")
    parser.add_argument("--outdir", default="output/inout_uv_scatter",
                        help="Output directory")
    parser.add_argument("--top", type=int, default=30,
                        help="Primary ranked words to label; 0 labels all (default: 30)")
    parser.add_argument("--page-size", type=int, default=0,
                        help="Labels per image; all points remain in every plot (default: 0)")
    parser.add_argument("--sort-by", choices=["projection", "min", "max"], default="projection",
                        help="Rank by diagonal projection u, min(in, out), or max(in, out)")
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
