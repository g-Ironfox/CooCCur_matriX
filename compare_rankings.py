#!/usr/bin/env python3
"""Compare linear n-hop, truncated Katz-4, and TextRank word rankings."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bool_hop_stability import iter_weighted_hops
from textrank import textrank


def read_boolean_matrix(path: Path) -> tuple[list[str], np.ndarray]:
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
    try:
        values = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    except ValueError as error:
        raise ValueError("matrix contains a non-numeric value") from error
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("input must be a Boolean matrix containing only 0 and 1")
    if np.diag(values).any():
        raise ValueError("self-loops are not supported; diagonal entries must be zero")
    return labels, values.astype(float)


def rank_scores(scores: np.ndarray, labels: list[str]) -> np.ndarray:
    order = sorted(range(len(labels)), key=lambda i: (-float(scores[i]), labels[i]))
    ranks = np.empty(len(labels), dtype=int)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


def compare(labels: list[str], adjacency: np.ndarray, beta: float | None = None,
            hop: int = 4):
    """Return rank and score arrays for all three methods."""
    if hop < 1:
        raise ValueError("hop must be >= 1")
    binary = np.asarray(adjacency, dtype=bool)
    linear = next(weighted for current_hop, _, weighted in iter_weighted_hops(binary, hop)
                  if current_hop == hop)
    np.fill_diagonal(linear, 0)
    linear_in = linear.sum(axis=0).astype(float)
    linear_out = linear.sum(axis=1).astype(float)
    linear_score = np.minimum(linear_in, linear_out)

    max_row_sum = float(np.max(adjacency.sum(axis=1), initial=0.0))
    effective_beta = beta if beta is not None else (0.85 / max_row_sum if max_row_sum else 0.85)
    if not np.isfinite(effective_beta) or effective_beta < 0:
        raise ValueError("beta must be finite and >= 0")
    term = effective_beta * adjacency
    katz = np.zeros_like(adjacency, dtype=float)
    for katz_hop in range(1, 5):
        katz += term
        if katz_hop < 4:
            term = effective_beta * (adjacency @ term)
    if not np.isfinite(katz).all():
        raise ValueError("Katz scores overflowed; reduce --beta")
    np.fill_diagonal(katz, 0)
    katz_in = katz.sum(axis=0)
    katz_out = katz.sum(axis=1)
    katz_score = np.minimum(katz_in, katz_out)

    pagerank, _ = textrank(adjacency)
    methods = {
        f"linear_{hop}hop": {"score": linear_score, "incoming": linear_in, "outgoing": linear_out},
        "katz_4hop": {"score": katz_score, "incoming": katz_in, "outgoing": katz_out},
        "textrank": {"score": pagerank},
    }
    for values in methods.values():
        values["rank"] = rank_scores(values["score"], labels)
    return methods, effective_beta


def write_scores(path: Path, labels: list[str], methods: dict) -> None:
    fields = ["word"]
    for name in methods:
        fields.extend(f"{name}_{field}" for field in ("rank", "score", "incoming", "outgoing")
                      if field in methods[name])
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, word in enumerate(labels):
            row = {"word": word}
            for name, values in methods.items():
                row[f"{name}_rank"] = int(values["rank"][index])
                for field in ("score", "incoming", "outgoing"):
                    if field in values:
                        row[f"{name}_{field}"] = format(float(values[field][index]), ".12g")
            writer.writerow(row)


def plot_comparison(path: Path, labels: list[str], methods: dict, top: int,
                    beta: float, hop: int) -> None:
    specs = [
        (f"linear_{hop}hop", f"Linear decay ({hop} hops)", "#4A8F5B"),
        ("katz_4hop", "Katz (K1-K4)", "#2878B5"),
        ("textrank", "TextRank (d=0.85)", "#D8782C"),
    ]
    shown = min(top, len(labels)) if top else len(labels)
    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, shown * 0.30)),
                             squeeze=False, constrained_layout=True)
    for ax, (name, title, color) in zip(axes[0], specs):
        values = methods[name]
        order = sorted(range(len(labels)), key=lambda i: (values["rank"][i], labels[i]))[:shown]
        scores = np.asarray([values["score"][i] for i in order], dtype=float)
        peak = float(scores.max(initial=0.0))
        normalized = scores / peak if peak > 0 else scores
        positions = np.arange(len(order))
        bars = ax.barh(positions, normalized, color=color)
        ax.set_yticks(positions, labels=[labels[i] for i in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.24)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_xlabel("Score normalized within method")
        for bar, index in zip(bars, order):
            ax.text(max(bar.get_width(), 0.015) + 0.025, bar.get_y() + bar.get_height() / 2,
                    f"#{values['rank'][index]}  {values['score'][index]:.3g}",
                    va="center", fontsize=7)
    fig.suptitle(f"Word ranking comparison (Katz beta={beta:.4g})")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(input_path: Path, outdir: Path, top: int = 30, beta: float | None = None,
        hop: int = 4):
    if top < 0:
        raise ValueError("--top must be >= 0")
    if hop < 1:
        raise ValueError("--hop must be >= 1")
    labels, adjacency = read_boolean_matrix(input_path)
    methods, effective_beta = compare(labels, adjacency, beta, hop)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = input_path.stem
    csv_path = outdir / f"{prefix}_comparison.csv"
    png_path = outdir / f"{prefix}_comparison.png"
    write_scores(csv_path, labels, methods)
    plot_comparison(png_path, labels, methods, top, effective_beta, hop)
    print(f"Vertices: {len(labels)}; Boolean hops: {hop}; Katz beta: {effective_beta:.6g}")
    print(f"Scores: {csv_path}")
    print(f"Plot: {png_path}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Labeled Boolean adjacency-matrix CSV with a zero diagonal")
    parser.add_argument("--outdir", default="output/method_comparison")
    parser.add_argument("--top", type=int, default=30, help="Words shown per method; 0 plots all")
    parser.add_argument("--hop", type=int, default=4, help="Boolean linear-decay hop count (default: 4)")
    parser.add_argument("--beta", type=float, help="Katz attenuation; default is 0.85 / maximum row sum")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.top, args.beta, args.hop)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
