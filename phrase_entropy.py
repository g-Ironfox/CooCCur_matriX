#!/usr/bin/env python3
"""Measure exact left, right, and joint boundary entropy of directed bigrams in Markdown."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cooccurrence as co
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

Pair = tuple[str, str]


@dataclass
class PhraseContexts:
    count: int = 0
    paragraphs: set[int] = field(default_factory=set)
    left: Counter[str] = field(default_factory=Counter)
    right: Counter[str] = field(default_factory=Counter)
    joint: Counter[tuple[str, str]] = field(default_factory=Counter)


def entropy(contexts: Counter) -> float:
    """Shannon entropy in nats; no observed boundary is missing, not zero."""
    total = sum(contexts.values())
    if not total:
        return math.nan
    return -sum((count / total) * math.log(count / total)
                for count in contexts.values()) + 0.0


def read_candidates(path: Path) -> set[Pair]:
    """Use all candidate rows, including unselected ones, as an ordered allowlist."""
    pairs = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"left", "right"}.issubset(reader.fieldnames or []):
            raise ValueError("candidate CSV must contain left and right columns")
        for row_number, row in enumerate(reader, 2):
            pair = tuple((row.get(side) or "").strip().casefold()
                         for side in ("left", "right"))
            if not all(co.WORD_RE.fullmatch(word) for word in pair):
                raise ValueError(f"candidate row {row_number}: left/right must be single English tokens")
            pairs.add(pair)
    if not pairs:
        raise ValueError("candidate CSV has no phrases")
    return pairs


def collect_contexts(markdown: str, min_length: int = 3,
                     candidates: set[Pair] | None = None) -> dict[Pair, PhraseContexts]:
    """Count raw contiguous AB occurrences and their immediate X/Y in X A B Y.

    Preserve stopwords and short tokens as boundaries; never bridge a filtered
    token or a segment boundary. Overlapping occurrences count independently.
    """
    results = {}
    paragraphs = co.unique_paragraphs(co.markdown_to_text(markdown))
    for paragraph_id, paragraph in enumerate(paragraphs):
        for part in co.SEGMENT_RE.split(paragraph):
            words = [word.casefold() for word in co.WORD_RE.findall(part)]
            for index, pair in enumerate(zip(words, words[1:])):
                if (pair[0] == pair[1]
                        or any(len(word) < min_length or word in co.STOPWORDS for word in pair)
                        or (candidates is not None and pair not in candidates)):
                    continue
                stats = results.setdefault(pair, PhraseContexts())
                stats.count += 1
                stats.paragraphs.add(paragraph_id)
                if index > 0:
                    stats.left[words[index - 1]] += 1
                right_context = words[index + 2] if index + 2 < len(words) else "<EOS>"
                left_context = words[index - 1] if index > 0 else "<BOS>"
                stats.joint[(left_context, right_context)] += 1
                if index + 2 < len(words):
                    stats.right[words[index + 2]] += 1
    return results


def build_matrices(phrases: dict[Pair, PhraseContexts]):
    words = sorted({word for pair in phrases for word in pair})
    indices = {word: index for index, word in enumerate(words)}
    counts = np.zeros((len(words), len(words)), dtype=np.int64)
    left = np.full(counts.shape, np.nan)
    right = np.full(counts.shape, np.nan)
    joint = np.full(counts.shape, np.nan)
    for pair, stats in phrases.items():
        at = (indices[pair[0]], indices[pair[1]])
        counts[at] = stats.count
        left[at], right[at], joint[at] = (
            entropy(stats.left), entropy(stats.right), entropy(stats.joint))
    return words, counts, left, right, joint


def number(value: float) -> str:
    return f"{value:.8g}" if math.isfinite(value) else ""


def write_matrix(path: Path, words: list[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["word", *words])
        for word, row in zip(words, matrix):
            writer.writerow([word, *map(number, row)])


def write_report(path: Path, phrases: dict[Pair, PhraseContexts]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["phrase", "left", "right", "count", "paragraph_frequency",
                         "left_context_total", "left_context_types", "left_missing_count",
                         "left_entropy", "top_left_contexts",
                         "right_context_total", "right_context_types", "right_missing_count",
                         "right_entropy", "top_right_contexts",
                         "joint_context_types", "joint_entropy", "top_joint_contexts"])
        for (left, right), stats in sorted(phrases.items(), key=lambda item: (-item[1].count, item[0])):
            row = [f"{left} {right}", left, right, stats.count, len(stats.paragraphs)]
            for contexts in (stats.left, stats.right):
                total = sum(contexts.values())
                top = "; ".join(f"{word}:{count}" for word, count in contexts.most_common(5))
                row.extend([total, len(contexts), stats.count - total, number(entropy(contexts)), top])
            joint_top = "; ".join(
                f"({left_word},{right_word}):{count}"
                for (left_word, right_word), count in stats.joint.most_common(5))
            row.extend([len(stats.joint), number(entropy(stats.joint)), joint_top])
            writer.writerow(row)


def plot_entropy(path: Path, words: list[str], values: np.ndarray,
                 side: str, figure_size: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(figure_size, figure_size * 0.88), constrained_layout=True)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(words)), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)), labels=words, fontsize=8)
    ax.set_xlabel("Second word B (ordered phrase A -> B)")
    ax.set_ylabel("First word A")
    boundary = {"left": "X | A B", "right": "Y | A B",
                "joint": "(X, Y) | A B"}[side]
    missing_note = ("Grey: no qualifying phrase" if side == "joint"
                    else "Grey: no qualifying phrase or no observed boundary context")
    ax.set_title(f"Directed phrase {side} boundary entropy H({boundary})\n"
                 + missing_note, fontsize=11)
    fig.colorbar(image, ax=ax, label="Entropy (natural log, nats)", shrink=0.8)
    fig.savefig(path, dpi=220)
    plt.close(fig)



def plot_left_right_heatmap(path: Path, words: list[str], left: np.ndarray,
                           right: np.ndarray, figure_size: float, vmax: float) -> None:
    """Draw both directional entropies as diagonal halves of each matrix cell."""
    cmap = plt.get_cmap("YlOrRd").copy()
    norm = Normalize(vmin=0, vmax=vmax)
    vertices, colors = [], []
    for row in range(len(words)):
        for column in range(len(words)):
            x0, x1 = column - 0.5, column + 0.5
            y0, y1 = row - 0.5, row + 0.5
            vertices.extend(([(x0, y0), (x1, y0), (x0, y1)],
                             [(x1, y0), (x1, y1), (x0, y1)]))
            for matrix in (left, right):
                value = matrix[row, column]
                colors.append(cmap(norm(value)) if np.isfinite(value) else "#d9d9d9")

    fig, ax = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
    collection = PolyCollection(vertices, facecolors=colors, edgecolors="white",
                                linewidths=0.25, antialiaseds=False)
    ax.add_collection(collection)
    ax.set_xlim(-0.5, len(words) - 0.5)
    ax.set_ylim(len(words) - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(words)), labels=words, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(words)), labels=words, fontsize=8)
    ax.set_xlabel("Second word B (ordered phrase A -> B)")
    ax.set_ylabel("First word A")
    ax.set_title("Directed phrase boundary entropy\n"
                 "Upper-left triangle: left; lower-right triangle: right", fontsize=11)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="Entropy (natural log, nats)", shrink=0.8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_heatmaps(outdir: Path, prefix: str, words: list[str], counts: np.ndarray,
                  left: np.ndarray, right: np.ndarray, joint: np.ndarray, top_words: int,
                  figure_size: float | None = None) -> dict[str, Path]:
    # Rank for display only: entropy always uses every original context.
    indices = np.arange(len(words))
    if top_words and top_words < len(words):
        strength = counts.sum(axis=0) + counts.sum(axis=1)
        indices = np.argsort(-strength, kind="stable")[:top_words]
    plot_words = [words[index] for index in indices]
    finite = np.concatenate([matrix[np.isfinite(matrix)]
                             for matrix in (left, right, joint)])
    maximum = float(finite.max()) if finite.size else 0.0
    vmax = maximum if maximum > 0 else 1.0
    size = figure_size or max(10, min(28, len(plot_words) * 0.55))
    paths = {
        "left": outdir / f"{prefix}_left_entropy_heatmap.png",
        "right": outdir / f"{prefix}_right_entropy_heatmap.png",
        "left/right": outdir / f"{prefix}_left_right_entropy_heatmap.png",
        "joint": outdir / f"{prefix}_joint_entropy_heatmap.png",
    }
    subset = np.ix_(indices, indices)
    plot_entropy(paths["left"], plot_words, left[subset], "left", size, vmax)
    plot_entropy(paths["right"], plot_words, right[subset], "right", size, vmax)
    plot_left_right_heatmap(paths["left/right"], plot_words,
                            left[subset], right[subset], size, vmax)
    plot_entropy(paths["joint"], plot_words, joint[subset], "joint", size, vmax)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Markdown and plot exact left, right, and joint phrase entropy for directed bigrams only.")
    parser.add_argument("input", nargs="?", default="1.md", help="Source Markdown (default: 1.md)")
    parser.add_argument("--outdir", default="output/phrase_entropy", help="Output directory")
    parser.add_argument("--prefix", help="Output filename prefix (default: input filename stem)")
    parser.add_argument("--phrase-candidates", help="Optional phrase_candidates.csv; analyze all listed ordered pairs")
    parser.add_argument("--min-phrase-frequency", type=int, default=3,
                        help="Minimum raw occurrence count after paragraph deduplication (default: 3)")
    parser.add_argument("--min-length", type=int, default=3, help="Minimum phrase token length (default: 3)")
    parser.add_argument("--plot-top-words", type=int, default=40,
                        help="Display top N words by qualifying phrase counts; 0 for all (default: 40)")
    parser.add_argument("--figure-size", type=float, help="Figure width in inches")
    args = parser.parse_args()
    if args.min_phrase_frequency < 1 or args.min_length < 1:
        parser.error("--min-phrase-frequency and --min-length must be >= 1")
    if args.plot_top_words < 0:
        parser.error("--plot-top-words must be >= 0")
    if args.figure_size is not None and (not math.isfinite(args.figure_size) or args.figure_size <= 0):
        parser.error("--figure-size must be finite and greater than 0")
    if args.prefix is not None and (not args.prefix.strip() or args.prefix in (".", "..")
                                   or any(char in args.prefix for char in '/\\:<>"|?*')):
        parser.error("--prefix must be a filename prefix, not a path")
    input_path = Path(args.input)
    try:
        candidates = read_candidates(Path(args.phrase_candidates)) if args.phrase_candidates else None
        phrases = collect_contexts(input_path.read_text(encoding="utf-8-sig"), args.min_length, candidates)
        phrases = {pair: stats for pair, stats in phrases.items() if stats.count >= args.min_phrase_frequency}
        if not phrases:
            raise ValueError("no qualifying phrases; lower --min-phrase-frequency/--min-length or check candidates")
        words, counts, left, right, joint = build_matrices(phrases)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or input_path.stem
    report = outdir / f"{prefix}_phrase_boundaries.csv"
    write_report(report, phrases)
    for name, matrix in (("phrase_counts", counts), ("left_entropy", left),
                         ("right_entropy", right), ("joint_entropy", joint)):
        path = outdir / f"{prefix}_{name}.csv"
        write_matrix(path, words, matrix)
        print(f"{name}: {path}")
    paths = plot_heatmaps(outdir, prefix, words, counts, left, right, joint,
                          args.plot_top_words, args.figure_size)
    print(f"Directed phrases: {len(phrases)}; words: {len(words)}")
    print(f"Phrase boundary statistics: {report}")
    for side, path in paths.items():
        print(f"{side.capitalize()} entropy heatmap: {path}")


if __name__ == "__main__":
    main()
