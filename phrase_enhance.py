#!/usr/bin/env python3
"""Mine adjacent bigrams in Markdown, merge occurrences, and rebuild matrices."""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import cooccurrence as co
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm


@dataclass(frozen=True)
class Candidate:
    left: str
    right: str
    count: int
    npmi: float

    @property
    def label(self) -> str:
        return f"{self.left} {self.right}"


def raw_segments(markdown: str) -> list[list[str]]:
    """Keep stopwords until after mining so phrase matches cannot bridge them."""
    paragraphs = co.unique_paragraphs(co.markdown_to_text(markdown))
    return [words for paragraph in paragraphs
            for part in co.SEGMENT_RE.split(paragraph)
            if (words := [word.casefold() for word in co.WORD_RE.findall(part)])]


def eligible(word: str, min_length: int) -> bool:
    return len(word) >= min_length and word not in co.STOPWORDS


def mine_phrases(segments: list[list[str]], min_frequency: int,
                 min_npmi: float, min_length: int):
    """Score raw directed adjacency events, then filter eligible content pairs."""
    pairs = Counter(pair for segment in segments
                    for pair in zip(segment, segment[1:]))
    left_totals, right_totals = Counter(), Counter()
    for (left, right), count in pairs.items():
        left_totals[left] += count
        right_totals[right] += count
    total = sum(pairs.values())
    candidates = []
    for (left, right), count in pairs.items():
        if (count < min_frequency or left == right
                or not eligible(left, min_length) or not eligible(right, min_length)):
            continue
        probability = count / total
        pmi = math.log(count) + math.log(total) - math.log(left_totals[left]) - math.log(right_totals[right])
        score = pmi / -math.log(probability) if probability < 1 else 1.0
        candidates.append(Candidate(left, right, count, max(-1.0, min(1.0, score))))
    candidates.sort(key=lambda item: (-item.npmi, -item.count, item.left, item.right))
    selected = [candidate for candidate in candidates if candidate.npmi >= min_npmi]
    return candidates, selected


def merge_phrases(segments: list[list[str]], selected: list[Candidate], min_length: int):
    """Merge non-overlapping original bigrams left to right, then filter words."""
    labels = {(item.left, item.right): item.label for item in selected}
    merged_segments, replacements = [], Counter()
    for segment in segments:
        merged, index = [], 0
        while index < len(segment):
            label = labels.get(tuple(segment[index:index + 2]))
            if label is not None:
                merged.append(label)
                replacements[label] += 1
                index += 2
            else:
                if eligible(segment[index], min_length):
                    merged.append(segment[index])
                index += 1
        if merged:
            merged_segments.append(merged)
    return merged_segments, replacements


def write_matrix(path: Path, terms: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["word", *terms])
        for term, row in zip(terms, matrix):
            writer.writerow([term, *row.tolist()])


def draw_heatmap(path: Path, terms, matrix, name: str, boolean: bool,
                 total_terms: int, figure_size: float | None) -> None:
    size = figure_size or max(10, min(28, len(terms) * 0.55))
    fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
    try:
        if boolean:
            cmap = ListedColormap([co.TERRACOTTA(0.0), co.TERRACOTTA(1.0)])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
        else:
            cmap = co.TERRACOTTA
            norm = PowerNorm(gamma=0.55, vmin=0, vmax=max(1, int(matrix.max())))
        image = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(terms)), labels=terms, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(terms)), labels=terms, fontsize=8)
        for label, term in zip(ax.get_xticklabels(), terms):
            if " " in term:
                label.set_color("#6F332B")
                label.set_fontweight("bold")
        for label, term in zip(ax.get_yticklabels(), terms):
            if " " in term:
                label.set_color("#6F332B")
                label.set_fontweight("bold")
        kind = "Directed" if name.startswith("directed") else "Undirected"
        variant = ("Boolean adjacency" if boolean else
                   "diagonal = segment frequency" if name.endswith("diag_count") else "diagonal = 0")
        note = (f"top {len(terms)} of {total_terms} by segment frequency"
                if len(terms) < total_terms else f"all {total_terms} terms")
        ax.set_title(f"Phrase-enhanced {kind.lower()}: {variant}\n{note}; phrases in bold brown")
        ax.set_xlabel("Next word / phrase" if kind == "Directed" else "Word / phrase")
        ax.set_ylabel("Previous word / phrase" if kind == "Directed" else "Word / phrase")
        scale = ("Edge present (1) / absent (0)" if boolean else
                 "Adjacent-pair count; diagonal = segment frequency" if name.endswith("diag_count")
                 else "Adjacent-pair count")
        fig.colorbar(image, ax=ax, label=scale, shrink=0.8, ticks=[0, 1] if boolean else None)
        fig.savefig(path, dpi=220)
    finally:
        plt.close(fig)


def write_outputs(outdir: Path, terms, directed, undirected, frequencies,
                  candidates, selected, replacements, plot_top_words: int = 40,
                  figure_size: float | None = None, no_plots: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(terms)))
    if plot_top_words and plot_top_words < len(terms):
        ranked = sorted(indices, key=lambda index: (-frequencies[terms[index]], terms[index]))
        indices = sorted(ranked[:plot_top_words])
    plot_terms = [terms[index] for index in indices]
    for name, base, diagonal, boolean in (
        ("directed_diag_count", directed, True, False),
        ("directed_diag_zero", directed, False, False),
        ("undirected_diag_count", undirected, True, False),
        ("undirected_diag_zero", undirected, False, False),
        ("directed_bool", directed, False, True),
        ("undirected_bool", undirected, False, True),
    ):
        matrix = (base > 0).astype(np.uint8) if boolean else base.copy()
        if diagonal:
            np.fill_diagonal(matrix, [frequencies[term] for term in terms])
        write_matrix(outdir / f"phrase_enhanced_{name}.csv", terms, matrix)
        if not no_plots:
            draw_heatmap(outdir / f"phrase_enhanced_{name}_heatmap.png", plot_terms,
                         matrix[np.ix_(indices, indices)], name, boolean, len(terms), figure_size)
    chosen = {item.label for item in selected}
    retained = set(terms)
    with (outdir / "phrase_candidates.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["phrase", "left", "right", "count", "npmi", "selected",
                         "merged_count", "segment_frequency", "included_in_matrix"])
        for item in candidates:
            writer.writerow([item.label, item.left, item.right, item.count,
                             f"{item.npmi:.8g}", item.label in chosen,
                             replacements[item.label], frequencies[item.label],
                             item.label in retained])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine adjacent bigram phrases from Markdown and rebuild six count/Boolean matrices."
    )
    parser.add_argument("input", nargs="?", default="1.md", help="Input Markdown file (default: 1.md)")
    parser.add_argument("--outdir", default="output/phrase", help="Output directory (default: output/phrase)")
    parser.add_argument("--min-phrase-frequency", type=int, default=3,
                        help="Minimum raw adjacent-pair count (default: 3)")
    parser.add_argument("--min-npmi", type=float, default=0.2,
                        help="Minimum pair NPMI in [-1, 1] (default: 0.2)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum segment frequency after merging (default: 2)")
    parser.add_argument("--top-words", type=int, default=40,
                        help="Keep top N terms after merging; 0 keeps all (default: 40)")
    parser.add_argument("--min-length", type=int, default=3,
                        help="Minimum component/word length (default: 3)")
    parser.add_argument("--plot-top-words", type=int, default=40,
                        help="Plot top N terms by merged segment frequency; 0 plots all (default: 40)")
    parser.add_argument("--figure-size", type=float,
                        help="Heatmap width in inches (default: based on displayed term count)")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV outputs only")
    args = parser.parse_args()
    if args.min_phrase_frequency < 1 or args.min_frequency < 1 or args.min_length < 1:
        parser.error("frequency and length arguments must be >= 1")
    if args.top_words < 0:
        parser.error("--top-words must be >= 0")
    if args.plot_top_words < 0:
        parser.error("--plot-top-words must be >= 0")
    if args.figure_size is not None and (not math.isfinite(args.figure_size) or args.figure_size <= 0):
        parser.error("--figure-size must be finite and greater than 0")
    if not math.isfinite(args.min_npmi) or not -1 <= args.min_npmi <= 1:
        parser.error("--min-npmi must be finite and between -1 and 1")
    input_path = Path(args.input)
    if input_path.suffix.casefold() not in {".md", ".markdown"}:
        parser.error("input must be a Markdown file (.md or .markdown)")
    try:
        segments = raw_segments(input_path.read_text(encoding="utf-8-sig"))
        candidates, selected = mine_phrases(
            segments, args.min_phrase_frequency, args.min_npmi, args.min_length
        )
        merged, replacements = merge_phrases(segments, selected, args.min_length)
        terms, directed, undirected, frequencies = co.build_matrices(
            merged, args.min_frequency, args.top_words
        )
        if not terms:
            parser.error("no terms remain; lower --min-frequency or --min-length")
        write_outputs(Path(args.outdir), terms, directed, undirected, frequencies,
                      candidates, selected, replacements, args.plot_top_words,
                      args.figure_size, args.no_plots)
    except (OSError, UnicodeError) as error:
        parser.error(str(error))
    print(f"Processed segments: {len(segments)}")
    print(f"Selected phrase candidates: {len(selected)}")
    print(f"Merged phrase occurrences: {sum(replacements.values())}")
    print(f"Retained terms: {len(terms)} ({sum(' ' in term for term in terms)} phrases)")
    if not selected:
        print("No phrases passed thresholds; wrote ordinary word matrices.")
    print(f"Matrices: {Path(args.outdir) / 'phrase_enhanced_*.csv'}")
    if not args.no_plots:
        print(f"Heatmaps: {Path(args.outdir) / 'phrase_enhanced_*_heatmap.png'}")
    print(f"Candidate report: {Path(args.outdir) / 'phrase_candidates.csv'}")


if __name__ == "__main__":
    main()
