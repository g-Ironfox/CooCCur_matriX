#!/usr/bin/env python3
"""Build directed and undirected adjacent-word matrices from Markdown."""
from __future__ import annotations
import argparse, csv, re
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, PowerNorm
import numpy as np

TERRACOTTA = LinearSegmentedColormap.from_list(
    "terracotta", ["#F5F4F3", "#E6D9D5", "#C99A8D", "#A85F50", "#6F332B"]
)

STOPWORDS = set("""a an the
and or but nor yet so
if unless whether
of to in on at by for from with as into onto upon about over under above below
between among through during before after since until while within without
is am are was were be been being
do does did done doing
have has had having
can could may might must shall should will would
i me my mine we us our ours you your yours he him his she her hers it its
they them their theirs who whom whose which what whatever whoever
this that these those here there
not no neither either
all any both each every few many much more most some several enough
such same other another
also just only even still already ever never always
very too quite rather almost
than then once again
per via
""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
SEGMENT_RE = re.compile(r"[\n.,]+")


def markdown_to_text(markdown: str) -> str:
    tick = chr(96)
    markdown = re.sub(tick * 3 + r".*?" + tick * 3, " ", markdown, flags=re.DOTALL)
    markdown = re.sub(tick + r"[^" + tick + r"]*" + tick, " ", markdown)
    markdown = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", markdown)
    markdown = re.sub(r"https?://\S+", " ", markdown)
    markdown = re.sub(r"^\s*\|.*\|\s*$", " ", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"^\s*[-*_]{3,}\s*$", " ", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"^\s{0,3}#{1,6}\s*", "", markdown, flags=re.MULTILINE)
    return re.sub(r"[*_~]", "", markdown)


def unique_paragraphs(text: str) -> list[str]:
    seen, result = set(), []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = re.sub(r"\s+", " ", paragraph).strip().casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(paragraph)
    return result


def tokenize_segments(paragraphs: list[str], min_length: int) -> list[list[str]]:
    segments = []
    for paragraph in paragraphs:
        for raw_sentence in SEGMENT_RE.split(paragraph):
            words = [w.casefold() for w in WORD_RE.findall(raw_sentence)
                     if len(w) >= min_length and w.casefold() not in STOPWORDS]
            if words:
                segments.append(words)
    return segments


def build_matrices(segments: list[list[str]], min_frequency: int):
    frequencies, directed_counts, undirected_counts = Counter(), Counter(), Counter()
    for segment in segments:
        frequencies.update(set(segment))
        for left, right in zip(segment, segment[1:]):
            if left != right:
                directed_counts[(left, right)] += 1
                undirected_counts[tuple(sorted((left, right)))] += 1

    terms = [word for word, count in frequencies.most_common() if count >= min_frequency]
    terms.sort()
    indices = {word: i for i, word in enumerate(terms)}

    def make_matrix(edge_counts, symmetric=False):
        matrix = np.zeros((len(terms), len(terms)), dtype=np.int64)
        for (left, right), weight in edge_counts.items():
            if left in indices and right in indices:
                i, j = indices[left], indices[right]
                matrix[i, j] = weight
                if symmetric:
                    matrix[j, i] = weight
        return matrix

    return terms, make_matrix(directed_counts), make_matrix(undirected_counts, symmetric=True), frequencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate six directed/undirected adjacent-word matrices and heatmaps.")
    parser.add_argument("input", nargs="?", default="1.md", help="Input Markdown file (default: 1.md)")
    parser.add_argument("--outdir", default="output", help="Output directory (default: output)")
    parser.add_argument("--plot-top-words", type=int, default=40, help="Most frequent N words to plot; use 0 for all (default: 40)")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum number of segments containing a word (default: 2)")
    parser.add_argument("--min-length", type=int, default=3, help="Minimum token length (default: 3)")
    parser.add_argument("--figure-size", type=float, help="Figure width in inches for all six heatmaps")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}")
    if args.plot_top_words < 0 or args.min_frequency < 1 or args.min_length < 1:
        parser.error("--plot-top-words must be >= 0; frequency and length must be >= 1")
    if args.figure_size is not None and args.figure_size <= 0:
        parser.error("--figure-size must be greater than 0 inches")
    raw = input_path.read_text(encoding="utf-8-sig")
    segments = tokenize_segments(unique_paragraphs(markdown_to_text(raw)), args.min_length)
    terms, directed, undirected, frequencies = build_matrices(segments, args.min_frequency)
    if not terms:
        parser.error("no words remain; try lowering --min-frequency or --min-length")

    plot_terms = [word for word, count in frequencies.most_common()
                  if count >= args.min_frequency]
    if args.plot_top_words > 0:
        plot_terms = plot_terms[:args.plot_top_words]
    plot_terms.sort()
    indices = {word: i for i, word in enumerate(terms)}
    plot_indices = [indices[word] for word in plot_terms]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    default_size = max(10, min(28, len(plot_terms) * 0.55))
    variants = (
        ("directed_diag_count", directed, True, False, "Directed, diagonal = word count", args.figure_size),
        ("directed_diag_zero", directed, False, False, "Directed, diagonal = 0", args.figure_size),
        ("undirected_diag_count", undirected, True, False, "Undirected, diagonal = word count", args.figure_size),
        ("undirected_diag_zero", undirected, False, False, "Undirected, diagonal = 0", args.figure_size),
        ("directed_bool", directed, False, True, "Directed Boolean adjacency", args.figure_size),
        ("undirected_bool", undirected, False, True, "Undirected Boolean adjacency", args.figure_size),
    )
    for name, base_matrix, keep_diagonal_count, boolean, title, requested_size in variants:
        size = requested_size if requested_size is not None else default_size
        matrix = (base_matrix > 0).astype(np.uint8) if boolean else base_matrix.copy()
        if keep_diagonal_count:
            for i, word in enumerate(terms):
                matrix[i, i] = frequencies[word]

        matrix_path = outdir / f"{name}.csv"
        heatmap_path = outdir / f"{name}_heatmap.png"
        with matrix_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["word", *terms])
            for word, row in zip(terms, matrix):
                writer.writerow([word, *row.tolist()])

        plot_matrix = matrix[np.ix_(plot_indices, plot_indices)]
        fig, ax = plt.subplots(figsize=(size, size * 0.88), constrained_layout=True)
        if boolean:
            cmap = ListedColormap([TERRACOTTA(0.0), TERRACOTTA(1.0)])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            image = ax.imshow(plot_matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        else:
            max_value = max(1, int(plot_matrix.max()))
            norm = PowerNorm(gamma=0.55, vmin=0, vmax=max_value)
            image = ax.imshow(plot_matrix, cmap=TERRACOTTA, norm=norm, interpolation="nearest", aspect="auto")
        ax.set_xticks(range(len(plot_terms)), labels=plot_terms, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(range(len(plot_terms)), labels=plot_terms, fontsize=8)
        ax.set_title(f"{title} ({len(segments)} segments; {len(plot_terms)} words)")
        ax.set_xlabel("Next word" if name.startswith("directed") else "Word")
        ax.set_ylabel("Previous word" if name.startswith("directed") else "Word")
        colorbar_label = "Edge present (1) / absent (0)" if boolean else "Adjacent-pair count; diagonal is segment frequency when retained"
        fig.colorbar(image, ax=ax, label=colorbar_label, shrink=0.8, ticks=[0, 1] if boolean else None)
        fig.savefig(heatmap_path, dpi=220)
        plt.close(fig)
        print(f"{name}: {matrix_path} | {heatmap_path}")

    print(f"Processed segments: {len(segments)}")
    print(f"Matrix words: {len(terms)}")
    print(f"Plotted words: {len(plot_terms)}")


if __name__ == "__main__":
    main()
