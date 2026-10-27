#!/usr/bin/env python3
'''Score and visualize left/right trigram extensions of selected bigrams.'''
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cooccurrence as co
import phrase_entropy as entropy


FIELDS = (
    'trigram', 'base_bigram', 'extension_side', 'extension_word',
    'trigram_count', 'paragraph_frequency', 'base_bigram_count',
    'base_boundary_entropy', 'left_entropy', 'right_entropy', 'joint_entropy',
    'pmi', 'ppmi', 'npmi',
)


def collect_extensions(markdown, candidates, min_length=3):
    '''Count observed triples and directional marginals in deduplicated segments.'''
    token_ok = lambda word: len(word) >= min_length and word not in co.STOPWORDS
    triples = Counter()
    paragraphs_by_triple = defaultdict(set)
    base_phrases = entropy.collect_contexts(markdown, min_length, candidates)
    paragraphs = co.unique_paragraphs(co.markdown_to_text(markdown))
    for paragraph_id, paragraph in enumerate(paragraphs):
        for part in co.SEGMENT_RE.split(paragraph):
            words = [word.casefold() for word in co.WORD_RE.findall(part)]
            for triple in zip(words, words[1:], words[2:]):
                if not all(token_ok(word) for word in triple):
                    continue
                triples[triple] += 1
                paragraphs_by_triple[triple].add(paragraph_id)

    prefix_totals, final_totals = Counter(), Counter()
    initial_totals, suffix_totals = Counter(), Counter()
    for (first, middle, last), count in triples.items():
        prefix_totals[(first, middle)] += count
        final_totals[last] += count
        initial_totals[first] += count
        suffix_totals[(middle, last)] += count

    total = sum(triples.values())
    results = []
    for triple, count in triples.items():
        first, middle, last = triple
        right_base = (first, middle)
        left_base = (middle, last)
        if right_base in candidates:
            results.append((right_base, 'right', last, triple, count,
                            prefix_totals[right_base], final_totals[last]))
        if left_base in candidates:
            results.append((left_base, 'left', first, triple, count,
                            initial_totals[first], suffix_totals[left_base]))

    rows = []
    for base, side, extension, triple, count, row_total, column_total in results:
        pmi = math.log(count * total / (row_total * column_total))
        ppmi = max(0.0, pmi)
        probability = count / total
        npmi = pmi / -math.log(probability) if probability < 1 else 1.0
        stats = base_phrases.get(base)
        boundary_entropy = (entropy.entropy(stats.right if side == 'right' else stats.left)
                            if stats else math.nan)
        rows.append({
            'trigram': ' '.join(triple),
            'base_bigram': ' '.join(base),
            'extension_side': side,
            'extension_word': extension,
            'trigram_count': count,
            'paragraph_frequency': len(paragraphs_by_triple[triple]),
            'base_bigram_count': stats.count if stats else 0,
            'base_boundary_entropy': boundary_entropy,
            'left_entropy': entropy.entropy(stats.left) if stats else math.nan,
            'right_entropy': entropy.entropy(stats.right) if stats else math.nan,
            'joint_entropy': entropy.entropy(stats.joint) if stats else math.nan,
            'pmi': pmi,
            'ppmi': ppmi,
            'npmi': max(-1.0, min(1.0, npmi)),
        })
    rows.sort(key=lambda row: (-row['npmi'], -row['trigram_count'],
                               row['base_bigram'], row['extension_side'], row['trigram']))
    return rows


def write_candidates(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f'{row[key]:.8g}' if isinstance(row[key], float)
                                   and math.isfinite(row[key]) else
                                   '' if isinstance(row[key], float) else row[key])
                             for key in FIELDS})


def plot_potential(path, rows, top):
    by_trigram = {}
    for row in rows:
        current = by_trigram.get(row['trigram'])
        rank = (row['npmi'], row['trigram_count'], row['base_bigram'])
        current_rank = ((current['npmi'], current['trigram_count'], current['base_bigram'])
                        if current else None)
        if current is None or rank > current_rank:
            by_trigram[row['trigram']] = row
    ranked = sorted(by_trigram.values(), key=lambda row: (
        -row['trigram_count'], -row['npmi'], row['trigram']))
    shown = ranked[:top] if top else ranked
    height = max(5, 1.8 + len(shown) * 0.34)
    fig, axes = plt.subplots(
        1, 2, figsize=(14, height), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [1, 1.35]})
    if not shown:
        axes[0].text(0.5, 0.5, 'No qualifying trigram extensions',
                     ha='center', va='center', transform=axes[0].transAxes)
        for ax in axes:
            ax.set_axis_off()
    else:
        norm = plt.Normalize(-1, 1)
        cmap = plt.get_cmap('coolwarm')
        labels = [row['trigram'] for row in shown]
        positions = list(range(len(shown)))
        count_values = [row['trigram_count'] for row in shown]
        count_bars = axes[0].barh(
            positions, count_values,
            color=[cmap(norm(row['npmi'])) for row in shown],
            edgecolor='white', linewidth=0.6)
        axes[0].bar_label(count_bars, fmt='%.0f', padding=3, fontsize=7)
        axes[0].set_title('Frequency')
        axes[0].set_xlabel('Count')
        axes[0].set_xlim(0, max(count_values, default=0) * 1.2 or 1)

        bar_height = 0.34
        offsets = [-bar_height / 2, bar_height / 2]
        entropy_series = (
            ('left_entropy', 'Left entropy', '#2F6F73', offsets[0]),
            ('right_entropy', 'Right entropy', '#D17A32', offsets[1]),
        )
        max_entropy = max((row[field] for row in shown for field, _, _, _ in entropy_series
                           if math.isfinite(row[field])), default=0)
        for field, legend_label, color, offset in entropy_series:
            values = [row[field] if math.isfinite(row[field]) else 0 for row in shown]
            bars = axes[1].barh(
                [position + offset for position in positions], values,
                height=bar_height, color=color, edgecolor='white',
                linewidth=0.6, label=legend_label)
            for bar, row, value in zip(bars, shown, values):
                if not math.isfinite(row[field]) or value == 0:
                    continue
                axes[1].annotate(
                    f'{value:.2f}',
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0), textcoords='offset points',
                    va='center', ha='left', fontsize=7)
        axes[1].set_title('Boundary entropy')
        axes[1].set_xlabel('Entropy (nat)')
        axes[1].set_xlim(0, max_entropy * 1.25 or 1)
        axes[1].legend(frameon=False, loc='lower right')
        for ax in axes:
            ax.grid(axis='x', color='#D3CDCA', linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
        axes[0].set_yticks(positions, labels)
        axes[0].invert_yaxis()
        axes[0].set_ylabel('Trigram')
        colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0],
            pad=0.025, fraction=0.045)
        colorbar.set_label('NPMI')
    fig.suptitle('Trigram extensions: frequency and boundary entropy')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Score and visualize observed three-word extensions of selected bigrams.')
    parser.add_argument('input', nargs='?', default='1.md', help='Source Markdown (default: 1.md)')
    parser.add_argument('--phrase-candidates', required=True,
                        help='Candidate CSV with left and right columns, e.g. select_bigrams output')
    parser.add_argument('--outdir', default='output/trigram_potential', help='Output directory')
    parser.add_argument('--prefix', help='Output filename prefix (default: input filename stem)')
    parser.add_argument('--min-trigram-count', type=int, default=2,
                        help='Minimum triple occurrences (default: 2)')
    parser.add_argument('--min-length', type=int, default=3,
                        help='Minimum token length; stopwords are excluded (default: 3)')
    parser.add_argument('--top', type=int, default=30,
                        help='Maximum plotted candidates by trigram frequency; 0 plots all (default: 30)')
    args = parser.parse_args()
    if args.min_trigram_count < 1 or args.min_length < 1 or args.top < 0:
        parser.error('--min-trigram-count and --min-length must be >= 1; --top must be >= 0')
    try:
        markdown = Path(args.input).read_text(encoding='utf-8-sig')
        candidates = entropy.read_candidates(Path(args.phrase_candidates))
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    all_rows = collect_extensions(markdown, candidates, args.min_length)
    rows = [row for row in all_rows if row['trigram_count'] >= args.min_trigram_count]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.input).stem
    csv_path = outdir / f'{prefix}_trigram_potential.csv'
    png_path = outdir / f'{prefix}_trigram_potential.png'
    write_candidates(csv_path, rows)
    plot_potential(png_path, rows, args.top)
    print(f'Trigram extensions: {len(rows)}')
    print(f'Candidate report: {csv_path}')
    print(f'Potential plot: {png_path}')


if __name__ == '__main__':
    main()
