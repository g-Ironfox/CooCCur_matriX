#!/usr/bin/env python3
'''Select directed bigram candidates using association and boundary entropy.'''
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path

import phrase_metrics as metrics


FIELDS = (
    'phrase', 'left', 'right', 'count', 'paragraph_frequency',
    'left_entropy', 'right_entropy', 'joint_entropy', 'ppmi', 'npmi',
    'top_left_contexts', 'top_right_contexts', 'top_joint_contexts',
)


def select_candidates(words, counts, left_entropy, right_entropy, ppmi, npmi,
                      joint_entropy, phrases, min_count, min_ppmi,
                      min_npmi, max_entropy):
    '''Return pairs meeting support, association, and any low-entropy signal.'''
    indices = {word: index for index, word in enumerate(words)}
    selected = []
    for (left, right), stats in phrases.items():
        row, column = indices[left], indices[right]
        count = int(counts[row, column])
        score_ppmi, score_npmi = float(ppmi[row, column]), float(npmi[row, column])
        entropies = (float(left_entropy[row, column]), float(right_entropy[row, column]),
                     float(joint_entropy[row, column]))
        finite_entropies = [value for value in entropies if math.isfinite(value)]
        if (count < min_count or not math.isfinite(score_ppmi) or not math.isfinite(score_npmi)
                or score_ppmi < min_ppmi or score_npmi < min_npmi
                or not any(value <= max_entropy for value in finite_entropies)):
            continue

        def contexts(counter):
            return '; '.join(f'{word}:{frequency}' for word, frequency in counter.most_common(5))
        selected.append({
            'phrase': f'{left} {right}', 'left': left, 'right': right,
            'count': count, 'paragraph_frequency': len(stats.paragraphs),
            'left_entropy': entropies[0], 'right_entropy': entropies[1],
            'joint_entropy': entropies[2], 'ppmi': score_ppmi, 'npmi': score_npmi,
            'top_left_contexts': contexts(stats.left),
            'top_right_contexts': contexts(stats.right),
            'top_joint_contexts': '; '.join(
                f'({x},{y}):{n}' for (x, y), n in stats.joint.most_common(5)),
        })
    selected.sort(key=lambda row: (-row['npmi'], -row['ppmi'], -row['count'],
                                   row['left'], row['right']))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description='Select directed bigrams with strong association and low boundary entropy.')
    parser.add_argument('input', nargs='?', default='1.md', help='Source Markdown (default: 1.md)')
    parser.add_argument('--outdir', default='output/bigram_candidates', help='Output directory')
    parser.add_argument('--prefix', help='Output filename prefix (default: input filename stem)')
    parser.add_argument('--min-count', type=int, default=3,
                        help='Minimum occurrences after paragraph deduplication (default: 3)')
    parser.add_argument('--min-ppmi', type=float, default=0.5,
                        help='Minimum PPMI in natural-log units (default: 0.5)')
    parser.add_argument('--min-npmi', type=float, default=0.2,
                        help='Minimum NPMI in [-1, 1] (default: 0.2)')
    parser.add_argument('--max-entropy', type=float, default=0.5,
                        help='Maximum accepted left, right, or joint entropy in nats (default: 0.5)')
    parser.add_argument('--min-length', type=int, default=3,
                        help='Minimum token length; stopwords are excluded (default: 3)')
    args = parser.parse_args()
    if args.min_count < 1:
        parser.error('--min-count must be >= 1')
    if args.min_length < 1:
        parser.error('--min-length must be >= 1')
    if not math.isfinite(args.min_ppmi):
        parser.error('--min-ppmi must be finite')
    if not math.isfinite(args.min_npmi) or not -1 <= args.min_npmi <= 1:
        parser.error('--min-npmi must be finite and between -1 and 1')
    if not math.isfinite(args.max_entropy) or args.max_entropy < 0:
        parser.error('--max-entropy must be finite and >= 0')
    if args.prefix is not None and (Path(args.prefix).name != args.prefix or args.prefix in ('.', '..')):
        parser.error('--prefix must be a filename prefix, not a path')
    input_path = Path(args.input)
    try:
        markdown = input_path.read_text(encoding='utf-8-sig')
        words, counts, left, right, ppmi, npmi, joint, phrases = metrics.compute_metrics(
            markdown, args.min_length)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    candidates = select_candidates(words, counts, left, right, ppmi, npmi, joint, phrases,
                                   args.min_count, args.min_ppmi, args.min_npmi, args.max_entropy)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f'{args.prefix or input_path.stem}_bigram_candidates.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: (f'{row[key]:.8g}' if isinstance(row[key], float) else row[key])
                             for key in FIELDS})
    print(f'Directed bigram candidates: {len(candidates)}')
    print(f'Candidate report: {output}')


if __name__ == '__main__':
    main()
