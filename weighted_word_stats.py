#!/usr/bin/env python3
"""Summarize weighted hop matrices by word without loading the whole matrix."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Compatibility entry point; the implementation lives in bool_hop_stability.
from bool_hop_stability import (
    WORD_STATS_FIELDS as FIELDS,
    summarize_weighted_words as summarize,
    plot_word_stats as plot_stats,
    write_word_stats as run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="output/bool_hop_stability/weighted_012.csv",
                        help="Weighted hop matrix CSV (default: output/bool_hop_stability/weighted_012.csv)")
    parser.add_argument("--outdir", default="output/weighted_word_stats", help="Output directory")
    parser.add_argument("--top", type=int, default=30,
                        help="Words to plot; 0 plots all, at most 40 per image (default: 30)")
    parser.add_argument("--sort-by", choices=["incoming", "outgoing", "min"], default="incoming",
                        help="Rank by column sum, row sum or their minimum (default: incoming)")
    parser.add_argument("--include-diagonal", action="store_true",
                        help="Include self-return scores in sums and reachable counts")
    args = parser.parse_args()
    try:
        run(Path(args.input), Path(args.outdir), args.top, args.include_diagonal, args.sort_by)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
