'''Tests for association-and-entropy bigram candidate selection.'''
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import phrase_entropy
import select_bigrams


class SelectBigramsTests(unittest.TestCase):
    def test_cli_selects_candidates_and_writes_boundary_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.md'
            output = root / 'out'
            source.write_text(
                'outerone alpha beta gamma.\n\n'
                'outertwo alpha beta gamma.\n\n'
                'outerthree alpha beta gamma.', encoding='utf-8')
            result = subprocess.run(
                [sys.executable, str(Path(select_bigrams.__file__).resolve()), str(source),
                 '--outdir', str(output), '--min-count', '3', '--min-ppmi', '0',
                 '--min-npmi', '-1', '--max-entropy', '0'],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Directed bigram candidates: 2', result.stdout)
            report_path = output / 'source_bigram_candidates.csv'
            with report_path.open(encoding='utf-8-sig', newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual({row['phrase'] for row in rows}, {'alpha beta', 'beta gamma'})
            alpha_beta = next(row for row in rows if row['phrase'] == 'alpha beta')
            self.assertEqual(alpha_beta['count'], '3')
            self.assertEqual(alpha_beta['top_right_contexts'], 'gamma:3')
            self.assertEqual(alpha_beta['right_entropy'], '0')
            self.assertGreater(float(alpha_beta['left_entropy']), 0)
            candidates = phrase_entropy.read_candidates(report_path)
            self.assertEqual(candidates, {('alpha', 'beta'), ('beta', 'gamma')})

    def test_cli_rejects_invalid_thresholds(self):
        result = subprocess.run(
            [sys.executable, str(Path(select_bigrams.__file__).resolve()),
             '--min-npmi', '1.1'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--min-npmi must be finite and between -1 and 1', result.stderr)


if __name__ == '__main__':
    unittest.main()
