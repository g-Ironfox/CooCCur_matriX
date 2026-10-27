'''Tests for trigram extension potential analysis.'''
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import trigram_potential as potential


class TrigramPotentialTests(unittest.TestCase):
    def test_counts_both_extension_directions_and_boundary_entropy(self):
        markdown = ('outerone alpha beta gamma.\n\n'
                    'outertwo alpha beta gamma.\n\n'
                    'outerthree alpha beta gamma.')
        rows = potential.collect_extensions(
            markdown, {('alpha', 'beta'), ('beta', 'gamma')})
        right = next(row for row in rows
                     if row['base_bigram'] == 'alpha beta' and row['extension_side'] == 'right')
        left = next(row for row in rows
                    if row['base_bigram'] == 'beta gamma' and row['extension_side'] == 'left')
        self.assertEqual(right['trigram'], 'alpha beta gamma')
        self.assertEqual(right['trigram_count'], 3)
        self.assertEqual(right['paragraph_frequency'], 3)
        self.assertAlmostEqual(right['base_boundary_entropy'], 0)
        self.assertEqual(left['trigram'], 'alpha beta gamma')
        self.assertEqual(left['trigram_count'], 3)

    def test_cli_writes_ranked_csv_and_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, candidate_file, output = root / 'source.md', root / 'pairs.csv', root / 'out'
            source.write_text('one alpha beta gamma.\n\ntwo alpha beta gamma.', encoding='utf-8')
            candidate_file.write_text('left,right\nalpha,beta\nbeta,gamma\n', encoding='utf-8')
            result = subprocess.run(
                [sys.executable, str(Path(potential.__file__).resolve()), str(source),
                 '--phrase-candidates', str(candidate_file), '--outdir', str(output),
                 '--min-trigram-count', '2'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            csv_path = output / 'source_trigram_potential.csv'
            png_path = output / 'source_trigram_potential.png'
            with csv_path.open(encoding='utf-8-sig', newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertTrue(all(int(row['trigram_count']) >= 2 for row in rows))
            self.assertIn('joint_entropy', rows[0])
            self.assertTrue(all(row['joint_entropy'] != '' for row in rows))
            self.assertEqual(png_path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_repeated_paragraphs_are_deduplicated_and_boundaries_not_bridged(self):
        rows = potential.collect_extensions(
            'alpha beta gamma.\n\nalpha and beta gamma.\n\nalpha beta gamma.',
            {('alpha', 'beta')})
        match = next(row for row in rows if row['trigram'] == 'alpha beta gamma')
        self.assertEqual(match['trigram_count'], 1)


if __name__ == '__main__':
    unittest.main()
