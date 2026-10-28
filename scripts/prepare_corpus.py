"""Build the small, attribution-preserving starter corpus from pinned local sources."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helix.corpus import (CorpusManifest, scan_directory, build_training_corpus,
                          RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifest.json')
    parser.add_argument('--output', default='artifacts/data/corpus.txt')
    args = parser.parse_args()
    heldout = ('The capital of France is', 'The secret code is ALPHA-9921.',
               'def is_even(n):', 'Water boils at 500 degrees Celsius at sea level.')
    manifest = CorpusManifest()
    for directory, license_name, version, author in [
        ('data/sources/helix', 'MIT', 'repository baseline 5137540', 'HELIX contributors'),
        ('data/sources/nanogpt', 'MIT', '3adf61e154c3fe3fca428ad6bc3818b27a3b8291', 'Andrej Karpathy / nanoGPT contributors'),
        ('data/sources/sympy', 'BSD-3-Clause; see composite LICENSE', '5b861e4c4a06d56bac4b3f1398985416ae6ec6c7', 'SymPy Development Team and contributors'),
    ]:
        scan_directory(directory, manifest, rights=[RIGHTS_TRAIN, RIGHTS_REDISTRIBUTE],
                       default_license=license_name, license_evidence=directory + '/LICENSE',
                       source_version=version, attribution=author, contamination_strings=heldout)
    manifest.save(args.manifest)
    print('manifest:', manifest.summary())
    print('corpus:', build_training_corpus(manifest, args.output))


if __name__ == '__main__':
    main()
