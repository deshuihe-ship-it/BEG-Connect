"""Build a separate enhanced Compact Bridge dataset for reference SIFT experiments."""

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/matching/compact_bridge'))
from run_reference_ablation import configure, enhance_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        parser.error(f'Refusing to overwrite existing output: {output}')
    if not (data/'images').is_dir() or not (data/'masks').is_dir():
        parser.error('Supply a Compact Bridge data directory with images and masks')
    output.mkdir(parents=True)
    _, beg = configure(data, output, None, 'colmap')
    beg.OUT = output/'.generation'
    enhance_images(beg, data, output/'images')
    for folder in ('metadata', 'masks', 'splits'):
        if (data/folder).exists():
            (output/folder).symlink_to(data/folder, target_is_directory=True)


if __name__ == '__main__':
    main()
