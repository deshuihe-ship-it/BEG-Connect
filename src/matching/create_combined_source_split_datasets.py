from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path('/home/oldwater/data/UVA+PHONE/combined3_vocab/processed_vocab_tree')
OUT_ROOT = Path('/home/oldwater/data/UVA+PHONE/combined_source_split_datasets')
EVAL_INTERVAL = 8


def frame_number(frame_path: str) -> int:
    stem = Path(frame_path).stem
    digits = ''.join(ch for ch in stem if ch.isdigit())
    return int(digits)


def source_name(frame_path: str) -> str:
    num = frame_number(frame_path)
    return 'uav' if num <= 331 else 'phone'


def build_split(meta: dict, source: str) -> dict:
    frames = meta['frames']
    image_paths = [frame['file_path'] for frame in frames]
    eval_paths = [path for idx, path in enumerate(image_paths) if idx % EVAL_INTERVAL == 0]
    train_paths = [
        path for idx, path in enumerate(image_paths)
        if idx % EVAL_INTERVAL != 0 and source_name(path) == source
    ]

    updated = dict(meta)
    updated['train_filenames'] = train_paths
    updated['val_filenames'] = eval_paths
    updated['test_filenames'] = eval_paths
    return updated


def ensure_link(out_dir: Path, name: str, target: Path) -> None:
    link = out_dir / name
    if link.exists() or link.is_symlink():
        return
    link.symlink_to(target)


def main() -> None:
    meta = json.loads((BASE_DIR / 'transforms.json').read_text())
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for source in ('uav', 'phone'):
        out_dir = OUT_ROOT / f'combined_{source}_train_all_eval'
        out_dir.mkdir(parents=True, exist_ok=True)
        updated = build_split(meta, source)
        (out_dir / 'transforms.json').write_text(json.dumps(updated, indent=4))
        ensure_link(out_dir, 'images', BASE_DIR / 'images')
        # Keep optional files available if later needed.
        for optional in ('colmap', 'sparse_pc.ply', 'images_2', 'images_4', 'images_8'):
            target = BASE_DIR / optional
            if target.exists():
                ensure_link(out_dir, optional, target)

        print(
            source,
            'train=', len(updated['train_filenames']),
            'eval=', len(updated['val_filenames']),
            'out=', out_dir,
        )


if __name__ == '__main__':
    main()
