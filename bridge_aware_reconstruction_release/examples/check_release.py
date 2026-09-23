"""Validate Compact Bridge release-candidate data files."""

from __future__ import annotations

import csv
import argparse
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a separately authorized Compact Bridge dataset")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="dataset directory containing images/, masks/, and metadata/image_source_mapping.csv")
    parser.add_argument("--deep", action="store_true", help="also decode images and masks (requires Pillow)")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    images = sorted((data_dir / "images").glob("*.jpg"))
    masks = sorted((data_dir / "masks").glob("*.png"))
    with (data_dir / "metadata" / "image_source_mapping.csv").open(newline="", encoding="utf-8") as f:
        mapping = list(csv.DictReader(f))

    counts = Counter(row["group_source"] for row in mapping)
    assert len(images) == 600, len(images)
    assert len(masks) == 600, len(masks)
    assert len(mapping) == 600, len(mapping)
    assert counts == {"UAV": 331, "PHONE": 269}, counts
    image_names = {p.name for p in images}
    mapped_names = [row["name"] for row in mapping]
    assert len(set(mapped_names)) == len(mapped_names), "duplicate mapped image names"
    assert set(mapped_names) == image_names, "mapping and image names differ"
    assert len({p.stem for p in masks}) == len(masks), "duplicate mask stems"
    assert {p.stem for p in masks} == {p.stem for p in images}, "image/mask basenames differ"
    assert all(row["group_source"] in {"UAV", "PHONE"} for row in mapping)

    if args.deep:
        from PIL import Image

        mask_by_stem = {path.stem: path for path in masks}
        for image_path in images:
            with Image.open(image_path) as photo, Image.open(mask_by_stem[image_path.stem]) as mask:
                assert photo.size == (960, 540), (image_path, photo.size)
                photo.verify()
                assert mask.size == (960, 540), (mask_by_stem[image_path.stem], mask.size)
                assert mask.getextrema() != (0, 0), f"Empty mask: {mask_by_stem[image_path.stem]}"
        print("deep image/mask decode passed")

    print("Dataset check passed")
    print(f"images={len(images)}, masks={len(masks)}, sources={dict(counts)}")


if __name__ == "__main__":
    main()
