#!/usr/bin/env python3
"""Build the exact processed-frame to bridge-mask manifest for xiaoqiao."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import cv2

def validate_split_membership(split_rows: dict, split_dir: Path) -> None:
    for split in ("train", "val", "test"):
        list_path = split_dir / f"{split}.txt"
        expected = list_path.read_text(encoding="utf-8").splitlines()
        actual = [row["source_name"] for row in split_rows[split]]
        if not expected or any(not name or Path(name).name != name for name in expected):
            raise RuntimeError(f"Invalid filename list: {list_path}")
        if len(set(expected)) != len(expected) or len(set(actual)) != len(actual):
            raise RuntimeError(f"Duplicate filenames in {split} split")
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        if missing or unexpected:
            raise RuntimeError(
                f"{split} split differs from retained list {list_path}: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}. "
                "Check image naming and Nerfstudio split settings before training."
            )

def source_of(name: str) -> str:
    index = int(Path(name).stem.rsplit("_", 1)[1])
    return "UAV" if index <= 331 else "PHONE"


def main() -> None:
    from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

    parser_cli = argparse.ArgumentParser(description=__doc__)
    parser_cli.add_argument("--workspace", type=Path, required=True)
    parser_cli.add_argument("--processed-data", type=Path, required=True)
    parser_cli.add_argument("--mask-dir", type=Path, required=True)
    parser_cli.add_argument(
        "--split-dir", type=Path,
        default=Path(__file__).resolve().parents[4] / "data/compact_bridge/metadata",
        help="Directory containing retained train.txt, val.txt and test.txt lists",
    )
    args = parser_cli.parse_args()
    DATA = args.processed_data.resolve()
    MASKS = args.mask_dir.resolve()
    MANIFEST = args.workspace.resolve() / "config/exact_manifest.json"
    SAMPLER = args.workspace.resolve() / "config/bridge_patch_sampler.json"
    if not (DATA / "transforms.json").is_file():
        parser_cli.error(f"Missing processed Nerfstudio transforms: {DATA / 'transforms.json'}")
    if not MASKS.is_dir():
        parser_cli.error(f"Mask directory not found: {MASKS}")
    transforms = json.loads((DATA / "transforms.json").read_text())
    all_rows = {}
    source_counts = {"UAV": 0, "PHONE": 0}
    empty_masks = []
    for frame in transforms["frames"]:
        name = Path(frame["file_path"]).name
        source = source_of(name)
        mask_path = MASKS / f"{Path(name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        area_ratio = float((mask > 127).mean())
        if area_ratio == 0:
            empty_masks.append(name)
        source_counts[source] += 1
        all_rows[name] = {
            "processed_file": str(DATA / "images" / name),
            "colmap_image_id": int(frame["colmap_im_id"]),
            "source_name": name,
            "source": source,
            "mask_path": str(mask_path),
            "mask_width": int(mask.shape[1]),
            "mask_height": int(mask.shape[0]),
            "mask_area_ratio": area_ratio,
        }

    parser = NerfstudioDataParserConfig(
        data=DATA, downscale_factor=2, eval_mode="interval", eval_interval=8
    ).setup()
    split_rows, split_counts = {}, {}
    sampler_masks, sampler_sources = {}, {}
    for split in ("train", "val", "test"):
        outputs = parser.get_dataparser_outputs(split=split)
        rows, counts = [], {"UAV": 0, "PHONE": 0}
        for dataset_index, image_path in enumerate(outputs.image_filenames):
            row = all_rows[Path(image_path).name]
            rows.append({"dataset_index": dataset_index, **row})
            counts[row["source"]] += 1
            if split == "train":
                sampler_masks[str(dataset_index)] = row["mask_path"]
                sampler_sources[str(dataset_index)] = row["source"]
        split_rows[split], split_counts[split] = rows, counts

    if len(all_rows) != 600 or source_counts != {"UAV": 331, "PHONE": 269}:
        raise RuntimeError(f"Unexpected dataset: frames={len(all_rows)}, sources={source_counts}")
    if len(split_rows["train"]) != 525 or len(split_rows["test"]) != 75:
        raise RuntimeError(f"Unexpected split: train={len(split_rows['train'])}, test={len(split_rows['test'])}")
    if empty_masks:
        raise RuntimeError(f"Empty masks are not allowed: {empty_masks[:5]}")
    validate_split_membership(split_rows, args.split_dir.resolve())

    manifest = {
        "dataset": str(DATA), "mask_root": str(MASKS),
        "mapping_method": "transforms frame basename -> exact same-basename mask",
        "total_frames": len(all_rows), "source_counts": source_counts,
        "split_counts": split_counts, "empty_mask_count": len(empty_masks),
        "verified_split_directory": str(args.split_dir.resolve()),
        "frames": all_rows, "splits": split_rows,
    }
    sampler = {
        "mode": "bridge_patch_v2", "patch_size": 32,
        "patch_layout": ["bridge", "bridge", "boundary", "global"],
        "mask_paths": sampler_masks, "source_labels": sampler_sources,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    SAMPLER.write_text(json.dumps(sampler, indent=2, ensure_ascii=False))
    print(json.dumps({
        "manifest": str(MANIFEST), "sampler": str(SAMPLER),
        "frames": len(all_rows), "sources": source_counts,
        "splits": {key: {"frames": len(value), "sources": split_counts[key]}
                   for key, value in split_rows.items()},
        "empty_masks": len(empty_masks),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
