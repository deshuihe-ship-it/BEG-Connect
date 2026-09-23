#!/usr/bin/env python3
"""Build an exact processed-frame -> COLMAP source -> bridge-mask manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
DATA = Path(os.environ.get("BEG_NERF_PROCESSED_DATA", "UNSET_PROCESSED_DATA"))
SPARSE = Path(os.environ.get("BEG_NERF_SPARSE_MODEL", "UNSET_SPARSE_MODEL"))
MASKS = Path(os.environ.get("BEG_NERF_MASK_DIR", "UNSET_MASK_DIR"))
MANIFEST = WORK / "config/exact_manifest.json"
SAMPLER = WORK / "config/bridge_patch_sampler.json"


def main() -> None:
    if not (DATA / "transforms.json").is_file() or not (SPARSE / "images.bin").is_file() or not MASKS.is_dir():
        raise FileNotFoundError("Set BEG_NERF_PROCESSED_DATA, BEG_NERF_SPARSE_MODEL and BEG_NERF_MASK_DIR to authorized inputs")
    transforms = json.loads((DATA / "transforms.json").read_text())
    frame_by_name = {Path(frame["file_path"]).name: frame for frame in transforms["frames"]}
    colmap_images = read_images_binary(SPARSE / "images.bin")

    all_rows = {}
    source_counts = {"UAV": 0, "PHONE": 0}
    empty_masks = []
    for processed_name, frame in frame_by_name.items():
        image_id = int(frame["colmap_im_id"])
        source_name = Path(colmap_images[image_id].name).name
        if source_name.startswith("uav_"):
            source = "UAV"
        elif source_name.startswith("phone1_"):
            source = "PHONE"
        else:
            raise RuntimeError(f"Unknown source image: {source_name}")
        mask_path = MASKS / f"{Path(source_name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(mask_path)
        area_ratio = float((mask > 127).mean())
        if area_ratio == 0:
            empty_masks.append(processed_name)
        source_counts[source] += 1
        all_rows[processed_name] = {
            "processed_file": str(DATA / "images" / processed_name),
            "colmap_image_id": image_id,
            "source_name": source_name,
            "source": source,
            "mask_path": str(mask_path),
            "mask_width": int(mask.shape[1]),
            "mask_height": int(mask.shape[0]),
            "mask_area_ratio": area_ratio,
        }

    parser = NerfstudioDataParserConfig(
        data=DATA, downscale_factor=2, eval_mode="interval", eval_interval=8
    ).setup()
    split_rows = {}
    sampler_masks = {}
    sampler_sources = {}
    split_counts = {}
    for split in ("train", "val", "test"):
        outputs = parser.get_dataparser_outputs(split=split)
        rows = []
        counts = {"UAV": 0, "PHONE": 0}
        for dataset_idx, image_path in enumerate(outputs.image_filenames):
            row = all_rows[Path(image_path).name]
            rows.append({"dataset_index": dataset_idx, **row})
            counts[row["source"]] += 1
            if split == "train":
                sampler_masks[str(dataset_idx)] = row["mask_path"]
                sampler_sources[str(dataset_idx)] = row["source"]
        split_rows[split] = rows
        split_counts[split] = counts

    if len(all_rows) != 400 or source_counts != {"UAV": 200, "PHONE": 200}:
        raise RuntimeError(f"Unexpected dataset: frames={len(all_rows)}, sources={source_counts}")
    if empty_masks:
        raise RuntimeError(f"Empty masks are not allowed: {empty_masks[:5]}")

    manifest = {
        "dataset": str(DATA),
        "sparse": str(SPARSE),
        "mask_root": str(MASKS),
        "mapping_method": "transforms.colmap_im_id -> images.bin name -> exact mask basename",
        "total_frames": len(all_rows),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "empty_mask_count": len(empty_masks),
        "frames": all_rows,
        "splits": split_rows,
    }
    sampler = {
        "mode": "bridge_patch_v2",
        "patch_size": 32,
        "patch_layout": ["bridge", "bridge", "boundary", "global"],
        "mask_paths": sampler_masks,
        "source_labels": sampler_sources,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    SAMPLER.write_text(json.dumps(sampler, indent=2, ensure_ascii=False))
    print(json.dumps({
        "manifest": str(MANIFEST), "sampler": str(SAMPLER),
        "frames": len(all_rows), "sources": source_counts,
        "splits": {k: {"frames": len(v), "sources": split_counts[k]} for k, v in split_rows.items()},
        "empty_masks": len(empty_masks),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
