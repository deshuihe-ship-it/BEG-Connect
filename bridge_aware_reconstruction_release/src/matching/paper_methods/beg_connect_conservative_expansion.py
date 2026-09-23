#!/usr/bin/env python3
"""Bridge-enhanced conservative cross-source pair expansion for UAV1+PHONE1.

This script starts from the completed 400-image DINOv2+LightGlue database.
It retains those matches, generates new masks/enhanced images from the same
1280 inputs, then adds only RANSAC-verified LightGlue matches from a bounded
temporal neighbourhood of reliable cross-source seed pairs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(os.environ.get("BRIDGE_LARGE_ROOT", "UNSET_LARGE_SPAN_ROOT"))
IMAGE_DIR = ROOT / "images"
BASE = ROOT / "all_dinov2_lightglue"
OUT = ROOT / "all_dinov2_lightglue_bridge_enhance_expand"
COLMAP = Path(os.environ.get("COLMAP_BIN", "colmap"))
YOLO_WEIGHTS = Path(os.environ.get("BRIDGE_YOLO_WEIGHTS", "UNSET_YOLO_WEIGHTS"))
MAX_IMAGE_ID = 2147483647

# The final conservative-expansion settings used on component_v2_phone2.
UAV_RADIUS = 2
PHONE_RADIUS = 1
MAX_NEW_PER_PHONE = 2
MAX_NEW_PER_UAV = 4
MIN_INLIERS = 30
MIN_INLIER_RATIO = 0.25
RANSAC_THRESHOLD = 2.0
ALIKED_RESIZE = 1024
ALIKED_MAX_KEYPOINTS = 2048
DIM_OUTSIDE = 0.90


def load_dl_module():
    path = Path(__file__).with_name("dl_dinov2_lightglue.py")
    spec = importlib.util.spec_from_file_location("uav1_phone1_dl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dl = load_dl_module()


def log(message: str) -> None:
    path = OUT / "logs" / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def source_of(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("uav_"):
        return "uav"
    if stem.startswith("phone1_"):
        return "phone1"
    return "unknown"


def frame_index(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def reset_output() -> Path:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUT}")
    (OUT / "colmap").mkdir(parents=True)
    (OUT / "reports").mkdir()
    (OUT / "bridge_masks_yolo").mkdir()
    (OUT / "bridge_enhanced_images").mkdir()
    shutil.copy2(BASE / "colmap" / "database.db", OUT / "colmap" / "database.db")
    (OUT / "images").symlink_to(IMAGE_DIR, target_is_directory=True)
    return OUT / "colmap" / "database.db"


def generate_masks(mask_dir: Path) -> dict[str, object]:
    from ultralytics import YOLO

    model = YOLO(str(YOLO_WEIGHTS))
    image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))
    valid = 0
    areas: list[float] = []
    device = 0 if torch.cuda.is_available() else "cpu"
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable input image: {image_path}")
        height, width = image.shape[:2]
        result = model.predict(
            source=str(image_path), imgsz=768, conf=0.25, device=device,
            batch=1, verbose=False, stream=False,
        )[0]
        mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is not None and result.masks.data is not None:
            for candidate in result.masks.data.detach().cpu().numpy():
                resized = cv2.resize(
                    (candidate > 0.5).astype(np.uint8), (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask[resized > 0] = 255
        valid += int(mask.any())
        areas.append(float((mask > 0).mean()))
        cv2.imwrite(str(mask_dir / f"{image_path.stem}.png"), mask)
        if index % 50 == 0 or index == len(image_paths):
            log(f"BEG masks {index}/{len(image_paths)}")
    return {
        "images": len(image_paths), "valid_masks": valid,
        "valid_ratio": valid / len(image_paths) if image_paths else 0.0,
        "mean_area_ratio": float(np.mean(areas)) if areas else 0.0,
        "weights": str(YOLO_WEIGHTS),
    }


def build_enhanced_images(mask_dir: Path, enhanced_dir: Path) -> dict[str, int]:
    image_paths = sorted(IMAGE_DIR.rglob("*.jpg"))
    nonempty = 0
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Unreadable input image: {image_path}")
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        inside = mask > 0
        output = image
        if inside.any():
            nonempty += 1
            output = np.clip(image.astype(np.float32) * DIM_OUTSIDE, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
            enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
            blur = cv2.GaussianBlur(enhanced, (0, 0), 1.1)
            sharpened = cv2.addWeighted(enhanced, 1.45, blur, -0.45, 0)
            output[inside] = sharpened[inside]
        relative = image_path.relative_to(IMAGE_DIR)
        destination = enhanced_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if index % 50 == 0 or index == len(image_paths):
            log(f"BEG enhanced images {index}/{len(image_paths)}")
    return {"images": len(image_paths), "with_nonempty_mask": nonempty}


def seed_pairs(db_path: Path) -> tuple[list[tuple[str, str, int]], set[tuple[str, str]]]:
    seeds: list[tuple[str, str, int]] = []
    existing: set[tuple[str, str]] = set()
    with sqlite3.connect(db_path) as connection:
        names = {int(image_id): name for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        for pid, rows in connection.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
            first, second = pair_id_to_image_ids(int(pid))
            name1, name2 = names.get(first), names.get(second)
            if not name1 or not name2 or {source_of(name1), source_of(name2)} != {"uav", "phone1"}:
                continue
            uav, phone = (name1, name2) if source_of(name1) == "uav" else (name2, name1)
            existing.add((uav, phone))
            if int(rows) >= MIN_INLIERS:
                seeds.append((uav, phone, int(rows)))
    return sorted(seeds, key=lambda row: row[2], reverse=True), existing


def expansion_pairs(seeds: list[tuple[str, str, int]], existing: set[tuple[str, str]], names: list[str]):
    by_source = {source: {frame_index(name): name for name in names if source_of(name) == source} for source in ("uav", "phone1")}
    scored: dict[tuple[str, str], float] = {}
    for uav_seed, phone_seed, support in seeds:
        for uav_i in range(frame_index(uav_seed) - UAV_RADIUS, frame_index(uav_seed) + UAV_RADIUS + 1):
            uav = by_source["uav"].get(uav_i)
            if uav is None:
                continue
            for phone_i in range(frame_index(phone_seed) - PHONE_RADIUS, frame_index(phone_seed) + PHONE_RADIUS + 1):
                phone = by_source["phone1"].get(phone_i)
                if phone is None or (uav, phone) in existing:
                    continue
                score = float(support - 5 * (abs(uav_i - frame_index(uav_seed)) + abs(phone_i - frame_index(phone_seed))))
                scored[(uav, phone)] = max(scored.get((uav, phone), -np.inf), score)
    selected = []
    per_uav: defaultdict[str, int] = defaultdict(int)
    per_phone: defaultdict[str, int] = defaultdict(int)
    for (uav, phone), score in sorted(scored.items(), key=lambda item: item[1], reverse=True):
        if per_uav[uav] >= MAX_NEW_PER_UAV or per_phone[phone] >= MAX_NEW_PER_PHONE:
            continue
        selected.append((uav, phone, score))
        per_uav[uav] += 1
        per_phone[phone] += 1
    return selected


def cross_source_db_stats(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as connection:
        names = {int(image_id): name for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        pairs = matches = 0
        for pid, rows in connection.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"):
            first, second = pair_id_to_image_ids(int(pid))
            if {source_of(names.get(first, "")), source_of(names.get(second, ""))} == {"uav", "phone1"}:
                pairs += 1
                matches += int(rows)
    return {"cross_source_verified_pairs": pairs, "cross_source_verified_matches": matches}


def main() -> None:
    if not IMAGE_DIR.is_dir() or not (BASE / "colmap/database.db").is_file():
        raise FileNotFoundError("Set BRIDGE_LARGE_ROOT to the authorized images and completed DL output")
    if not YOLO_WEIGHTS.is_file():
        raise FileNotFoundError("Set BRIDGE_YOLO_WEIGHTS to an authorized bridge-segmentation weight")
    db_path = reset_output()
    log("STAGE=BEG_BUILD_START input_images=400 bridge_enhance=true guide=conservative_pair_expansion")
    masks = generate_masks(OUT / "bridge_masks_yolo")
    enhanced = build_enhanced_images(OUT / "bridge_masks_yolo", OUT / "bridge_enhanced_images")
    (OUT / "reports" / "bridge_enhance_summary.json").write_text(json.dumps({**masks, **enhanced}, indent=2), encoding="utf-8")
    if masks["valid_masks"] == 0:
        raise RuntimeError("All bridge masks are empty; stopping before matching")

    seeds, existing = seed_pairs(BASE / "colmap" / "database.db")
    names = sorted(path.relative_to(IMAGE_DIR).as_posix() for path in IMAGE_DIR.rglob("*.jpg"))
    candidates = expansion_pairs(seeds, existing, names)
    (OUT / "reports" / "pair_expansion_summary.json").write_text(json.dumps({
        "seed_pairs_ge30": len(seeds), "existing_cross_pairs": len(existing),
        "new_candidate_pairs": len(candidates), "uav_radius": UAV_RADIUS,
        "phone_radius": PHONE_RADIUS, "max_new_per_uav": MAX_NEW_PER_UAV,
        "max_new_per_phone": MAX_NEW_PER_PHONE, "min_inliers": MIN_INLIERS,
        "min_inlier_ratio": MIN_INLIER_RATIO,
    }, indent=2), encoding="utf-8")
    if not candidates:
        raise RuntimeError("No conservative expansion candidates were generated")
    log(f"pair expansion seeds={len(seeds)} candidates={len(candidates)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this BEG LightGlue run")
    unique_names = sorted({name for uav, phone, _ in candidates for name in (uav, phone)})
    extractor, matcher = dl.build_models(device)
    original_dir = dl.IMAGE_DIR
    dl.IMAGE_DIR = OUT / "bridge_enhanced_images"
    try:
        features = dl.extract_all_aliked_features(unique_names, extractor, device)
    finally:
        dl.IMAGE_DIR = original_dir

    inserted_pairs = inserted_inliers = 0
    with sqlite3.connect(db_path) as connection:
        image_ids = {name: int(image_id) for image_id, name in connection.execute("SELECT image_id, name FROM images")}
        offsets = dl.append_keypoints(connection, image_ids, features)
        for index, (uav, phone, _) in enumerate(candidates, start=1):
            inliers, fmat, _, count, ratio = dl.match_and_verify(uav, phone, features, matcher, MIN_INLIERS, RANSAC_THRESHOLD)
            if len(inliers) and count >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO:
                inserted_inliers += dl.upsert_matches(connection, image_ids, offsets, uav, phone, inliers, fmat)
                inserted_pairs += 1
            if index % 50 == 0 or index == len(candidates):
                connection.commit()
                log(f"BEG LightGlue {index}/{len(candidates)} inserted_pairs={inserted_pairs}")
        connection.commit()

    sparse_dir = OUT / "colmap" / "sparse"
    sparse_dir.mkdir()
    log("Starting BEG mapper")
    dl.run_mapper(db_path, IMAGE_DIR, sparse_dir, OUT / "logs" / "mapper.log")
    models = sorted((path for path in sparse_dir.iterdir() if path.is_dir() and path.name.isdigit()), key=lambda path: int(path.name))
    if not models:
        raise RuntimeError("Mapper produced no sparse model")
    model_summaries = []
    for model_dir in models:
        model_summary = OUT / "reports" / f"registered_summary_{model_dir.name}.json"
        model_summaries.append((model_dir, dl.registered_summary(model_dir, model_summary)))
    selected_model, registered = max(model_summaries, key=lambda item: item[1]["registered_total"])
    (OUT / "reports" / "selected_model.txt").write_text(str(selected_model), encoding="utf-8")
    shutil.copy2(OUT / "reports" / f"registered_summary_{selected_model.name}.json", OUT / "reports" / "registered_summary.json")
    summary = {
        "input_images": len(names), "bridge_enhance": True,
        "bridge_guide": "conservative_pair_expansion", "inserted_pairs": inserted_pairs,
        "inserted_inliers": inserted_inliers, **cross_source_db_stats(db_path), **registered,
    }
    (OUT / "reports" / "pipeline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"STAGE=BEG_BUILD_COMPLETE registered={registered['registered_total']} inserted_pairs={inserted_pairs}")


if __name__ == "__main__":
    main()
