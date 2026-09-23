#!/usr/bin/env python3
from __future__ import annotations
"""
Full DINOv2 + ALIKED+LightGlue pipeline for uav1+phone11280.
All matching via DINO candidates: uav↔uav, phone1↔phone1, phone1↔uav.
"""

import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(os.environ.get("BRIDGE_LARGE_ROOT", "UNSET_LARGE_SPAN_ROOT"))
LIGHTGLUE_REPO = Path(os.environ.get("LIGHTGLUE_REPO", "UNSET_LIGHTGLUE_REPO"))

COLMAP = Path(os.environ.get("COLMAP_BIN", "colmap"))
PYTHON = Path(sys.executable)

IMAGE_DIR = ROOT / "images"
OUT_DIR = ROOT / "all_dinov2_lightglue"
DB_DIR = OUT_DIR / "colmap"
DB_PATH = DB_DIR / "database.db"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
SPARSE_DIR = OUT_DIR / "colmap" / "sparse"
PAIR_FILE = REPORT_DIR / "all_candidate_pairs.txt"
MAX_IMAGE_ID = 2147483647

TOP_K = 20
MIN_INLIERS = 30
RANSAC_THRESHOLD = 2.0
RESIZE = 1024
MAX_KEYPOINTS = 2048

sys.path.insert(0, str(LIGHTGLUE_REPO))

# ── helpers ──────────────────────────────────────────────────────────

def source_of(name: str) -> str:
    if "uav_" in name: return "uav"
    if "phone1_" in name: return "phone1"
    if "phone2_" in name: return "phone2"
    return "unknown"

def pair_id(image_id1: int, image_id2: int) -> int:
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2

def log(msg: str):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

# ── Step 1: DINOv2 feature extraction ───────────────────────────────

def load_dinov2(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()
    return model

def make_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def extract_dinov2_features(paths: list[Path], batch_size: int = 32) -> tuple[list[str], np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"DINOv2 device={device}")
    model = load_dinov2(device)
    transform = make_transform()
    names = []
    features = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        tensors = []
        for p in batch:
            tensors.append(transform(Image.open(p).convert("RGB")))
            names.append(p.relative_to(IMAGE_DIR).as_posix())
        x = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            y = model(x)
            y = torch.nn.functional.normalize(y, dim=1)
        features.append(y.cpu().numpy().astype(np.float32))
        log(f"DINOv2 {min(start + batch_size, len(paths))}/{len(paths)}")
    return names, np.concatenate(features, axis=0)

# ── Step 2: Candidate pair generation ───────────────────────────────

def build_candidates_internal(names: list[str], feats: np.ndarray, source: str, topk: int) -> list[tuple[str, str]]:
    name_to_idx = {n: i for i, n in enumerate(names)}
    src_names = [n for n in names if source_of(n) == source]
    src_matrix = feats[[name_to_idx[n] for n in src_names]]
    pairs: set[tuple[str, str]] = set()
    for i, img in enumerate(src_names):
        vec = feats[name_to_idx[img]]
        sims = src_matrix @ vec
        order = np.argsort(-sims)[:topk + 1]  # +1 because self is always #1
        for idx in order:
            candidate = src_names[int(idx)]
            if candidate == img:
                continue
            a, b = (img, candidate) if img < candidate else (candidate, img)
            pairs.add((a, b))
        if (i + 1) % 50 == 0:
            log(f"candidates {source} {i+1}/{len(src_names)} unique_pairs={len(pairs)}")
    return sorted(pairs)

def build_candidates_cross(names: list[str], feats: np.ndarray, phone_source: str, topk: int) -> list[tuple[str, str]]:
    name_to_idx = {n: i for i, n in enumerate(names)}
    uav_names = [n for n in names if source_of(n) == "uav"]
    phone_names = [n for n in names if source_of(n) == phone_source]
    uav_matrix = feats[[name_to_idx[n] for n in uav_names]]
    pairs: list[tuple[str, str]] = []
    for phone in phone_names:
        phone_vec = feats[name_to_idx[phone]]
        sims = uav_matrix @ phone_vec
        order = np.argsort(-sims)[:topk]
        for idx in order:
            pairs.append((phone, uav_names[int(idx)]))
    log(f"candidates {phone_source}→uav: {len(pairs)} pairs")
    return pairs

# ── Step 3: ALIKED + LightGlue matching ─────────────────────────────

def build_models(device: torch.device):
    from lightglue import ALIKED, LightGlue
    extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    matcher = LightGlue(features="aliked", filter_threshold=0.05).eval().to(device)
    return extractor, matcher

def extract_all_aliked_features(image_names: list[str], extractor, device):
    from lightglue.utils import load_image
    features = {}
    with torch.no_grad():
        for idx, name in enumerate(image_names, start=1):
            image = load_image(IMAGE_DIR / name).to(device)
            feats = extractor.extract(image, resize=RESIZE)
            features[name] = feats
            if idx % 50 == 0:
                log(f"ALIKED {idx}/{len(image_names)}")
    return features

def match_and_verify(img1: str, img2: str, features, matcher, min_inliers: int, ransac_threshold: float):
    from lightglue.utils import rbd
    feats0 = features[img1]
    feats1 = features[img2]
    with torch.no_grad():
        matches01 = matcher({"image0": feats0, "image1": feats1})
    matches01 = rbd(matches01)
    matches = matches01["matches"].detach().cpu().numpy().astype(np.uint32)
    if len(matches) < 8:
        return np.empty((0, 2), dtype=np.uint32), None, int(len(matches)), 0, 0.0
    kpts0 = rbd(feats0)["keypoints"].detach().cpu().numpy()
    kpts1 = rbd(feats1)["keypoints"].detach().cpu().numpy()
    pts0 = kpts0[matches[:, 0]].astype(np.float32)
    pts1 = kpts1[matches[:, 1]].astype(np.float32)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.FM_RANSAC
    fmat, mask = cv2.findFundamentalMat(pts0, pts1, method,
        ransacReprojThreshold=ransac_threshold, confidence=0.999, maxIters=10000)
    if fmat is None or mask is None:
        return np.empty((0, 2), dtype=np.uint32), None, int(len(matches)), 0, 0.0
    keep = mask.ravel().astype(bool)
    inlier_matches = matches[keep]
    inliers = int(len(inlier_matches))
    if inliers < min_inliers:
        return np.empty((0, 2), dtype=np.uint32), fmat, int(len(matches)), inliers, float(inliers / len(matches))
    return inlier_matches, fmat.astype(np.float64), int(len(matches)), inliers, float(inliers / len(matches))

# ── Step 4: Database operations ─────────────────────────────────────

def read_keypoints(con: sqlite3.Connection, image_id: int) -> np.ndarray:
    row = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    if row is None or row[0] == 0:
        return np.empty((0, 6), dtype=np.float32)
    arr = np.frombuffer(row[2], dtype=np.float32).reshape(int(row[0]), int(row[1]))
    if arr.shape[1] == 6:
        return arr.copy()
    out = np.zeros((arr.shape[0], 6), dtype=np.float32)
    out[:, :arr.shape[1]] = arr
    return out

def make_colmap_keypoints(xy: np.ndarray) -> np.ndarray:
    out = np.zeros((xy.shape[0], 6), dtype=np.float32)
    out[:, 0:2] = xy.astype(np.float32)
    out[:, 2] = 1.0
    return out

def append_keypoints(con: sqlite3.Connection, image_ids: dict[str, int], features: dict) -> dict[str, int]:
    offsets = {}
    for name, feats in features.items():
        image_id = image_ids[name]
        old = read_keypoints(con, image_id)
        offsets[name] = int(old.shape[0])
        xy = feats["keypoints"][0].detach().cpu().numpy()
        new = make_colmap_keypoints(xy)
        merged = np.vstack([old, new]).astype(np.float32)
        con.execute(
            "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?)",
            (image_id, int(merged.shape[0]), 6, merged.tobytes()),
        )
    return offsets

def read_uint32_pairs(con: sqlite3.Connection, table: str, pid: int) -> np.ndarray:
    row = con.execute(f"SELECT rows, cols, data FROM {table} WHERE pair_id=?", (pid,)).fetchone()
    if row is None or row[0] == 0:
        return np.empty((0, 2), dtype=np.uint32)
    return np.frombuffer(row[2], dtype=np.uint32).reshape(int(row[0]), int(row[1])).copy()

def write_match_rows(con: sqlite3.Connection, pid: int, pairs: np.ndarray):
    con.execute(
        "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) VALUES(?, ?, 2, ?)",
        (pid, int(pairs.shape[0]), pairs.astype(np.uint32).tobytes()),
    )

def write_two_view_rows(con: sqlite3.Connection, pid: int, pairs: np.ndarray, fmat: np.ndarray | None):
    f_blob = None if fmat is None else fmat.astype(np.float64).reshape(3, 3).tobytes()
    h_blob = np.eye(3, dtype=np.float64).tobytes()
    con.execute(
        "INSERT OR REPLACE INTO two_view_geometries(pair_id, rows, cols, data, config, F, E, H, qvec, tvec) "
        "VALUES (?, ?, 2, ?, 3, ?, NULL, ?, NULL, NULL)",
        (pid, int(pairs.shape[0]), pairs.astype(np.uint32).tobytes(), f_blob, h_blob),
    )

def upsert_matches(con, image_ids, offsets, name1, name2, inlier_matches, fmat) -> int:
    id1 = image_ids[name1]
    id2 = image_ids[name2]
    pid = pair_id(id1, id2)
    offset_matches = inlier_matches.copy().astype(np.uint32)
    offset_matches[:, 0] += offsets[name1]
    offset_matches[:, 1] += offsets[name2]
    if id1 > id2:
        offset_matches = offset_matches[:, [1, 0]]
    old_matches = read_uint32_pairs(con, "matches", pid)
    merged = np.vstack([old_matches, offset_matches]).astype(np.uint32)
    write_match_rows(con, pid, merged)
    old_verified = read_uint32_pairs(con, "two_view_geometries", pid)
    merged_v = np.vstack([old_verified, offset_matches]).astype(np.uint32)
    write_two_view_rows(con, pid, merged_v, fmat)
    return int(offset_matches.shape[0])

# ── Step 0: Feature extractor ───────────────────────────────────────

def run_feature_extractor(db_path: Path, image_dir: Path, log_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(COLMAP), "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "0",
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.max_num_features", "8192",
    ]
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

# ── Step 5: Mapper ──────────────────────────────────────────────────

def run_mapper(db_path: Path, image_dir: Path, sparse_dir: Path, log_path: Path):
    sparse_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = sparse_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    cmd = [
        str(COLMAP), "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.min_num_matches", "15",
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_local_max_num_iterations", "10",
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.ba_global_max_refinements", "5",
        "--Mapper.ba_global_frames_freq", "400",
        "--Mapper.ba_global_points_freq", "200000",
        "--Mapper.snapshot_path", str(snapshot_dir),
        "--Mapper.snapshot_frames_freq", "50",
    ]
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

def registered_summary(sparse_dir: Path, out_path: Path) -> dict:
    tmp = out_path.parent / "_txt_model"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    subprocess.run(
        [str(COLMAP), "model_converter", "--input_path", str(sparse_dir), "--output_path", str(tmp), "--output_type", "TXT"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    src_images = tmp / "images.txt"
    names = []
    for line in src_images.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9].lower().endswith((".jpg", ".jpeg", ".png")):
            names.append(parts[9])
    summary = {
        "registered_total": len(names),
        "registered_uav": sum(1 for n in names if source_of(n) == "uav"),
        "registered_phone1": sum(1 for n in names if source_of(n) == "phone1"),
        "registered_names": names,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    shutil.rmtree(tmp)
    return summary

# ── Main ────────────────────────────────────────────────────────────

def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Set BRIDGE_LARGE_ROOT to a dataset with images/: {IMAGE_DIR}")
    if not LIGHTGLUE_REPO.is_dir():
        raise FileNotFoundError(f"Set LIGHTGLUE_REPO to the installed LightGlue repository: {LIGHTGLUE_REPO}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 0: Feature extraction
    log("Running COLMAP feature_extractor...")
    run_feature_extractor(DB_PATH, IMAGE_DIR, LOG_DIR / "feature_extractor.log")
    log("Feature extraction complete")

    image_paths = sorted(IMAGE_DIR.glob("**/*.jpg"))
    # Use relative names like "uav/uav_000000.jpg"
    image_names = sorted(p.relative_to(IMAGE_DIR).as_posix() for p in image_paths)
    log(f"Total images: {len(image_names)}")

    # Step 1: DINOv2 features
    names, feats = extract_dinov2_features(image_paths)

    # Step 2: Candidate pairs
    pairs_uu = build_candidates_internal(names, feats, "uav", TOP_K)
    log(f"uav↔uav candidates: {len(pairs_uu)}")

    pairs_pp = build_candidates_internal(names, feats, "phone1", TOP_K)
    log(f"phone1↔phone1 candidates: {len(pairs_pp)}")

    pairs_pu = build_candidates_cross(names, feats, "phone1", TOP_K)
    log(f"phone1↔uav candidates: {len(pairs_pu)}")

    all_pairs = pairs_uu + pairs_pp + pairs_pu
    log(f"Total candidate pairs: {len(all_pairs)}")

    # Deduplicate
    seen = set()
    unique_pairs = []
    for a, b in all_pairs:
        key = (a, b) if a < b else (b, a)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((a, b))
    log(f"Unique candidate pairs: {len(unique_pairs)}")

    # Save pair file
    with PAIR_FILE.open("w") as f:
        for a, b in unique_pairs:
            f.write(f"{a} {b}\n")

    # Step 3: ALIKED + LightGlue
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"LightGlue device={device}")

    unique_images = sorted({name for pair in unique_pairs for name in pair})
    log(f"Unique images for LightGlue: {len(unique_images)}")

    extractor, matcher = build_models(device)
    features = extract_all_aliked_features(unique_images, extractor, device)

    # Step 4: Database insertion
    with sqlite3.connect(DB_PATH) as con:
        image_ids = {name: int(iid) for iid, name in con.execute("SELECT image_id, name FROM images")}
        offsets = append_keypoints(con, image_ids, features)
        con.commit()
        log(f"Appended ALIKED keypoints to {len(offsets)} images")

        @dataclass
        class Result:
            img1: str; img2: str; matches: int; inliers: int; inlier_ratio: float; inserted: int

        results = []
        for idx, (img1, img2) in enumerate(unique_pairs, start=1):
            inlier_matches, fmat, n_matches, inliers, ratio = match_and_verify(
                img1, img2, features, matcher, MIN_INLIERS, RANSAC_THRESHOLD,
            )
            inserted = 0
            if len(inlier_matches) >= MIN_INLIERS:
                inserted = upsert_matches(con, image_ids, offsets, img1, img2, inlier_matches, fmat)
            results.append(Result(img1, img2, n_matches, inliers, ratio, inserted))
            if idx % 200 == 0:
                con.commit()
                log(f"LightGlue {idx}/{len(unique_pairs)} inserted={sum(r.inserted > 0 for r in results)}")
        con.commit()

    insert_summary = {
        "candidate_pairs": len(unique_pairs),
        "unique_images": len(unique_images),
        "min_inliers": MIN_INLIERS,
        "inserted_pairs": sum(r.inserted > 0 for r in results),
        "inserted_uav_uav": sum(r.inserted > 0 and source_of(r.img1) == "uav" and source_of(r.img2) == "uav" for r in results),
        "inserted_phone1_phone": sum(r.inserted > 0 and source_of(r.img1) == "phone1" and source_of(r.img2) == "phone1" for r in results),
        "inserted_phone1_uav": sum(r.inserted > 0 and (
            (source_of(r.img1) == "phone1" and source_of(r.img2) == "uav") or
            (source_of(r.img1) == "uav" and source_of(r.img2) == "phone1")
        ) for r in results),
        "inserted_inliers_total": int(sum(r.inserted for r in results)),
    }
    (REPORT_DIR / "lightglue_summary.json").write_text(json.dumps(insert_summary, indent=2))
    log(f"Insert summary: {json.dumps(insert_summary)}")

    # CSV
    csv_path = REPORT_DIR / "lightglue_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["img1", "img2", "source", "matches", "inliers", "inlier_ratio", "inserted"])
        w.writeheader()
        for r in results:
            w.writerow({"img1": r.img1, "img2": r.img2, "source": f"{source_of(r.img1)}-{source_of(r.img2)}",
                        "matches": r.matches, "inliers": r.inliers, "inlier_ratio": r.inlier_ratio, "inserted": r.inserted})

    # Step 5: Mapper
    log("Starting COLMAP mapper...")
    start = time.time()
    run_mapper(DB_PATH, IMAGE_DIR, SPARSE_DIR, LOG_DIR / "mapper.log")
    elapsed = time.time() - start
    log(f"Mapper finished in {elapsed/60:.1f} min")

    # Summary
    model_dirs = sorted([p for p in SPARSE_DIR.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if model_dirs:
        reg = registered_summary(model_dirs[0], REPORT_DIR / "registered_summary.json")
    else:
        reg = {"registered_total": 0, "registered_uav": 0, "registered_phone1": 0}
        (REPORT_DIR / "registered_summary.json").write_text(json.dumps(reg, indent=2))
    log(f"Registration: {json.dumps(reg)}")

    # Full report
    report = {**insert_summary, "mapper_minutes": round(elapsed / 60, 1), **reg}
    (REPORT_DIR / "pipeline_summary.json").write_text(json.dumps(report, indent=2))
    log(f"Pipeline complete. Report: {REPORT_DIR / 'pipeline_summary.json'}")


if __name__ == "__main__":
    main()
