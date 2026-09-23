#!/usr/bin/env python3
"""Build D_R_BEG for the daqiao UAV1+PHONE1 dataset.

D_R_BEG keeps the same DINOv2 Top-20 candidate retrieval and fixed D_L
intra-source geometry used by D_R.  It then applies the published conservative
BEG temporal expansion to verified D_R seed pairs, matches only those expanded
pairs on bridge-enhanced images with RoMa, and rebuilds COLMAP from a combined
RoMa match set.

The completed 4,000-pair D_R match file is reused.  No existing experiment
database or sparse model is modified.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

import h5py


ROOT = Path("/home/oldwater/data/daqiao/uav1+phone11280")
IMAGE_DIR = ROOT / "images"
DR_ROOT = ROOT / "all_dinov2_roma_coarse560_reuse2"
DR_DATABASE = DR_ROOT / "colmap" / "database.db"
DR_MATCHES = DR_ROOT / "reports" / "roma_pairwise_matches.h5"
DR_PAIR_FILE = DR_ROOT / "reports" / "dino_cross_pairs.txt"
ENHANCED_IMAGE_DIR = (
    ROOT
    / "all_dinov2_lightglue_bridge_enhance_expand"
    / "bridge_enhanced_images"
)
DEFAULT_OUTPUT = ROOT / "all_dinov2_roma_bridge_enhance_expand"
ROMA_SCRIPT = Path("/home/oldwater/hello/AIC/scripts/run_dino_roma_daqiao.py")
MAX_IMAGE_ID = 2_147_483_647

# Identical to the final D_L_BEG conservative expansion.
UAV_RADIUS = 2
PHONE_RADIUS = 1
MAX_NEW_PER_UAV = 4
MAX_NEW_PER_PHONE = 2
MIN_INLIERS = 30


def load_roma_module():
    spec = importlib.util.spec_from_file_location("dino_roma_daqiao", ROMA_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {ROMA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


roma = load_roma_module()


def source(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("uav_"):
        return "uav"
    if stem.startswith("phone1_"):
        return "phone1"
    return "unknown"


def frame_index(name: str) -> int:
    return int(Path(name).stem.rsplit("_", 1)[1])


def decode_pair_id(value: int) -> tuple[int, int]:
    image_id1 = value % MAX_IMAGE_ID
    image_id0 = (value - image_id1) // MAX_IMAGE_ID
    return int(image_id0), int(image_id1)


def seed_pairs() -> tuple[list[tuple[str, str, int]], set[tuple[str, str]]]:
    seeds: list[tuple[str, str, int]] = []
    existing: set[tuple[str, str]] = set()
    with sqlite3.connect(DR_DATABASE) as connection:
        names = {
            int(image_id): str(name)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }
        for pair_id, rows in connection.execute(
            "SELECT pair_id, rows FROM two_view_geometries WHERE rows > 0"
        ):
            image_id0, image_id1 = decode_pair_id(int(pair_id))
            name0, name1 = names.get(image_id0), names.get(image_id1)
            if (
                not name0
                or not name1
                or {source(name0), source(name1)} != {"uav", "phone1"}
            ):
                continue
            uav, phone = (
                (name0, name1) if source(name0) == "uav" else (name1, name0)
            )
            existing.add((uav, phone))
            if int(rows) >= MIN_INLIERS:
                seeds.append((uav, phone, int(rows)))
    return sorted(seeds, key=lambda row: row[2], reverse=True), existing


def expansion_pairs(
    seeds: list[tuple[str, str, int]],
    existing: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    names = sorted(
        path.relative_to(IMAGE_DIR).as_posix()
        for path in IMAGE_DIR.rglob("*.jpg")
    )
    by_source = {
        src: {
            frame_index(name): name for name in names if source(name) == src
        }
        for src in ("uav", "phone1")
    }
    scored: dict[tuple[str, str], float] = {}
    for uav_seed, phone_seed, support in seeds:
        uav_seed_index = frame_index(uav_seed)
        phone_seed_index = frame_index(phone_seed)
        for uav_index in range(
            uav_seed_index - UAV_RADIUS, uav_seed_index + UAV_RADIUS + 1
        ):
            uav = by_source["uav"].get(uav_index)
            if uav is None:
                continue
            for phone_index in range(
                phone_seed_index - PHONE_RADIUS,
                phone_seed_index + PHONE_RADIUS + 1,
            ):
                phone = by_source["phone1"].get(phone_index)
                if phone is None or (uav, phone) in existing:
                    continue
                distance = abs(uav_index - uav_seed_index) + abs(
                    phone_index - phone_seed_index
                )
                score = float(support - 5 * distance)
                scored[(uav, phone)] = max(
                    scored.get((uav, phone), float("-inf")), score
                )

    selected: list[tuple[str, str]] = []
    per_uav: defaultdict[str, int] = defaultdict(int)
    per_phone: defaultdict[str, int] = defaultdict(int)
    for (uav, phone), _ in sorted(
        scored.items(), key=lambda item: item[1], reverse=True
    ):
        if (
            per_uav[uav] >= MAX_NEW_PER_UAV
            or per_phone[phone] >= MAX_NEW_PER_PHONE
        ):
            continue
        selected.append((uav, phone))
        per_uav[uav] += 1
        per_phone[phone] += 1
    return selected


def read_pair_file(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            name0, name1 = line.split()
            pairs.append((name0, name1))
    return pairs


def pair_group_names(path: Path) -> set[str]:
    names: set[str] = set()
    with h5py.File(path, "r") as handle:
        def collect(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and "keypoints0" in obj:
                names.add(name)

        handle.visititems(collect)
    return names


def merge_match_files(seed_path: Path, beg_path: Path, output_path: Path) -> dict:
    """Copy seed matches, replacing any retried pair with its BEG result."""
    shutil.copy2(seed_path, output_path)
    replaced = 0
    added = 0
    with h5py.File(beg_path, "r") as source_file, h5py.File(
        output_path, "a"
    ) as destination_file:
        leaf_groups: list[str] = []

        def collect(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and "keypoints0" in obj:
                leaf_groups.append(name)

        source_file.visititems(collect)
        for name in leaf_groups:
            if name in destination_file:
                del destination_file[name]
                replaced += 1
            else:
                added += 1
            parent_name, leaf_name = name.rsplit("/", 1)
            parent = destination_file.require_group(parent_name)
            source_file.copy(name, parent, name=leaf_name)
        # aggregate_matches writes these derived arrays in place.  They may be
        # present in the reused D_R file from its earlier aggregation and must
        # be removed from this copy before a fresh 4,400-pair aggregation.
        combined_leaf_groups: list[str] = []

        def collect_combined(name: str, obj) -> None:
            if isinstance(obj, h5py.Group) and "keypoints0" in obj:
                combined_leaf_groups.append(name)

        destination_file.visititems(collect_combined)
        for name in combined_leaf_groups:
            group = destination_file[name]
            for derived_name in ("matches0", "matching_scores0"):
                if derived_name in group:
                    del group[derived_name]
    return {
        "beg_match_groups": len(leaf_groups),
        "replaced_seed_groups": replaced,
        "added_groups": added,
    }


def accepted_pairs(path: Path) -> set[tuple[str, str]]:
    accepted: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["accepted"]):
                accepted.add((row["name0"], row["name1"]))
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=["candidates", "match", "combine", "database", "mapper", "all"],
        default="all",
    )
    parser.add_argument("--coarse-res", type=int, default=560)
    parser.add_argument("--samples-per-pair", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--max-keypoints", type=int, default=8192)
    parser.add_argument("--merge-error", type=float, default=2.0)
    parser.add_argument("--cell-size", type=int, default=2)
    parser.add_argument("--reuse-existing-radius", type=float, default=2.0)
    args = parser.parse_args()

    output = args.output_root.resolve()
    reports = output / "reports"
    logs = output / "logs"
    colmap = output / "colmap"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    colmap.mkdir(parents=True, exist_ok=True)

    beg_pairs_path = reports / "beg_expansion_pairs.txt"
    beg_match_path = reports / "beg_roma_pairwise_matches.h5"
    beg_metrics_path = reports / "beg_roma_pair_metrics.csv"
    combined_match_path = reports / "combined_roma_pairwise_matches.h5"
    combined_feature_path = reports / "combined_roma_features.h5"
    combined_pair_path = reports / "combined_cross_pairs.txt"
    database_path = colmap / "database.db"
    sparse_path = colmap / "sparse"

    if args.stage in {"candidates", "all"}:
        seeds, existing = seed_pairs()
        beg_pairs = expansion_pairs(seeds, existing)
        roma.write_pair_file(beg_pairs, beg_pairs_path)
        candidate_summary = {
            "method": "D_R_BEG",
            "seed_database": str(DR_DATABASE),
            "verified_dr_seed_pairs_ge30": len(seeds),
            "existing_dr_cross_pairs": len(existing),
            "new_beg_candidate_pairs": len(beg_pairs),
            "uav_radius": UAV_RADIUS,
            "phone_radius": PHONE_RADIUS,
            "max_new_per_uav": MAX_NEW_PER_UAV,
            "max_new_per_phone": MAX_NEW_PER_PHONE,
            "matching_images": str(ENHANCED_IMAGE_DIR),
        }
        (reports / "pair_expansion_summary.json").write_text(
            json.dumps(candidate_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(candidate_summary, ensure_ascii=False, indent=2))

    beg_pairs = read_pair_file(beg_pairs_path)
    if args.stage in {"match", "all"}:
        if beg_match_path.exists() or beg_metrics_path.exists():
            raise FileExistsError(
                "BEG match output already exists; use a later --stage to resume."
            )
        original_image_dir = roma.IMAGE_DIR
        roma.IMAGE_DIR = ENHANCED_IMAGE_DIR
        roma.read_rgb.cache_clear()
        try:
            match_summary = roma.match_pairs(
                beg_pairs,
                beg_match_path,
                beg_metrics_path,
                args.coarse_res,
                args.samples_per_pair,
                MIN_INLIERS,
                args.ransac_threshold,
            )
        finally:
            roma.read_rgb.cache_clear()
            roma.IMAGE_DIR = original_image_dir
        (reports / "beg_matching_summary.json").write_text(
            json.dumps(match_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.stage in {"combine", "all"}:
        if combined_match_path.exists() or combined_feature_path.exists():
            raise FileExistsError(
                "Combined match/features already exist; use a later --stage to resume."
            )
        merge_summary = merge_match_files(
            DR_MATCHES, beg_match_path, combined_match_path
        )
        accepted = accepted_pairs(beg_metrics_path)
        seed_pairs_list = read_pair_file(DR_PAIR_FILE)
        accepted_beg_pairs = [
            pair for pair in beg_pairs if pair in accepted
        ]
        # A retried pair must use the enhanced RoMa result only once.
        accepted_keys = {
            roma.names_to_pair(name0, name1)
            for name0, name1 in accepted_beg_pairs
        }
        combined_pairs = [
            pair
            for pair in seed_pairs_list
            if roma.names_to_pair(*pair) not in accepted_keys
        ] + accepted_beg_pairs
        roma.write_pair_file(combined_pairs, combined_pair_path)
        roma.aggregate_pairwise_matches(
            combined_pairs,
            combined_match_path,
            combined_feature_path,
            args.max_keypoints,
            args.merge_error,
            args.cell_size,
        )
        merge_summary.update(
            {
                "seed_pair_count": len(seed_pairs_list),
                "accepted_beg_pair_count": len(accepted_beg_pairs),
                "combined_pair_count": len(combined_pairs),
                "combined_leaf_group_count": len(
                    pair_group_names(combined_match_path)
                ),
            }
        )
        (reports / "combined_matches_summary.json").write_text(
            json.dumps(merge_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.stage in {"database", "all"}:
        combined_pairs = read_pair_file(combined_pair_path)
        if database_path.exists():
            raise FileExistsError(
                "D_R_BEG database already exists; use --stage mapper to resume."
            )
        database_summary = roma.prepare_database(
            combined_pairs,
            combined_pair_path,
            combined_match_path,
            combined_feature_path,
            database_path,
            args.reuse_existing_radius,
        )
        (reports / "database_summary.json").write_text(
            json.dumps(database_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.stage in {"mapper", "all"}:
        combined_pairs = read_pair_file(combined_pair_path)
        if sparse_path.exists():
            raise FileExistsError("Sparse output already exists; refusing overwrite.")
        roma.verify_and_map(
            database_path,
            combined_pair_path,
            sparse_path,
            logs,
            MIN_INLIERS,
            args.ransac_threshold,
        )

    config = {
        "method": "D_R_BEG",
        **vars(args),
        "output_root": str(output),
        "fixed_intra_source_database": str(roma.BASE_DATABASE),
        "dr_seed_database": str(DR_DATABASE),
        "reused_dr_matches": str(DR_MATCHES),
        "bridge_enhanced_images": str(ENHANCED_IMAGE_DIR),
        "matcher": "RoMa outdoor",
        "candidate_retrieval": "DINOv2 Top-20 plus conservative BEG expansion",
    }
    (reports / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
