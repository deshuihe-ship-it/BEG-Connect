#!/usr/bin/env python3
"""DINOv2-candidate + RoMa cross-source reconstruction for the daqiao dataset.

The experiment reuses the exact DINOv2 Top-20 UAV-PHONE candidate pairs from
the existing D_L run. Intra-source matches remain fixed to the D_L database;
only cross-source two-view matches are replaced with RoMa.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path("/home/oldwater/data/daqiao/uav1+phone11280")
IMAGE_DIR = ROOT / "images"
SOURCE_PAIRS = ROOT / "all_dinov2_lightglue/reports/all_candidate_pairs.txt"
BASE_DATABASE = ROOT / "all_dinov2_lightglue/colmap/database.db"
DEFAULT_OUTPUT = ROOT / "all_dinov2_roma_coarse560"
ROMA_REPO = Path("/home/oldwater/tools/RoMa")
ROMA_WEIGHTS = Path("/home/oldwater/.cache/torch/hub/checkpoints/roma_outdoor.pth")
DINO_WEIGHTS = Path(
    "/home/oldwater/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth"
)
COLMAP = Path("/home/oldwater/tools/colmap-cuda/install/bin/colmap")
MAX_IMAGE_ID = 2_147_483_647

sys.path.insert(0, str(ROMA_REPO))
from romatch.models.model_zoo.roma_models import roma_model  # noqa: E402

from hloc.match_dense import aggregate_matches, assign_matches  # noqa: E402
from hloc.utils.io import get_matches  # noqa: E402
from hloc.utils.parsers import names_to_pair  # noqa: E402


def source(name: str) -> str:
    if "/uav_" in name or name.startswith("uav_"):
        return "uav"
    if "/phone" in name or name.startswith("phone"):
        return "phone"
    return "unknown"


def is_cross_pair(name0: str, name1: str) -> bool:
    return {source(name0), source(name1)} == {"uav", "phone"}


def pair_id(image_id0: int, image_id1: int) -> int:
    if image_id0 > image_id1:
        image_id0, image_id1 = image_id1, image_id0
    return image_id0 * MAX_IMAGE_ID + image_id1


def load_cross_pairs(max_pairs: int | None = None) -> list[tuple[str, str]]:
    pairs = []
    with SOURCE_PAIRS.open("r", encoding="utf-8") as handle:
        for line in handle:
            name0, name1 = line.split()
            if is_cross_pair(name0, name1):
                pairs.append((name0, name1))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def write_pair_file(pairs: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{name0} {name1}\n" for name0, name1 in pairs),
        encoding="utf-8",
    )


@lru_cache(maxsize=48)
def read_rgb(name: str) -> Image.Image:
    with Image.open(IMAGE_DIR / name) as image:
        return image.convert("RGB")


def build_roma(device: torch.device, coarse_res: int):
    torch.set_float32_matmul_precision("highest")
    roma_weights = torch.load(ROMA_WEIGHTS, map_location="cpu")
    dino_weights = torch.load(DINO_WEIGHTS, map_location="cpu")
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=False,
        device=device,
        weights=roma_weights,
        dinov2_weights=dino_weights,
        amp_dtype=torch.float16,
        use_custom_corr=False,
        symmetric=False,
        upsample_res=None,
        sample_mode="threshold_balanced",
    ).eval()
    del roma_weights, dino_weights
    return model


def match_pairs(
    pairs: list[tuple[str, str]],
    match_path: Path,
    metrics_path: Path,
    coarse_res: int,
    samples_per_pair: int,
    min_inliers: int,
    ransac_threshold: float,
) -> dict[str, float | int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full RoMa matching.")
    device = torch.device("cuda")
    model = build_roma(device, coarse_res)
    torch.cuda.reset_peak_memory_stats()

    existing = set()
    if match_path.exists():
        with h5py.File(match_path, "r") as fd:
            existing = {name for name in fd.keys()}

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metrics_path.exists()
    metrics_handle = metrics_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_handle,
        fieldnames=[
            "index",
            "name0",
            "name1",
            "sampled",
            "ransac_inliers",
            "inlier_ratio",
            "accepted",
            "seconds",
        ],
    )
    if write_header:
        writer.writeheader()

    accepted = 0
    total_inliers = 0
    total_seconds = 0.0
    with h5py.File(match_path, "a") as fd:
        for index, (name0, name1) in enumerate(pairs, start=1):
            pair_name = names_to_pair(name0, name1)
            if pair_name in fd:
                group = fd[pair_name]
                inliers = int(group.attrs.get("ransac_inliers", 0))
                accepted += int(inliers >= min_inliers)
                total_inliers += inliers
                continue

            started = time.perf_counter()
            image0 = read_rgb(name0)
            image1 = read_rgb(name1)
            torch.manual_seed(index)
            warp, certainty = model.match(image0, image1, device=device)
            matches, scores = model.sample(
                warp, certainty, num=samples_per_pair
            )
            keypoints0, keypoints1 = model.to_pixel_coordinates(
                matches,
                image0.height,
                image0.width,
                image1.height,
                image1.width,
            )
            keypoints0 = keypoints0.cpu().numpy().astype(np.float32)
            keypoints1 = keypoints1.cpu().numpy().astype(np.float32)
            scores_np = scores.cpu().numpy().astype(np.float32)

            fundamental, mask = cv2.findFundamentalMat(
                keypoints0,
                keypoints1,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=ransac_threshold,
                confidence=0.999,
                maxIters=10000,
            )
            if fundamental is None or mask is None:
                keep = np.zeros(len(keypoints0), dtype=bool)
            else:
                keep = mask.reshape(-1).astype(bool)
            inliers = int(keep.sum())
            is_accepted = inliers >= min_inliers
            if not is_accepted:
                keep[:] = False

            group = fd.create_group(pair_name)
            group.create_dataset("keypoints0", data=keypoints0[keep])
            group.create_dataset("keypoints1", data=keypoints1[keep])
            group.create_dataset("scores", data=scores_np[keep])
            group.attrs["sampled"] = int(len(keypoints0))
            group.attrs["ransac_inliers"] = inliers
            group.attrs["accepted"] = int(is_accepted)

            elapsed = time.perf_counter() - started
            accepted += int(is_accepted)
            total_inliers += inliers
            total_seconds += elapsed
            writer.writerow(
                {
                    "index": index,
                    "name0": name0,
                    "name1": name1,
                    "sampled": len(keypoints0),
                    "ransac_inliers": inliers,
                    "inlier_ratio": inliers / max(len(keypoints0), 1),
                    "accepted": int(is_accepted),
                    "seconds": elapsed,
                }
            )
            metrics_handle.flush()
            if index % 20 == 0:
                fd.flush()
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(pairs)}",
                            "accepted": accepted,
                            "inliers": total_inliers,
                            "last_seconds": round(elapsed, 3),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    metrics_handle.close()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2
    del model
    torch.cuda.empty_cache()
    return {
        "pairs": len(pairs),
        "accepted_pairs": accepted,
        "total_ransac_inliers": total_inliers,
        "new_matching_seconds": total_seconds,
        "peak_gpu_memory_mb": peak_memory,
    }


def aggregate_pairwise_matches(
    pairs: list[tuple[str, str]],
    match_path: Path,
    feature_path: Path,
    max_keypoints: int,
    merge_error: float,
    cell_size: int,
) -> None:
    if feature_path.exists():
        feature_path.unlink()
    required = {name for pair in pairs for name in pair}
    conf = {"max_error": merge_error, "cell_size": cell_size}
    keypoints = aggregate_matches(
        conf,
        pairs,
        match_path,
        feature_path,
        required_queries=required,
        max_kps=max_keypoints,
    )
    assign_matches(pairs, match_path, keypoints, max_error=merge_error)


def decode_pair_id(value: int) -> tuple[int, int]:
    image_id1 = value % MAX_IMAGE_ID
    image_id0 = (value - image_id1) // MAX_IMAGE_ID
    return int(image_id0), int(image_id1)


def prepare_database(
    pairs: list[tuple[str, str]],
    pair_file: Path,
    match_path: Path,
    feature_path: Path,
    database_path: Path,
    reuse_existing_radius: float,
) -> dict[str, int]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_DATABASE, database_path)
    with sqlite3.connect(database_path) as connection:
        image_ids = {
            str(name): int(image_id)
            for image_id, name in connection.execute(
                "SELECT image_id, name FROM images"
            )
        }

        cross_pair_ids = []
        for (value,) in connection.execute("SELECT pair_id FROM matches"):
            image_id0, image_id1 = decode_pair_id(int(value))
            name0 = next(
                (name for name, image_id in image_ids.items() if image_id == image_id0),
                "",
            )
            name1 = next(
                (name for name, image_id in image_ids.items() if image_id == image_id1),
                "",
            )
            if is_cross_pair(name0, name1):
                cross_pair_ids.append(int(value))
        connection.executemany(
            "DELETE FROM matches WHERE pair_id=?", ((value,) for value in cross_pair_ids)
        )
        connection.executemany(
            "DELETE FROM two_view_geometries WHERE pair_id=?",
            ((value,) for value in cross_pair_ids),
        )

        keypoint_maps = {}
        reused_keypoints = 0
        appended_keypoints = 0
        with h5py.File(feature_path, "r") as features:
            for name, image_id in image_ids.items():
                row = connection.execute(
                    "SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                    (image_id,),
                ).fetchone()
                if row is None:
                    old = np.empty((0, 6), dtype=np.float32)
                else:
                    old = np.frombuffer(row[2], dtype=np.float32).reshape(
                        int(row[0]), int(row[1])
                    )
                    if old.shape[1] != 6:
                        converted = np.zeros((len(old), 6), dtype=np.float32)
                        converted[:, : min(old.shape[1], 6)] = old[:, :6]
                        old = converted
                if name not in features:
                    continue
                new_xy = np.asarray(features[name]["keypoints"], dtype=np.float32)
                new_xy_colmap = new_xy + 0.5
                mapping = np.empty(len(new_xy), dtype=np.uint32)
                reuse = np.zeros(len(new_xy), dtype=bool)
                if reuse_existing_radius > 0 and len(old):
                    distance, nearest = cKDTree(old[:, :2]).query(
                        new_xy_colmap, k=1
                    )
                    reuse = distance <= reuse_existing_radius
                    mapping[reuse] = nearest[reuse].astype(np.uint32)
                append_indices = np.flatnonzero(~reuse)
                mapping[append_indices] = (
                    len(old) + np.arange(len(append_indices), dtype=np.uint32)
                )
                keypoint_maps[name] = mapping
                reused_keypoints += int(reuse.sum())
                appended_keypoints += int(len(append_indices))
                new = np.zeros((len(append_indices), 6), dtype=np.float32)
                new[:, :2] = new_xy_colmap[append_indices]
                merged = np.vstack([old, new]).astype(np.float32)
                connection.execute(
                    "INSERT OR REPLACE INTO keypoints(image_id, rows, cols, data) "
                    "VALUES(?, ?, 6, ?)",
                    (image_id, len(merged), merged.tobytes()),
                )

        inserted_pairs = 0
        inserted_matches = 0
        for name0, name1 in pairs:
            matches, _ = get_matches(match_path, name0, name1)
            if len(matches) == 0:
                continue
            image_id0, image_id1 = image_ids[name0], image_ids[name1]
            adjusted = np.column_stack(
                (
                    keypoint_maps[name0][matches[:, 0]],
                    keypoint_maps[name1][matches[:, 1]],
                )
            ).astype(np.uint32)
            adjusted = np.unique(adjusted, axis=0)
            if image_id0 > image_id1:
                adjusted = adjusted[:, [1, 0]]
            value = pair_id(image_id0, image_id1)
            connection.execute(
                "INSERT OR REPLACE INTO matches(pair_id, rows, cols, data) "
                "VALUES(?, ?, 2, ?)",
                (value, len(adjusted), adjusted.tobytes()),
            )
            inserted_pairs += 1
            inserted_matches += len(adjusted)
        connection.commit()

    write_pair_file(pairs, pair_file)
    return {
        "removed_cross_pairs": len(cross_pair_ids),
        "inserted_roma_pairs": inserted_pairs,
        "inserted_roma_matches": inserted_matches,
        "reused_existing_keypoints": reused_keypoints,
        "appended_roma_keypoints": appended_keypoints,
    }


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT)


def verify_and_map(
    database_path: Path,
    pair_file: Path,
    sparse_path: Path,
    logs_path: Path,
    min_inliers: int,
    ransac_threshold: float,
) -> None:
    run_command(
        [
            str(COLMAP),
            "geometric_verifier",
            "--database_path",
            str(database_path),
            "--TwoViewGeometry.min_num_inliers",
            str(min_inliers),
            "--TwoViewGeometry.max_error",
            str(ransac_threshold),
            "--TwoViewGeometry.confidence",
            "0.999",
            "--TwoViewGeometry.max_num_trials",
            "10000",
            "--TwoViewGeometry.min_inlier_ratio",
            "0.0",
        ],
        logs_path / "geometric_verifier.log",
    )
    sparse_path.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(COLMAP),
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(IMAGE_DIR),
            "--output_path",
            str(sparse_path),
            "--Mapper.min_num_matches",
            "15",
            "--Mapper.ba_refine_focal_length",
            "1",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_local_max_num_iterations",
            "10",
            "--Mapper.ba_global_max_num_iterations",
            "50",
            "--Mapper.ba_global_max_refinements",
            "5",
            "--Mapper.ba_global_frames_freq",
            "400",
            "--Mapper.ba_global_points_freq",
            "200000",
        ],
        logs_path / "mapper.log",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=["match", "aggregate", "database", "mapper", "all"],
        default="all",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--coarse-res", type=int, default=560)
    parser.add_argument("--samples-per-pair", type=int, default=2048)
    parser.add_argument("--min-inliers", type=int, default=30)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--max-keypoints", type=int, default=8192)
    parser.add_argument("--merge-error", type=float, default=2.0)
    parser.add_argument("--cell-size", type=int, default=2)
    parser.add_argument("--reuse-existing-radius", type=float, default=0.0)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    reports = output_root / "reports"
    logs = output_root / "logs"
    colmap_root = output_root / "colmap"
    match_path = reports / "roma_pairwise_matches.h5"
    feature_path = reports / "roma_features.h5"
    metrics_path = reports / "roma_pair_metrics.csv"
    pair_file = reports / "dino_cross_pairs.txt"
    database_path = colmap_root / "database.db"
    sparse_path = colmap_root / "sparse"
    reports.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    pairs = load_cross_pairs(args.max_pairs)
    write_pair_file(pairs, pair_file)
    config = {
        **vars(args),
        "output_root": str(output_root),
        "image_dir": str(IMAGE_DIR),
        "source_pairs": str(SOURCE_PAIRS),
        "base_database": str(BASE_DATABASE),
        "cross_pair_count": len(pairs),
        "roma_weights": str(ROMA_WEIGHTS),
        "dino_weights": str(DINO_WEIGHTS),
        "intra_source_backbone": "existing D_L intra-source matches",
        "cross_source_matcher": "RoMa outdoor",
        "upsample_predictions": False,
        "symmetric_inference": False,
    }
    config["output_root"] = str(output_root)
    (reports / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary = {}
    if args.stage in {"match", "all"}:
        summary.update(
            match_pairs(
                pairs,
                match_path,
                metrics_path,
                args.coarse_res,
                args.samples_per_pair,
                args.min_inliers,
                args.ransac_threshold,
            )
        )
    if args.stage in {"aggregate", "all"}:
        aggregate_pairwise_matches(
            pairs,
            match_path,
            feature_path,
            args.max_keypoints,
            args.merge_error,
            args.cell_size,
        )
    if args.stage in {"database", "all"}:
        summary.update(
            prepare_database(
                pairs,
                pair_file,
                match_path,
                feature_path,
                database_path,
                args.reuse_existing_radius,
            )
        )
    if args.stage in {"mapper", "all"}:
        verify_and_map(
            database_path,
            pair_file,
            sparse_path,
            logs,
            args.min_inliers,
            args.ransac_threshold,
        )
    (reports / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
