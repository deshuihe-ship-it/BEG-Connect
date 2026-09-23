#!/usr/bin/env python3
"""Controlled Table 15 evaluation: one fixed, held-out 50-view set.

The legacy D-L split has no image-content overlap with the other three runs.
This script preserves their existing checkpoints, retrains only D-L using the
fixed 350/50 split, and evaluates all four methods on the same 50 source images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/oldwater/miniconda3/envs/nerfstudio/bin/python")
NS_EVAL = Path("/home/oldwater/miniconda3/envs/nerfstudio/bin/ns-eval")

CASES = {
    "D-L": {
        "data": Path("/home/oldwater/data/daqiao/uav1+phone11280/all_dinov2_lightglue/ns_processed"),
        "config": Path("/home/oldwater/outputs/uav1+phone1_1280_DL/config.yml"),
        "checkpoint": Path("/home/oldwater/outputs/uav1+phone1_1280_DL/nerfstudio_models/step-000014999.ckpt"),
        "split": "fraction",
    },
    "D-L-BEG": {
        "data": Path("/home/oldwater/data/daqiao/uav1+phone11280/all_dinov2_lightglue_bridge_enhance_expand/ns_processed"),
        "config": Path("/home/oldwater/outputs/uav1_phone1_1280_baseline_dl_beg/DL_BEG/nerfacto/2026-07-12_155733/config.yml"),
        "checkpoint": Path("/home/oldwater/outputs/uav1_phone1_1280_baseline_dl_beg/DL_BEG/nerfacto/2026-07-12_155733/nerfstudio_models/step-000014999.ckpt"),
        "split": "interval",
    },
    "D-R": {
        "data": Path("/home/oldwater/data/daqiao/uav1+phone11280/all_dinov2_roma_coarse560_reuse2/ns_processed"),
        "config": Path("/home/oldwater/outputs/uav1_phone1_1280_dino_roma/dino_roma_nerfacto_15000/nerfacto/2026-07-24_140553/config.yml"),
        "checkpoint": Path("/home/oldwater/outputs/uav1_phone1_1280_dino_roma/dino_roma_nerfacto_15000/nerfacto/2026-07-24_140553/nerfstudio_models/step-000014999.ckpt"),
        "split": "interval",
    },
    "D-R-BEG": {
        "data": Path("/home/oldwater/data/daqiao/uav1+phone11280/all_dinov2_roma_bridge_enhance_expand/ns_processed"),
        "config": Path("/home/oldwater/outputs/uav1_phone1_1280_dino_roma_beg/d_r_beg_nerfacto_15000/nerfacto/2026-07-26_174115/config.yml"),
        "checkpoint": Path("/home/oldwater/outputs/uav1_phone1_1280_dino_roma_beg/d_r_beg_nerfacto_15000/nerfacto/2026-07-26_174115/nerfstudio_models/step-000014999.ckpt"),
        "split": "interval",
    },
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sorted_frames(data: Path) -> list[dict]:
    frames = json.loads((data / "transforms.json").read_text())["frames"]
    return sorted(frames, key=lambda row: row["file_path"])


def heldout_hashes(name: str, info: dict) -> dict[str, dict]:
    frames = sorted_frames(info["data"])
    if info["split"] == "fraction":
        n_train = math.ceil(len(frames) * 0.9)
        train_idx = set(np.linspace(0, len(frames) - 1, n_train, dtype=int))
        indices = [idx for idx in range(len(frames)) if idx not in train_idx]
    else:
        indices = list(range(0, len(frames), 8))
    result = {}
    for idx in indices:
        frame = frames[idx]
        digest = sha256(info["data"] / frame["file_path"])
        if digest in result:
            raise RuntimeError(f"Duplicate image content in {name}: {frame['file_path']}")
        result[digest] = frame
    return result


def patch_eval_config(source: Path, destination: Path, dataset: Path) -> None:
    """Point an existing model config at a fixed-split evaluation proxy."""
    lines = source.read_text().splitlines(keepends=True)
    result = []
    idx = 0
    replaced = False
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("      data: !!python/object/apply:pathlib.PosixPath") and not replaced:
            result.append("      data: !!python/object/apply:pathlib.PosixPath\n")
            result.extend(f"      - {part}\n" for part in dataset.resolve().parts)
            idx += 1
            while idx < len(lines) and lines[idx].startswith("      - "):
                idx += 1
            replaced = True
            continue
        if line.startswith("      eval_num_rays_per_chunk:"):
            result.append("      eval_num_rays_per_chunk: 2048\n")
        else:
            result.append(line)
        idx += 1
    if not replaced:
        raise RuntimeError(f"Could not patch dataparser path in {source}")
    destination.write_text("".join(result))


def prepare() -> None:
    if (ROOT / "manifest.json").exists():
        print("Already prepared:", ROOT / "manifest.json")
        return
    for info in CASES.values():
        for path in (info["data"] / "transforms.json", info["config"], info["checkpoint"]):
            if not path.exists():
                raise FileNotFoundError(path)
    heldout = {name: heldout_hashes(name, info) for name, info in CASES.items()}
    common = set.intersection(*(set(rows) for name, rows in heldout.items() if name != "D-L"))
    if len(common) != 50:
        raise RuntimeError(f"Expected 50 common D-L-BEG/D-R/D-R-BEG held-out views, found {len(common)}")
    if common & set(heldout["D-L"]):
        raise RuntimeError("D-L legacy held-out views unexpectedly overlap the common set")
    common = sorted(common)
    manifest = {
        "protocol": "One fixed 50-view held-out set, defined by image-content SHA-256.",
        "legacy_D-L_heldout_count": len(heldout["D-L"]),
        "legacy_D-L_overlap_with_common": len(set(common) & set(heldout["D-L"])),
        "common_heldout_count": len(common),
        "cases": {},
        "common_image_sha256": common,
    }
    for name, info in CASES.items():
        source = info["data"]
        proxy = ROOT / "datasets" / name
        proxy.mkdir(parents=True)
        for folder in ("images", "images_2"):
            target = source / folder
            if target.exists():
                (proxy / folder).symlink_to(target, target_is_directory=True)
        meta = json.loads((source / "transforms.json").read_text())
        by_hash = {sha256(source / row["file_path"]): row for row in meta["frames"]}
        test = [by_hash[digest] for digest in common]
        test_names = {row["file_path"] for row in test}
        meta["train_filenames"] = [row["file_path"] for row in meta["frames"] if row["file_path"] not in test_names]
        meta["val_filenames"] = [row["file_path"] for row in test]
        meta["test_filenames"] = [row["file_path"] for row in test]
        if len(meta["train_filenames"]) != 350 or len(meta["test_filenames"]) != 50:
            raise RuntimeError(f"Unexpected split sizes for {name}")
        (proxy / "transforms.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        if "ply_file_path" in meta and (source / meta["ply_file_path"]).exists():
            (proxy / meta["ply_file_path"]).symlink_to(source / meta["ply_file_path"])
        manifest["cases"][name] = {
            "source_data": str(source),
            "source_config": str(info["config"]),
            "source_checkpoint": str(info["checkpoint"]),
            "source_config_sha256": sha256(info["config"]),
            "dataset_proxy": str(proxy),
            "train_count": 350,
            "test_count": 50,
            "test_file_paths": meta["test_filenames"],
        }
        if name != "D-L":
            patch_eval_config(info["config"], ROOT / "eval_configs" / f"{name}_common50.yml", proxy)
    write_json(ROOT / "manifest.json", manifest)
    print(json.dumps({"common_heldout_count": 50, "legacy_D-L_overlap": 0}, indent=2))


def train_dl() -> None:
    if not (ROOT / "manifest.json").exists():
        raise RuntimeError("Run prepare first")
    marker = ROOT / "results" / "D-L_retrained.json"
    if marker.exists():
        print("D-L retraining is already complete")
        return
    import torch
    import yaml
    from nerfstudio.engine.trainer import Trainer
    from nerfstudio.scripts.train import main as train_main
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the D-L controlled retraining")
    original_iteration = Trainer.train_iteration
    def checked_iteration(self, step):
        output = original_iteration(self, step)
        if not torch.isfinite(output[0]).all():
            raise FloatingPointError(f"Non-finite D-L loss at step {step}")
        if step % 100 == 0:
            write_json(ROOT / "progress.json", {"stage": "D-L_train", "step": step, "loss": float(output[0].detach()), "time": time.time()})
        return output
    Trainer.train_iteration = checked_iteration
    source = CASES["D-L"]["config"]
    config = yaml.load(source.read_text(), Loader=yaml.Loader)
    data = ROOT / "datasets" / "D-L"
    config.data = data
    config.pipeline.datamanager.data = data
    config.pipeline.datamanager.dataparser.data = data
    config.output_dir = ROOT / "runs"
    config.experiment_name = "D-L_common50_retrained"
    config.timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    config.max_num_iterations = 15000
    config.machine.seed = 42
    config.vis = "tensorboard"
    config.viewer.quit_on_train_completion = True
    config.load_dir = None
    config.load_step = None
    config.load_checkpoint = None
    config.load_config = None
    train_main(config)
    run_dir = config.get_base_dir()
    checkpoint = run_dir / "nerfstudio_models" / "step-000014999.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    write_json(marker, {"run_dir": str(run_dir), "config": str(run_dir / "config.yml"), "checkpoint": str(checkpoint), "seed": 42, "steps": 15000, "train_count": 350, "test_count": 50})
    print("D-L training complete:", run_dir)


def evaluate() -> None:
    if not (ROOT / "results" / "D-L_retrained.json").exists():
        raise RuntimeError("D-L retraining must finish before common evaluation")
    configs = {name: ROOT / "eval_configs" / f"{name}_common50.yml" for name in CASES if name != "D-L"}
    configs["D-L"] = Path(json.loads((ROOT / "results" / "D-L_retrained.json").read_text())["config"])
    report = {"protocol": "All values use the same 50 held-out source images; D-L was retrained on the complementary 350 images.", "runs": {}}
    for name in CASES:
        config = configs[name]
        output = ROOT / "results" / f"{name}_common50_ns_eval.json"
        command = [str(NS_EVAL), "--load-config", str(config), "--output-path", str(output)]
        print("RUN", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        data = json.loads(output.read_text())
        results = data.get("results", data)
        report["runs"][name] = {"config": str(config), "result_file": str(output), "metrics": {k: results[k] for k in ("psnr", "ssim", "lpips") if k in results}}
    if any(set(row["metrics"]) != {"psnr", "ssim", "lpips"} for row in report["runs"].values()):
        raise RuntimeError("One or more evaluations did not yield PSNR, SSIM, and LPIPS")
    write_json(ROOT / "results" / "table15_common50_summary.json", report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "train", "evaluate"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "train":
        train_dl()
    else:
        evaluate()


if __name__ == "__main__":
    main()
