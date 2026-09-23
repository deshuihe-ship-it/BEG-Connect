#!/usr/bin/env python3
"""Render exact train/eval views from BEG-NeRF v2 and cache PNG pairs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from nerfstudio.utils.eval_utils import eval_setup

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
OUT = WORK / "render_refiner"
MANIFEST_PATH = WORK / "config/exact_manifest.json"


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def main() -> None:
    latest = json.loads((WORK / "results/latest_run.json").read_text())
    exact = json.loads(MANIFEST_PATH.read_text())
    run_dir = Path(latest["run_dir"])
    config = yaml.load((run_dir / "config.yml").read_text(), Loader=yaml.Loader)
    config.load_dir = run_dir / "nerfstudio_models"
    config.load_step = int(latest["expected_fine_tune_final_step"])
    config.load_checkpoint = None
    eval_config = OUT / "cache_eval_config.yml"
    OUT.mkdir(parents=True, exist_ok=True)
    eval_config.write_text(yaml.dump(config))
    _, pipeline, checkpoint, step = eval_setup(eval_config)

    rows = []
    for split, dataset in (("train", pipeline.datamanager.train_dataset),
                           ("test", pipeline.datamanager.eval_dataset)):
        for subdir in ("input", "target", "mask"):
            (OUT / "cache" / split / subdir).mkdir(parents=True, exist_ok=True)
        for index in range(len(dataset)):
            name = dataset.image_filenames[index].name
            if name not in exact["frames"]:
                raise KeyError(f"Missing exact source/mask mapping for {name}")
            input_path = OUT / "cache" / split / "input" / f"{Path(name).stem}.png"
            target_path = OUT / "cache" / split / "target" / f"{Path(name).stem}.png"
            mask_path = OUT / "cache" / split / "mask" / f"{Path(name).stem}.png"
            if not (input_path.exists() and target_path.exists() and mask_path.exists()):
                camera = dataset.cameras[index:index + 1].to(pipeline.device)
                batch = dataset[index]
                with torch.no_grad():
                    pred = pipeline.model.get_outputs_for_camera(camera)["rgb"]
                gt = pipeline.model.renderer_rgb.blend_background(batch["image"].to(pipeline.device))
                source_mask = cv2.imread(exact["frames"][name]["mask_path"], cv2.IMREAD_GRAYSCALE)
                if source_mask is None:
                    raise FileNotFoundError(exact["frames"][name]["mask_path"])
                mask = cv2.resize(source_mask, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                save_rgb(input_path, pred)
                save_rgb(target_path, gt)
                cv2.imwrite(str(mask_path), mask)
            rows.append({
                "split": split,
                "name": name,
                "source": exact["frames"][name]["source"],
                "input": str(input_path),
                "target": str(target_path),
                "mask": str(mask_path),
            })
            print(f"cached {split} {index + 1}/{len(dataset)} {name}", flush=True)

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "train_count": sum(row["split"] == "train" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "rows": rows,
    }
    (OUT / "cache_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
