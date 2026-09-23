#!/usr/bin/env python3
"""Evaluate full images and exact-mask bridge crops for the latest BEG-NeRF v2 run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.utils.eval_utils import eval_setup
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import structural_similarity_index_measure

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
MANIFEST = WORK / "config/exact_manifest.json"


def metrics(gt, pred):
    gt = gt.permute(2, 0, 1)[None]
    pred = pred.permute(2, 0, 1)[None]
    mse = torch.mean((gt - pred) ** 2)
    return (
        float((-10 * torch.log10(torch.clamp(mse, min=1e-10))).item()),
        float(structural_similarity_index_measure(pred, gt, data_range=1.0)),
        float(learned_perceptual_image_patch_similarity(pred, gt, net_type="alex", normalize=True)),
    )


def main() -> None:
    latest = json.loads((WORK / "results/latest_run.json").read_text())
    manifest = json.loads(MANIFEST.read_text())
    run_dir = Path(latest["run_dir"])
    train_config_path = run_dir / "config.yml"
    config = yaml.load(train_config_path.read_text(), Loader=yaml.Loader)
    config.load_dir = run_dir / "nerfstudio_models"
    config.load_step = int(latest["expected_fine_tune_final_step"])
    config.load_checkpoint = None
    config_path = WORK / "results/eval_config.yml"
    config_path.write_text(yaml.dump(config))
    _, pipeline, checkpoint, step = eval_setup(config_path)
    dataset = pipeline.datamanager.eval_dataset
    loader = getattr(pipeline.datamanager, "fixed_indices_eval_dataloader", None)
    if loader is None:
        loader = FixedIndicesEvalDataloader(dataset, pipeline.device, num_workers=0)
    full, crop, rows = [], [], []
    with torch.no_grad():
        for camera, batch in loader:
            idx = int(batch["image_idx"])
            name = dataset.image_filenames[idx].name
            row = manifest["frames"][name]
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            pred = pipeline.model.get_outputs_for_camera(camera)["rgb"]
            gt = pipeline.model.renderer_rgb.blend_background(batch["image"].to(pipeline.device))
            full_metric = metrics(gt, pred)
            scaled = cv2.resize(mask, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.nonzero(scaled > 127)
            margin = max(1, round(30 * pred.shape[1] / mask.shape[1]))
            x0, x1 = max(0, int(xs.min()) - margin), min(pred.shape[1], int(xs.max()) + 1 + margin)
            y0, y1 = max(0, int(ys.min()) - margin), min(pred.shape[0], int(ys.max()) + 1 + margin)
            crop_metric = metrics(gt[y0:y1, x0:x1], pred[y0:y1, x0:x1])
            full.append(full_metric); crop.append(crop_metric)
            rows.append({"image_name": name, "source_name": row["source_name"],
                         "full_psnr": full_metric[0], "full_ssim": full_metric[1], "full_lpips": full_metric[2],
                         "crop_psnr": crop_metric[0], "crop_ssim": crop_metric[1], "crop_lpips": crop_metric[2]})
    mean = lambda values, i: float(np.mean([value[i] for value in values]))
    result = {
        "run_dir": latest["run_dir"], "checkpoint": str(checkpoint), "step": int(step), "num_eval_images": len(rows),
        "full_image_metrics": {"psnr": mean(full, 0), "ssim": mean(full, 1), "lpips": mean(full, 2)},
        "bridge_crop_metrics": {"psnr": mean(crop, 0), "ssim": mean(crop, 1), "lpips": mean(crop, 2), "margin_px_original": 30},
        "per_image": rows,
    }
    out = WORK / "results/evaluation.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
