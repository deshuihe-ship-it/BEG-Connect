#!/usr/bin/env python3
"""Evaluate the selected render refiner on the untouched 50-view test set."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
OUT = WORK / "render_refiner"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_refiner_model import ResidualRenderUNet  # noqa: E402


def read_rgb(path: str) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def metric(pred: torch.Tensor, gt: torch.Tensor, lpips) -> tuple[float, float, float]:
    mse = F.mse_loss(pred, gt)
    return (float((-10 * torch.log10(mse.clamp_min(1e-10))).item()),
            float(structural_similarity_index_measure(pred, gt, data_range=1.0).item()),
            float(lpips(pred, gt).item()))


def mean_metrics(values: list[tuple[float, float, float]]) -> dict:
    return {"psnr": float(np.mean([x[0] for x in values])),
            "ssim": float(np.mean([x[1] for x in values])),
            "lpips": float(np.mean([x[2] for x in values]))}


def save_comparison(path: Path, inp: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor) -> None:
    images = []
    for image in (inp, pred, gt):
        array = (image[0].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)
        images.append(array)
    canvas = np.concatenate(images, axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main() -> None:
    cache = json.loads((OUT / "cache_manifest.json").read_text())
    rows = sorted((row for row in cache["rows"] if row["split"] == "test"), key=lambda row: row["name"])
    checkpoint = torch.load(OUT / "model/best.pt", map_location="cpu")
    device = torch.device("cuda")
    model = ResidualRenderUNet(checkpoint["width"], checkpoint["residual_scale"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    base_full, refined_full, base_crop, refined_crop, per_image = [], [], [], [], []
    comparison_dir = OUT / "comparison_png"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index, row in enumerate(rows):
            inp = read_rgb(row["input"])[None].to(device)
            gt = read_rgb(row["target"])[None].to(device)
            pred = model(inp)
            mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(mask > 127)
            margin = 15
            x0, x1 = max(0, int(xs.min()) - margin), min(mask.shape[1], int(xs.max()) + 1 + margin)
            y0, y1 = max(0, int(ys.min()) - margin), min(mask.shape[0], int(ys.max()) + 1 + margin)
            bf, rf = metric(inp, gt, lpips), metric(pred, gt, lpips)
            bc = metric(inp[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1], lpips)
            rc = metric(pred[:, :, y0:y1, x0:x1], gt[:, :, y0:y1, x0:x1], lpips)
            base_full.append(bf); refined_full.append(rf); base_crop.append(bc); refined_crop.append(rc)
            per_image.append({"name": row["name"], "source": row["source"],
                              "base_full": bf, "refined_full": rf, "base_crop": bc, "refined_crop": rc})
            if index in (0, 10, 20, 30, 40, 49):
                save_comparison(comparison_dir / f"{Path(row['name']).stem}_input_refined_gt.png", inp, pred, gt)

    result = {
        "checkpoint": str(OUT / "model/best.pt"), "checkpoint_step": checkpoint["step"],
        "num_test_images": len(rows),
        "base": {"full": mean_metrics(base_full), "bridge_crop": mean_metrics(base_crop)},
        "refined": {"full": mean_metrics(refined_full), "bridge_crop": mean_metrics(refined_crop)},
        "per_image": per_image,
    }
    result["delta"] = {
        area: {key: result["refined"][area][key] - result["base"][area][key]
               for key in ("psnr", "ssim", "lpips")}
        for area in ("full", "bridge_crop")
    }
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
