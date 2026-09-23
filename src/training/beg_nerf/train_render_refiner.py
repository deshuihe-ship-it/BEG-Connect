#!/usr/bin/env python3
"""Train a compact residual U-Net on cached BEG-NeRF v2 render/GT pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
OUT = WORK / "render_refiner"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_refiner_model import ResidualRenderUNet  # noqa: E402


def read_rgb(path: str) -> torch.Tensor:
    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)


def read_mask(path: str) -> torch.Tensor:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy((image > 127).astype(np.float32))[None]


class PairStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def sample_patch(self, row: dict, size: int, focus_bridge: bool) -> tuple[torch.Tensor, ...]:
        inp, target, mask = read_rgb(row["input"]), read_rgb(row["target"]), read_mask(row["mask"])
        _, height, width = inp.shape
        if focus_bridge and mask.any():
            ys, xs = torch.nonzero(mask[0] > 0.5, as_tuple=True)
            selected = random.randrange(len(xs))
            cy, cx = int(ys[selected]), int(xs[selected])
            y0 = max(0, min(height - size, cy - random.randrange(size)))
            x0 = max(0, min(width - size, cx - random.randrange(size)))
        else:
            y0 = random.randrange(height - size + 1)
            x0 = random.randrange(width - size + 1)
        return tuple(value[:, y0:y0 + size, x0:x0 + size] for value in (inp, target, mask))

    def batch(self, batch_size: int, patch_size: int) -> tuple[torch.Tensor, ...]:
        items = []
        for index in range(batch_size):
            row = random.choice(self.rows)
            items.append(self.sample_patch(row, patch_size, focus_bridge=index < round(0.65 * batch_size)))
        return tuple(torch.stack([item[i] for item in items]) for i in range(3))


def edge_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    kernel_x = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(-1, -2)
    kernels_x, kernels_y = kernel_x.repeat(3, 1, 1, 1), kernel_y.repeat(3, 1, 1, 1)
    pred_x = F.conv2d(pred, kernels_x, padding=1, groups=3)
    pred_y = F.conv2d(pred, kernels_y, padding=1, groups=3)
    gt_x = F.conv2d(target, kernels_x, padding=1, groups=3)
    gt_y = F.conv2d(target, kernels_y, padding=1, groups=3)
    return (((pred_x - gt_x).abs() + (pred_y - gt_y).abs()) * weight).mean()


@torch.no_grad()
def validation_score(model: nn.Module, rows: list[dict], device: torch.device) -> dict:
    model.eval()
    psnr, ssim = [], []
    for row in rows:
        inp, target = read_rgb(row["input"])[None].to(device), read_rgb(row["target"])[None].to(device)
        pred = model(inp)
        mse = F.mse_loss(pred, target)
        psnr.append(float((-10 * torch.log10(mse.clamp_min(1e-10))).item()))
        ssim.append(float(structural_similarity_index_measure(pred, target, data_range=1.0).item()))
    return {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cache = json.loads((OUT / "cache_manifest.json").read_text())
    rows = sorted((row for row in cache["rows"] if row["split"] == "train"), key=lambda row: row["name"])
    val_names = set()
    for source in ("UAV", "PHONE"):
        source_rows = [row for row in rows if row["source"] == source]
        val_names.update(row["name"] for row in source_rows[::max(1, len(source_rows) // 15)][:15])
    train_rows = [row for row in rows if row["name"] not in val_names]
    val_rows = [row for row in rows if row["name"] in val_names]
    if len(train_rows) + len(val_rows) != len(rows) or len(val_rows) != 30:
        raise RuntimeError(f"Unexpected internal split: train={len(train_rows)}, val={len(val_rows)}")

    device = torch.device("cuda")
    model = ResidualRenderUNet().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    for parameter in lpips.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=2e-5)
    scaler = torch.cuda.amp.GradScaler()
    store = PairStore(train_rows)
    model_dir = OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    history, best_score = [], float("-inf")

    for step in range(1, args.steps + 1):
        model.train()
        inp, target, mask = (value.to(device, non_blocking=True) for value in store.batch(args.batch_size, args.patch_size))
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            pred = model(inp)
            weight = 1.0 + mask
            rgb = (((pred - target) ** 2) * weight).mean()
            dssim = 1.0 - structural_similarity_index_measure(pred, target, data_range=1.0)
            perceptual = lpips(pred, target)
            edge = edge_loss(pred, target, weight)
            identity = (pred - inp).abs().mean()
            loss = rgb + 0.10 * dssim + 0.02 * perceptual + 0.03 * edge + 0.005 * identity
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()

        if step == 1 or step % 100 == 0:
            print(f"step={step}/{args.steps} loss={loss.item():.6f} rgb={rgb.item():.6f} "
                  f"dssim={dssim.item():.6f} lpips={perceptual.item():.6f} lr={scheduler.get_last_lr()[0]:.7f}", flush=True)
        if step % 1000 == 0 or step == args.steps:
            metrics = validation_score(model, val_rows, device)
            score = metrics["psnr"] + 5.0 * metrics["ssim"]
            history.append({"step": step, **metrics, "score": score})
            print(f"validation step={step} psnr={metrics['psnr']:.6f} ssim={metrics['ssim']:.6f}", flush=True)
            state = {"model": model.state_dict(), "step": step, "metrics": metrics,
                     "width": 24, "residual_scale": 0.12}
            torch.save(state, model_dir / "latest.pt")
            if score > best_score:
                best_score = score
                torch.save(state, model_dir / "best.pt")

    summary = {"train_count": len(train_rows), "validation_count": len(val_rows),
               "test_count": cache["test_count"], "steps": args.steps, "seed": args.seed,
               "best_score": best_score, "history": history}
    (OUT / "training.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
