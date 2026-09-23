#!/usr/bin/env python3
"""Warm-start Nerfacto with bridge/boundary/global patch sampling and patch losses."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Int
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

from nerfstudio.data.pixel_samplers import PixelSampler
from nerfstudio.engine.trainer import Trainer
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.scripts.train import main as train_main

WORK = Path(os.environ.get("BEG_NERF_WORKSPACE", "UNSET_BEG_NERF_WORKSPACE"))
BASE_CONFIG = Path(os.environ.get("BEG_NERF_BASE_CONFIG", "UNSET_BASE_CONFIG"))
BASE_CHECKPOINTS = BASE_CONFIG.parent / "nerfstudio_models"
SAMPLER_CONFIG = WORK / "config/bridge_patch_sampler.json"
PATCH_SIZE = 32
PATCH_LAYOUT = ("bridge", "bridge", "boundary", "global")


def _candidate_centers(mask_path: str, height: int, width: int, kind: str, patch_size: int) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.empty((0, 2), dtype=np.int32)
    binary = cv2.resize((mask > 127).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    half = patch_size // 2
    valid = np.zeros_like(binary)
    valid[half : height - half, half : width - half] = 1
    if kind == "bridge":
        kernel = np.ones((max(3, patch_size // 2), max(3, patch_size // 2)), np.uint8)
        region = cv2.erode(binary, kernel)
        if not region.any():
            region = binary
    else:
        kernel = np.ones((max(3, patch_size // 2), max(3, patch_size // 2)), np.uint8)
        region = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
    candidates = np.argwhere((region > 0) & (valid > 0))
    return candidates[::2].astype(np.int32)


def bridge_patch_indices(self: PixelSampler, batch, batch_size: int, num_images: int,
                         image_height: int, image_width: int, device="cpu") -> Optional[Int[Tensor, "batch_size 3"]]:
    if self.bcat_config is None or self.bcat_config.get("mode") != "bridge_patch_v2":
        return ORIGINAL_BCAT(self, batch, batch_size, num_images, image_height, image_width, device)
    patch_size = int(self.bcat_config.get("patch_size", PATCH_SIZE))
    patch_area = patch_size * patch_size
    patch_count = batch_size // patch_area
    layout = list(self.bcat_config.get("patch_layout", PATCH_LAYOUT))
    if patch_count != len(layout):
        raise RuntimeError(f"Expected {len(layout)} patches, got {patch_count}")
    image_idx = batch["image_idx"].detach().cpu().long()
    mask_paths = {int(k): v for k, v in self.bcat_config.get("mask_paths", {}).items()}
    if not hasattr(self, "_beg_v2_candidates"):
        self._beg_v2_candidates = {}

    yy, xx = torch.meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
    parts = []
    valid_local = [i for i, actual in enumerate(image_idx.tolist()) if int(actual) in mask_paths]
    for kind in layout:
        local_idx = random.randrange(num_images)
        if kind == "global" or not valid_local:
            y0 = random.randrange(max(1, image_height - patch_size + 1))
            x0 = random.randrange(max(1, image_width - patch_size + 1))
        else:
            local_idx = random.choice(valid_local)
            actual_idx = int(image_idx[local_idx])
            key = (actual_idx, image_height, image_width, kind)
            if key not in self._beg_v2_candidates:
                self._beg_v2_candidates[key] = _candidate_centers(
                    mask_paths[actual_idx], image_height, image_width, kind, patch_size
                )
            candidates = self._beg_v2_candidates[key]
            if len(candidates):
                cy, cx = candidates[random.randrange(len(candidates))]
                y0, x0 = int(cy) - patch_size // 2, int(cx) - patch_size // 2
            else:
                y0 = random.randrange(max(1, image_height - patch_size + 1))
                x0 = random.randrange(max(1, image_width - patch_size + 1))
        indices = torch.stack([
            torch.full_like(yy, local_idx), yy + y0, xx + x0
        ], dim=-1).reshape(-1, 3)
        parts.append(indices)
    return torch.cat(parts, dim=0).long()


def _edge_loss(pred: Tensor, gt: Tensor) -> Tensor:
    sobel_x = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    channels = pred.shape[1]
    sx = sobel_x.repeat(channels, 1, 1, 1)
    sy = sobel_y.repeat(channels, 1, 1, 1)
    pred_edge = torch.sqrt(F.conv2d(pred, sx, padding=1, groups=channels).square() + F.conv2d(pred, sy, padding=1, groups=channels).square() + 1e-6)
    gt_edge = torch.sqrt(F.conv2d(gt, sx, padding=1, groups=channels).square() + F.conv2d(gt, sy, padding=1, groups=channels).square() + 1e-6)
    return F.l1_loss(pred_edge, gt_edge)


def beg_loss(self: NerfactoModel, outputs, batch, metrics_dict=None):
    losses = ORIGINAL_LOSS(self, outputs, batch, metrics_dict)
    if not self.training or outputs["rgb"].shape[0] % (PATCH_SIZE * PATCH_SIZE):
        return losses
    image = batch["image"].to(self.device)
    pred, gt = self.renderer_rgb.blend_background_for_loss_computation(
        pred_image=outputs["rgb"], pred_accumulation=outputs["accumulation"], gt_image=image
    )
    pred = pred.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 3, 1, 2)
    gt = gt.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3).permute(0, 3, 1, 2)
    focus_pred, focus_gt = pred[:3], gt[:3]
    ssim_value = structural_similarity_index_measure(focus_pred, focus_gt, data_range=1.0)
    losses["beg_dssim_loss"] = 0.10 * (1.0 - ssim_value)
    lp_pred = F.interpolate(focus_pred, size=(64, 64), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    lp_gt = F.interpolate(focus_gt, size=(64, 64), mode="bilinear", align_corners=False).clamp(0.0, 1.0)
    losses["beg_lpips_loss"] = 0.02 * self.lpips(lp_pred, lp_gt)
    losses["beg_edge_loss"] = 0.05 * _edge_loss(focus_pred, focus_gt)
    return losses


ORIGINAL_BCAT = PixelSampler._bcat_sample_indices
ORIGINAL_LOSS = NerfactoModel.get_loss_dict
ORIGINAL_LOAD_CHECKPOINT = Trainer._load_checkpoint


def load_base_weights_only(self: Trainer) -> None:
    """Warm start the model while intentionally resetting optimizers for low-LR fine-tuning."""
    if self.config.load_dir != BASE_CHECKPOINTS:
        ORIGINAL_LOAD_CHECKPOINT(self)
        return
    checkpoint = BASE_CHECKPOINTS / "step-000014999.ckpt"
    state = torch.load(checkpoint, map_location="cpu")
    self.pipeline.load_pipeline(state["pipeline"], state["step"])
    self._start_step = 0
    print(f"Loaded model weights only from {checkpoint}; optimizers reset for BEG-NeRF v2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--additional-iterations", type=int, default=5000)
    parser.add_argument("--experiment", default="daqiao_DLBEG_BEG_NeRF_v2")
    args = parser.parse_args()
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"Set BEG_NERF_BASE_CONFIG or use the Compact Bridge wrapper: {BASE_CONFIG}")
    if not SAMPLER_CONFIG.exists():
        raise FileNotFoundError(f"Run prepare_beg_nerf_v2.py first: {SAMPLER_CONFIG}")

    os.environ["BCAT_SAMPLER_CONFIG"] = str(SAMPLER_CONFIG)
    PixelSampler._bcat_sample_indices = bridge_patch_indices
    NerfactoModel.get_loss_dict = beg_loss
    Trainer._load_checkpoint = load_base_weights_only

    config = yaml.load(BASE_CONFIG.read_text(), Loader=yaml.Loader)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config.output_dir = WORK / "runs"
    config.experiment_name = args.experiment
    config.timestamp = timestamp
    config.max_num_iterations = args.additional_iterations
    config.load_dir = BASE_CHECKPOINTS
    config.load_step = 14999
    config.load_scheduler = False
    config.pipeline.datamanager.patch_size = PATCH_SIZE
    config.pipeline.datamanager.train_num_rays_per_batch = PATCH_SIZE * PATCH_SIZE * len(PATCH_LAYOUT)
    config.steps_per_eval_batch = 500
    config.steps_per_save = 1000
    config.optimizers["fields"]["optimizer"].lr = 0.001
    config.optimizers["fields"]["scheduler"].max_steps = args.additional_iterations
    config.optimizers["fields"]["scheduler"].lr_final = 0.0001
    config.optimizers["proposal_networks"]["optimizer"].lr = 0.001
    config.optimizers["proposal_networks"]["scheduler"].max_steps = args.additional_iterations
    config.optimizers["proposal_networks"]["scheduler"].lr_final = 0.0001
    config.viewer.quit_on_train_completion = True
    train_main(config)
    run_dir = config.get_base_dir()
    result = {
        "run_dir": str(run_dir),
        "base_checkpoint_step": 14999,
        "fine_tune_start_step": 0,
        "additional_iterations": args.additional_iterations,
        "expected_fine_tune_final_step": args.additional_iterations - 1,
        "timestamp": timestamp,
    }
    (WORK / "results").mkdir(parents=True, exist_ok=True)
    (WORK / "results/latest_run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
