from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image import structural_similarity_index_measure

from nerfstudio.utils.eval_utils import eval_setup


DATA_ROOT = Path('/home/oldwater/data/UVA+PHONE')
OUTPUT_ROOT = Path('/home/oldwater/outputs/UVA+PHONE')
MASK_DIR = Path(os.environ.get('COMPACT_BRIDGE_MASK_DIR', 'UNSET_AUTHORIZED_MASK_DIR'))
EVAL_ROOT = DATA_ROOT / 'combined_source_split_bridge_crop_eval'

_LPIPS_METRIC: LearnedPerceptualImagePatchSimilarity | None = None


@dataclass(frozen=True)
class EvalSpec:
    experiment_name: str
    label: str
    method_dir: str = 'nerfacto'


SPECS = [
    EvalSpec('combined_uav_only_train_all_eval_nerfacto', 'UAV-only train, combined eval'),
    EvalSpec('combined_phone_only_train_all_eval_nerfacto', 'PHONE-only train, combined eval'),
    EvalSpec('baseline_vocab_tree', 'UAV+PHONE COLMAP-NeRF baseline'),
    EvalSpec('bridge_sparse_depth_supervised_v1_grouped_bcat_dloss_0004', 'BEG-NeRF full', 'depth-nerfacto'),
]


def source_from_image_name(image_name: str) -> str:
    number = int(''.join(ch for ch in Path(image_name).stem if ch.isdigit()))
    return 'UAV' if number <= 331 else 'PHONE'


def latest_config_path(spec: EvalSpec) -> Path:
    exp_dir = OUTPUT_ROOT / spec.experiment_name / spec.method_dir
    runs = sorted([path for path in exp_dir.iterdir() if path.is_dir() and (path / 'config.yml').exists()])
    if not runs:
        raise FileNotFoundError(f'No trained config found for {spec.experiment_name}')
    return runs[-1] / 'config.yml'


def load_mask(image_name: str) -> torch.Tensor:
    mask_path = MASK_DIR / f'{Path(image_name).stem}.png'
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Missing bridge mask for {image_name}')
    return torch.from_numpy(mask > 127)


def expanded_box_from_mask(mask: torch.Tensor, margin: int) -> tuple[int, int, int, int]:
    ys, xs = torch.where(mask)
    x0 = max(0, int(xs.min().item()) - margin)
    x1 = min(mask.shape[1], int(xs.max().item()) + 1 + margin)
    y0 = max(0, int(ys.min().item()) - margin)
    y1 = min(mask.shape[0], int(ys.max().item()) + 1 + margin)
    return x0, y0, x1, y1


def scale_box(box: tuple[int, int, int, int], src_width: int, src_height: int, dst_width: int, dst_height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    sx = dst_width / src_width
    sy = dst_height / src_height
    scaled_x0 = max(0, min(dst_width - 1, int(round(x0 * sx))))
    scaled_y0 = max(0, min(dst_height - 1, int(round(y0 * sy))))
    scaled_x1 = max(scaled_x0 + 1, min(dst_width, int(round(x1 * sx))))
    scaled_y1 = max(scaled_y0 + 1, min(dst_height, int(round(y1 * sy))))
    return scaled_x0, scaled_y0, scaled_x1, scaled_y1


def crop_to_box(image: torch.Tensor, box: tuple[int, int, int, int]) -> torch.Tensor:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1, :]


def image_to_bchw(image: torch.Tensor) -> torch.Tensor:
    return torch.moveaxis(image, -1, 0)[None, ...]


def get_lpips_metric(device: torch.device) -> LearnedPerceptualImagePatchSimilarity:
    global _LPIPS_METRIC
    if _LPIPS_METRIC is None:
        _LPIPS_METRIC = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
        _LPIPS_METRIC.eval()
    elif next(_LPIPS_METRIC.parameters()).device != device:
        _LPIPS_METRIC = _LPIPS_METRIC.to(device)
    return _LPIPS_METRIC


def crop_metrics(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor) -> tuple[float, float, float]:
    gt_bchw = image_to_bchw(gt_rgb)
    pred_bchw = image_to_bchw(pred_rgb)
    mse = torch.mean((gt_bchw - pred_bchw) ** 2)
    psnr = (-10.0 * torch.log10(torch.clamp(mse, min=1e-10))).item()
    ssim = float(structural_similarity_index_measure(pred_bchw, gt_bchw, data_range=1.0))
    lpips_metric = get_lpips_metric(pred_bchw.device)
    lpips = float(lpips_metric(pred_bchw, gt_bchw))
    return psnr, ssim, lpips


def evaluate_spec(spec: EvalSpec, margin: int) -> list[dict[str, object]]:
    config_path = latest_config_path(spec)
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    dataset = pipeline.datamanager.eval_dataset
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        total = len(dataset.image_filenames)
        for idx, (camera, batch) in enumerate(dataloader, start=1):
            image_idx = int(batch['image_idx'])
            image_name = dataset.image_filenames[image_idx].name
            mask = load_mask(image_name)
            base_box = expanded_box_from_mask(mask, margin)
            outputs = pipeline.model.get_outputs_for_camera(camera)
            pred_rgb = outputs['rgb']
            gt_rgb = pipeline.model.renderer_rgb.blend_background(batch['image'].to(pipeline.device))
            pred_h, pred_w = pred_rgb.shape[:2]
            crop_box = scale_box(base_box, int(mask.shape[1]), int(mask.shape[0]), pred_w, pred_h)
            gt_crop = crop_to_box(gt_rgb, crop_box)
            pred_crop = crop_to_box(pred_rgb, crop_box)
            psnr, ssim, lpips = crop_metrics(gt_crop, pred_crop)
            rows.append({
                'method': spec.label,
                'experiment_name': spec.experiment_name,
                'config_path': str(config_path),
                'checkpoint': str(checkpoint_path),
                'step': step,
                'image_idx': image_idx,
                'image_name': image_name,
                'source': source_from_image_name(image_name),
                'crop_psnr': psnr,
                'crop_ssim': ssim,
                'crop_lpips': lpips,
            })
            if idx % 10 == 0 or idx == total:
                print(f'[{spec.experiment_name}] {idx}/{total} eval images complete', flush=True)
    return rows


def summarize(per_image_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, method_df in per_image_df.groupby('method'):
        for scope, scope_df in [('ALL', method_df), ('UAV', method_df[method_df['source'] == 'UAV']), ('PHONE', method_df[method_df['source'] == 'PHONE'])]:
            rows.append({
                'method': method,
                'eval_scope': scope,
                'num_eval_images': len(scope_df),
                'crop_psnr': float(scope_df['crop_psnr'].mean()),
                'crop_ssim': float(scope_df['crop_ssim'].mean()),
                'crop_lpips': float(scope_df['crop_lpips'].mean()),
            })
    return pd.DataFrame(rows)


def write_markdown(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        '# Combined-Source-Split Bridge Crop Eval Summary',
        '',
        '| method | eval scope | num eval images | crop_psnr | crop_ssim | crop_lpips |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary_df.to_dict('records'):
        lines.append(
            f"| {row['method']} | {row['eval_scope']} | {int(row['num_eval_images'])} | {float(row['crop_psnr']):.4f} | {float(row['crop_ssim']):.4f} | {float(row['crop_lpips']):.4f} |"
        )
    path.write_text('\n'.join(lines))


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for spec in SPECS:
        all_rows.extend(evaluate_spec(spec, margin=30))
    per_image_df = pd.DataFrame(all_rows)
    summary_df = summarize(per_image_df)
    per_image_df.to_csv(EVAL_ROOT / 'combined_source_split_bridge_crop_per_image.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(EVAL_ROOT / 'combined_source_split_bridge_crop_summary.csv', index=False, quoting=csv.QUOTE_MINIMAL)
    with pd.ExcelWriter(EVAL_ROOT / 'combined_source_split_bridge_crop_summary.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        per_image_df.to_excel(writer, sheet_name='per_image', index=False)
    write_markdown(EVAL_ROOT / 'combined_source_split_bridge_crop_summary.md', summary_df)
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
