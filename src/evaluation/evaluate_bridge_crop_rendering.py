from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
import torch
from torchmetrics.functional.image import learned_perceptual_image_patch_similarity
from torchmetrics.functional.image import structural_similarity_index_measure

from nerfstudio.utils.eval_utils import eval_setup


DATA_ROOT = Path('/home/oldwater/data/UVA+PHONE')
OUTPUT_ROOT = Path('/home/oldwater/outputs/UVA+PHONE')
MASK_DIR = Path(os.environ.get('COMPACT_BRIDGE_MASK_DIR', 'UNSET_AUTHORIZED_MASK_DIR'))
EVAL_ROOT = DATA_ROOT / 'bridge_region_render_eval'


@dataclass(frozen=True)
class EvalSpec:
    experiment_name: str
    label: str


DEFAULT_SPECS = [
    EvalSpec('baseline_vocab_tree', 'baseline_vocab_tree'),
    EvalSpec('bridge_guided_progressive_v1', 'bridge_guided_progressive_v1'),
    EvalSpec('bridge_enhanced_then_bridge_guided_v1', 'bridge_enhanced_then_bridge_guided_v1'),
    EvalSpec('graph_balanced_cross_source_matching_v1', 'graph_balanced_cross_source_matching_v1'),
    EvalSpec('scheme2_bridge_enhanced_images', 'scheme2_bridge_enhanced_images'),
]


def latest_config_path(experiment_name: str) -> Path | None:
    candidate_dirs = [OUTPUT_ROOT / experiment_name / 'nerfacto', OUTPUT_ROOT / experiment_name / 'depth-nerfacto']
    for exp_dir in candidate_dirs:
        if not exp_dir.exists():
            continue
        runs = sorted([path for path in exp_dir.iterdir() if path.is_dir() and (path / 'config.yml').exists()])
        if runs:
            return runs[-1] / 'config.yml'
    return None


def load_mask(image_name: str) -> torch.Tensor:
    mask_path = MASK_DIR / f'{Path(image_name).stem}.png'
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Missing bridge mask for {image_name}: {mask_path}')
    return torch.from_numpy(mask > 127)


def expanded_box_from_mask(mask: torch.Tensor, margin: int) -> tuple[int, int, int, int]:
    ys, xs = torch.where(mask)
    if len(xs) == 0:
        raise ValueError('Bridge mask is empty.')
    x0 = max(0, int(xs.min().item()) - margin)
    x1 = min(mask.shape[1], int(xs.max().item()) + 1 + margin)
    y0 = max(0, int(ys.min().item()) - margin)
    y1 = min(mask.shape[0], int(ys.max().item()) + 1 + margin)
    return x0, y0, x1, y1


def scale_box(
    box: tuple[int, int, int, int],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> tuple[int, int, int, int]:
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


def crop_metrics(gt_rgb: torch.Tensor, pred_rgb: torch.Tensor) -> tuple[float, float, float]:
    gt_bchw = image_to_bchw(gt_rgb)
    pred_bchw = image_to_bchw(pred_rgb)
    mse = torch.mean((gt_bchw - pred_bchw) ** 2)
    psnr = (-10.0 * torch.log10(torch.clamp(mse, min=1e-10))).item()
    ssim = float(structural_similarity_index_measure(pred_bchw, gt_bchw, data_range=1.0))
    lpips = float(
        learned_perceptual_image_patch_similarity(pred_bchw, gt_bchw, net_type='alex', normalize=True)
    )
    return psnr, ssim, lpips


def evaluate_spec(spec: EvalSpec, margin: int) -> tuple[list[dict[str, object]], dict[str, object]] | None:
    config_path = latest_config_path(spec.experiment_name)
    if config_path is None:
        print(f'Skipping {spec.label}: no trained config found.')
        return None

    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    dataset = pipeline.datamanager.eval_dataset
    dataloader = pipeline.datamanager.fixed_indices_eval_dataloader

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for camera, batch in dataloader:
            image_idx = int(batch['image_idx'])
            image_name = dataset.image_filenames[image_idx].name
            mask = load_mask(image_name)
            base_box = expanded_box_from_mask(mask, margin)

            outputs = pipeline.model.get_outputs_for_camera(camera)
            pred_rgb = outputs['rgb']
            gt_rgb = pipeline.model.renderer_rgb.blend_background(batch['image'].to(pipeline.device))

            pred_h, pred_w = pred_rgb.shape[:2]
            mask_h, mask_w = int(mask.shape[0]), int(mask.shape[1])
            crop_box = scale_box(base_box, mask_w, mask_h, pred_w, pred_h)
            gt_crop = crop_to_box(gt_rgb, crop_box)
            pred_crop = crop_to_box(pred_rgb, crop_box)
            psnr, ssim, lpips = crop_metrics(gt_crop, pred_crop)

            rows.append(
                {
                    'method': spec.label,
                    'experiment_name': spec.experiment_name,
                    'config_path': str(config_path),
                    'checkpoint': str(checkpoint_path),
                    'step': step,
                    'image_idx': image_idx,
                    'image_name': image_name,
                    'crop_margin_px_original': margin,
                    'crop_x0': crop_box[0],
                    'crop_y0': crop_box[1],
                    'crop_x1': crop_box[2],
                    'crop_y1': crop_box[3],
                    'crop_width': crop_box[2] - crop_box[0],
                    'crop_height': crop_box[3] - crop_box[1],
                    'crop_psnr': psnr,
                    'crop_ssim': ssim,
                    'crop_lpips': lpips,
                }
            )

    summary = {
        'method': spec.label,
        'experiment_name': spec.experiment_name,
        'num_eval_images': len(rows),
        'crop_margin_px_original': margin,
        'crop_psnr': float(pd.DataFrame(rows)['crop_psnr'].mean()),
        'crop_ssim': float(pd.DataFrame(rows)['crop_ssim'].mean()),
        'crop_lpips': float(pd.DataFrame(rows)['crop_lpips'].mean()),
    }
    print(json.dumps(summary, indent=2))
    return rows, summary


def write_markdown(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        '# Bridge Crop Eval Summary',
        '',
        '| method | num_eval_images | crop_psnr | crop_ssim | crop_lpips |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summary_df.to_dict('records'):
        lines.append(
            f"| {row['method']} | {int(row['num_eval_images'])} | {float(row['crop_psnr']):.4f} | {float(row['crop_ssim']):.4f} | {float(row['crop_lpips']):.4f} |"
        )
    path.write_text('\n'.join(lines))


def main() -> None:
    margin = 30
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for spec in DEFAULT_SPECS:
        result = evaluate_spec(spec, margin)
        if result is None:
            continue
        rows, summary = result
        all_rows.extend(rows)
        summaries.append(summary)

    per_image_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(summaries).sort_values('crop_psnr', ascending=False)

    per_image_path = EVAL_ROOT / 'bridge_crop_eval_per_image.csv'
    summary_csv_path = EVAL_ROOT / 'bridge_crop_eval_summary.csv'
    summary_xlsx_path = EVAL_ROOT / 'bridge_crop_eval_summary.xlsx'
    summary_md_path = EVAL_ROOT / 'bridge_crop_eval_summary.md'

    per_image_df.to_csv(per_image_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with pd.ExcelWriter(summary_xlsx_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        per_image_df.to_excel(writer, sheet_name='per_image', index=False)
    write_markdown(summary_md_path, summary_df)

    print(f'Wrote bridge crop eval to {EVAL_ROOT}')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
