from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get('COMPACT_BRIDGE_IMAGE_DIR', 'UNSET_AUTHORIZED_IMAGE_DIR'))
TASK_ROOT = PACKAGE_ROOT / 'outputs/mask_generation_v3_1'
DEFAULT_MASK_DIR = TASK_ROOT / 'masks'
DEFAULT_OVERLAY_DIR = TASK_ROOT / 'overlays'
DEFAULT_QC_CSV = TASK_ROOT / 'quality_check.csv'
DEFAULT_CONTACT_DIR = TASK_ROOT / 'contact_sheets'


def frame_num(path: Path) -> int:
    return int(path.stem.split('_')[-1])


def segment_for_frame(n: int) -> str:
    if n <= 105:
        return 'uva1_far'
    if n <= 331:
        return 'uva2_far'
    if n <= 379:
        return 'phone_early'
    if n <= 550:
        return 'phone_close_mostly_bridge'
    return 'phone_late'


def polygon_mask(shape: tuple[int, int], points_norm: list[tuple[float, float]]) -> np.ndarray:
    h, w = shape
    pts = np.array(points_norm, dtype=np.float32)
    pts[:, 0] *= w
    pts[:, 1] *= h
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask


def search_roi(shape: tuple[int, int], segment: str) -> np.ndarray:
    if segment == 'uva1_far':
        pts = [(0.20, 0.37), (0.36, 0.29), (0.55, 0.28), (0.73, 0.36), (0.80, 0.50), (0.73, 0.69), (0.56, 0.78), (0.35, 0.75), (0.17, 0.62)]
    elif segment == 'uva2_far':
        pts = [(0.19, 0.36), (0.35, 0.28), (0.56, 0.27), (0.75, 0.36), (0.82, 0.51), (0.74, 0.70), (0.55, 0.80), (0.34, 0.76), (0.16, 0.63)]
    elif segment == 'phone_early':
        pts = [(0.05, 0.35), (0.22, 0.20), (0.46, 0.14), (0.71, 0.24), (0.91, 0.43), (0.89, 0.69), (0.64, 0.88), (0.30, 0.84), (0.07, 0.65)]
    elif segment == 'phone_close_mostly_bridge':
        pts = [(0.00, 0.08), (1.00, 0.08), (1.00, 1.00), (0.00, 1.00)]
    else:
        pts = [(0.00, 0.14), (1.00, 0.12), (1.00, 1.00), (0.00, 1.00)]
    return polygon_mask(shape, pts)


def core_roi(shape: tuple[int, int], segment: str) -> np.ndarray:
    if segment == 'uva1_far':
        pts = [(0.31, 0.42), (0.45, 0.35), (0.61, 0.40), (0.68, 0.55), (0.56, 0.66), (0.39, 0.64)]
    elif segment == 'uva2_far':
        pts = [(0.30, 0.41), (0.46, 0.34), (0.63, 0.40), (0.70, 0.56), (0.56, 0.68), (0.38, 0.65)]
    elif segment == 'phone_early':
        pts = [(0.20, 0.42), (0.42, 0.29), (0.68, 0.38), (0.76, 0.60), (0.56, 0.76), (0.28, 0.70)]
    elif segment == 'phone_close_mostly_bridge':
        pts = [(0.08, 0.25), (0.92, 0.22), (0.96, 0.88), (0.06, 0.92)]
    else:
        pts = [(0.08, 0.27), (0.92, 0.24), (0.96, 0.90), (0.06, 0.94)]
    return polygon_mask(shape, pts)


def texture_map(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return cv2.GaussianBlur(np.abs(lap), (11, 11), 0)


def class_masks(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(image_bgr)
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    maxc = np.maximum.reduce([r, g, b]).astype(np.int16)
    minc = np.minimum.reduce([r, g, b]).astype(np.int16)
    spread = maxc - minc
    tex = texture_map(image_bgr)
    yy = np.linspace(0, 1, image_bgr.shape[0], dtype=np.float32)[:, None]

    green = ((h >= 33) & (h <= 95) & (s > 34)) | ((g16 > r16 + 16) & (g16 > b16 + 10) & (s > 24))
    yellow_float = (h >= 20) & (h <= 42) & (s > 45) & (v > 45)
    red_road = (((h <= 12) | (h >= 168)) & (s > 28) & (v > 38)) | ((r16 > g16 + 18) & (r16 > b16 + 16) & (v > 42))
    neutral = (s < 82) & (spread < 78) & (v > 48)
    gray_bridge = neutral & ~(green | yellow_float | red_road)

    smooth_bright = (s < 45) & (v > 120) & (tex < 13)
    smooth_mid = (s < 58) & (v > 65) & (tex < 8)
    top_sky = smooth_bright & (yy < 0.32)
    lower_water = smooth_mid & (yy > 0.66)

    return {
        'gray_bridge': gray_bridge.astype(np.uint8) * 255,
        'green': green.astype(np.uint8) * 255,
        'yellow_float': yellow_float.astype(np.uint8) * 255,
        'red_road': red_road.astype(np.uint8) * 255,
        'top_sky': top_sky.astype(np.uint8) * 255,
        'lower_water': lower_water.astype(np.uint8) * 255,
    }


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            out[labels == label] = 255
    return out


def fill_region_holes(mask: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    out = np.zeros_like(mask)
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
            cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out


def expand_to_region(mask: np.ndarray, segment: str) -> np.ndarray:
    if segment in ('phone_close_mostly_bridge', 'phone_late'):
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    else:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = fill_region_holes(mask)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, smooth_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, smooth_kernel)
    mask = fill_region_holes(mask)
    return mask


def keep_components_touching_core(mask: np.ndarray, core: np.ndarray, min_area_px: int, min_core_overlap: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    core_bool = core > 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        comp = labels == label
        overlap = int(np.logical_and(comp, core_bool).sum())
        if overlap >= min_core_overlap:
            out[comp] = 255
    return out


def subtract(mask: np.ndarray, *negatives: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for negative in negatives:
        out[negative > 0] = 0
    return out


def make_bridge_mask(image_bgr: np.ndarray, n: int) -> tuple[np.ndarray, str, str]:
    h, w = image_bgr.shape[:2]
    segment = segment_for_frame(n)
    roi = search_roi((h, w), segment)
    core = core_roi((h, w), segment)
    classes = class_masks(image_bgr)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mid_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    if segment in ('phone_close_mostly_bridge', 'phone_late'):
        mask = roi.copy()
        neg = cv2.bitwise_or(classes['green'], classes['yellow_float'])
        neg = cv2.bitwise_or(neg, classes['red_road'])
        neg = cv2.bitwise_or(neg, classes['top_sky'])
        neg = cv2.bitwise_or(neg, classes['lower_water'])
        neg = cv2.dilate(neg, small_kernel, iterations=1)
        mask = subtract(mask, neg)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = keep_components_touching_core(mask, core, int(h * w * 0.01), int(h * w * 0.002))
    else:
        mask = cv2.bitwise_and(classes['gray_bridge'], roi)
        neg = cv2.bitwise_or(classes['green'], classes['yellow_float'])
        neg = cv2.bitwise_or(neg, classes['red_road'])
        neg = cv2.bitwise_or(neg, classes['top_sky'])
        neg = cv2.bitwise_or(neg, classes['lower_water'])
        mask = subtract(mask, cv2.dilate(neg, small_kernel, iterations=1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mid_kernel)
        mask = keep_components_touching_core(mask, core, int(h * w * 0.001), 30)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, mid_kernel)

    mask = expand_to_region(mask, segment)
    mask = remove_small_components(mask, max(80, int(h * w * 0.001)))
    area_ratio = float((mask > 0).mean())
    status = 'valid'
    if area_ratio < 0.01:
        status = 'invalid_too_small'
    elif segment not in ('phone_close_mostly_bridge', 'phone_late') and area_ratio > 0.42:
        status = 'warning_too_large'
    elif segment in ('phone_close_mostly_bridge', 'phone_late') and area_ratio > 0.91:
        status = 'warning_nearly_full_frame'
    elif segment == 'phone_close_mostly_bridge' and area_ratio < 0.35:
        status = 'warning_close_view_low_bridge_area'
    return mask, segment, status


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, status: str, segment: str, name: str) -> np.ndarray:
    color = np.zeros_like(image_bgr)
    color[..., 1] = mask
    overlay = cv2.addWeighted(image_bgr, 0.72, color, 0.28, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 38), (255, 255, 255), -1)
    text = f'{name} | {segment} | {status} | area={(mask > 0).mean():.3f}'
    cv2.putText(overlay, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return overlay


def contact_sheet(image_paths: list[Path], overlay_dir: Path, out_path: Path, thumb_w: int = 320) -> None:
    thumbs = []
    for image_path in image_paths:
        im = cv2.imread(str(overlay_dir / f'{image_path.stem}_overlay.jpg'), cv2.IMREAD_COLOR)
        if im is None:
            continue
        scale = thumb_w / im.shape[1]
        thumbs.append(cv2.resize(im, (thumb_w, int(im.shape[0] * scale)), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    h, w = thumbs[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * h
        x = (idx % cols) * w
        sheet[y:y + h, x:x + w] = thumb
    cv2.imwrite(str(out_path), sheet)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate V3.1 bridge masks with slightly larger region tolerance.')
    parser.add_argument('--image-dir', type=Path, default=DATA_ROOT)
    parser.add_argument('--mask-dir', type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument('--overlay-dir', type=Path, default=DEFAULT_OVERLAY_DIR)
    parser.add_argument('--qc-csv', type=Path, default=DEFAULT_QC_CSV)
    parser.add_argument('--contact-dir', type=Path, default=DEFAULT_CONTACT_DIR)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Pass --image-dir or set COMPACT_BRIDGE_IMAGE_DIR to an authorized image folder: {args.image_dir}")
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    args.contact_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(args.image_dir.glob('*.jpg'))
    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    rows = []
    for idx, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f'Cannot read image: {image_path}')
        n = frame_num(image_path)
        mask, segment, status = make_bridge_mask(image_bgr, n)
        mask_hash = hashlib.md5((mask > 0).astype(np.uint8).tobytes()).hexdigest()
        area_ratio = float((mask > 0).mean())
        cv2.imwrite(str(args.mask_dir / f'{image_path.stem}.png'), mask)
        cv2.imwrite(str(args.overlay_dir / f'{image_path.stem}_overlay.jpg'), overlay_mask(image_bgr, mask, status, segment, image_path.stem))
        rows.append({'frame': n, 'name': image_path.name, 'segment': segment, 'status': status, 'area_ratio': f'{area_ratio:.6f}', 'mask_hash': mask_hash})
        if idx % 50 == 0:
            print(f'Processed {idx}/{len(image_paths)} V3.1 bridge masks')

    hash_counts: dict[str, int] = {}
    for row in rows:
        hash_counts[row['mask_hash']] = hash_counts.get(row['mask_hash'], 0) + 1
    for row in rows:
        row['exact_duplicate_count'] = hash_counts[row['mask_hash']]
        if int(row['exact_duplicate_count']) > 1 and row['status'] == 'valid':
            row['status'] = 'warning_exact_duplicate'

    with args.qc_csv.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'name', 'segment', 'status', 'area_ratio', 'exact_duplicate_count', 'mask_hash'])
        writer.writeheader()
        writer.writerows(rows)

    samples = {
        'uva1_sample.jpg': [1, 20, 40, 60, 80, 100],
        'uva2_sample.jpg': [120, 160, 200, 240, 280, 320],
        'phone_early_sample.jpg': [332, 340, 350, 360, 370, 379],
        'phone_close_380_450_sample.jpg': [380, 390, 405, 420, 435, 450],
        'phone_close_451_550_sample.jpg': [455, 475, 495, 515, 535, 550],
        'phone_late_sample.jpg': [555, 565, 575, 585, 595, 600],
    }
    by_num = {frame_num(p): p for p in image_paths}
    for out_name, nums in samples.items():
        contact_sheet([by_num[n] for n in nums if n in by_num], args.overlay_dir, args.contact_dir / out_name)

    print(f'Wrote V3.1 masks to {args.mask_dir}')
    print(f'Wrote V3.1 overlays to {args.overlay_dir}')
    print(f'Wrote V3.1 QC to {args.qc_csv}')
    print(f'Wrote V3.1 contact sheets to {args.contact_dir}')


if __name__ == '__main__':
    main()
