from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary, read_points3D_binary

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PACKAGE_ROOT / 'outputs/bridge_participation_analysis_v3_1'
DEFAULT_MASK_DIR = Path(os.environ.get('COMPACT_BRIDGE_MASK_DIR', 'UNSET_AUTHORIZED_MASK_DIR'))
DEFAULT_OUT_ROOT = TASK_ROOT
MAX_IMAGE_ID = 2147483647

DEFAULT_DATASETS = {
    'baseline_vocab_tree': Path('/home/oldwater/data/UVA+PHONE/combined3_vocab/processed_vocab_tree'),
    'bridge_guided_progressive_v1': Path('/home/oldwater/data/UVA+PHONE/processed_bridge_guided_progressive_v1'),
    'scale_normalized_cross_source_matching_v1': Path('/home/oldwater/data/UVA+PHONE/processed_scale_normalized_v1'),
    'heatmap_matching_v1': Path('/home/oldwater/data/UVA+PHONE/processed_heatmap_matching_v1'),
}


def parse_frame_number(name: str) -> int:
    return int(Path(name).stem.split('_')[-1])


def group_source_for_name(name: str) -> str:
    return 'PHONE' if parse_frame_number(name) >= 332 else 'UAV'


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def load_image_rows(db_path: Path) -> dict[int, dict[str, str | int]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute('SELECT image_id, name FROM images ORDER BY image_id').fetchall()
    finally:
        conn.close()
    return {
        int(image_id): {
            'image_id': int(image_id),
            'name': name,
            'group_source': group_source_for_name(name),
        }
        for image_id, name in rows
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyze bridge participation using V3.1 bridge-region masks.')
    parser.add_argument('--mask-dir', type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        '--dataset',
        action='append',
        default=[],
        help='Optional dataset spec in the form name=/abs/or/rel/path. Can be passed multiple times.',
    )
    return parser.parse_args()


def read_keypoints(db_path: Path) -> dict[int, np.ndarray]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute('SELECT image_id, rows, cols, data FROM keypoints').fetchall()
    finally:
        conn.close()
    out: dict[int, np.ndarray] = {}
    for image_id, rows_n, cols_n, data in rows:
        arr = np.frombuffer(data, dtype=np.float32).reshape((rows_n, cols_n))[:, :2].copy()
        out[int(image_id)] = arr
    return out


def read_two_view_matches(db_path: Path) -> list[tuple[int, int, np.ndarray]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute('SELECT pair_id, rows, cols, data FROM two_view_geometries WHERE rows > 0').fetchall()
    finally:
        conn.close()
    out = []
    for pair_id, rows_n, cols_n, data in rows:
        image_id1, image_id2 = pair_id_to_image_ids(int(pair_id))
        matches = np.frombuffer(data, dtype=np.uint32).reshape((rows_n, cols_n)).copy()
        out.append((image_id1, image_id2, matches))
    return out


def load_masks(mask_dir: Path, image_names: list[str]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name in image_names:
        mask_path = mask_dir / f'{Path(name).stem}.png'
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Missing bridge mask for {name}: {mask_path}')
        masks[name] = mask > 127
    return masks


def sample_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape
    xi = int(np.clip(round(float(x)), 0, w - 1))
    yi = int(np.clip(round(float(y)), 0, h - 1))
    return bool(mask[yi, xi])


def analyze_cross_source_matches(
    image_map: dict[int, dict[str, str | int]],
    keypoints: dict[int, np.ndarray],
    pair_rows: list[tuple[int, int, np.ndarray]],
    masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_records: list[dict[str, object]] = []
    total_pairs = 0
    total_endpoint_obs = 0
    total_bridge_endpoint_obs = 0
    total_any_bridge_pairs = 0
    total_both_bridge_pairs = 0

    for image_id1, image_id2, matches in pair_rows:
        row1 = image_map[image_id1]
        row2 = image_map[image_id2]
        if row1['group_source'] == row2['group_source']:
            continue
        name1 = str(row1['name'])
        name2 = str(row2['name'])
        mask1 = masks[name1]
        mask2 = masks[name2]
        pts1 = keypoints[image_id1][matches[:, 0]]
        pts2 = keypoints[image_id2][matches[:, 1]]

        bridge1 = np.array([sample_mask(mask1, x, y) for x, y in pts1], dtype=bool)
        bridge2 = np.array([sample_mask(mask2, x, y) for x, y in pts2], dtype=bool)
        any_bridge = bridge1 | bridge2
        both_bridge = bridge1 & bridge2

        match_count = int(len(matches))
        endpoint_obs = match_count * 2
        bridge_endpoint_obs = int(bridge1.sum() + bridge2.sum())
        any_bridge_count = int(any_bridge.sum())
        both_bridge_count = int(both_bridge.sum())

        total_pairs += match_count
        total_endpoint_obs += endpoint_obs
        total_bridge_endpoint_obs += bridge_endpoint_obs
        total_any_bridge_pairs += any_bridge_count
        total_both_bridge_pairs += both_bridge_count

        pair_records.append(
            {
                'image_id1': image_id1,
                'name1': name1,
                'image_id2': image_id2,
                'name2': name2,
                'match_count': match_count,
                'bridge_endpoint_observation_count': bridge_endpoint_obs,
                'bridge_endpoint_ratio': bridge_endpoint_obs / endpoint_obs if endpoint_obs else 0.0,
                'any_bridge_match_count': any_bridge_count,
                'any_bridge_match_ratio': any_bridge_count / match_count if match_count else 0.0,
                'both_bridge_match_count': both_bridge_count,
                'both_bridge_match_ratio': both_bridge_count / match_count if match_count else 0.0,
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                'metric': 'cross_source_match_count',
                'value': int(total_pairs),
                'definition': 'Total number of UAV-PHONE two-view correspondences.',
            },
            {
                'metric': 'cross_source_match_endpoint_observation_count',
                'value': int(total_endpoint_obs),
                'definition': 'Two endpoint observations per cross-source match.',
            },
            {
                'metric': 'bridge_match_count',
                'value': int(total_bridge_endpoint_obs),
                'definition': 'Endpoint observations from cross-source matches that fall inside bridge masks.',
            },
            {
                'metric': 'bridge_match_ratio',
                'value': float(total_bridge_endpoint_obs / total_endpoint_obs) if total_endpoint_obs else 0.0,
                'definition': 'Bridge endpoint observations divided by all cross-source match endpoint observations.',
            },
            {
                'metric': 'bridge_match_any_pair_ratio',
                'value': float(total_any_bridge_pairs / total_pairs) if total_pairs else 0.0,
                'definition': 'Cross-source matches with at least one endpoint inside a bridge mask.',
            },
            {
                'metric': 'bridge_match_both_pair_ratio',
                'value': float(total_both_bridge_pairs / total_pairs) if total_pairs else 0.0,
                'definition': 'Cross-source matches with both endpoints inside bridge masks.',
            },
        ]
    )
    pair_df = pd.DataFrame(pair_records).sort_values('both_bridge_match_ratio', ascending=False)
    return summary_df, pair_df


def analyze_sparse_points(
    image_map: dict[int, dict[str, str | int]],
    sparse_dir: Path,
    masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = read_images_binary(sparse_dir / 'images.bin')
    points3d = read_points3D_binary(sparse_dir / 'points3D.bin')

    point_rows: list[dict[str, object]] = []
    total_points = 0
    global_bridge_majority_count = 0
    global_bridge_any_obs_count = 0
    shared_point_count = 0
    shared_bridge_strict_count = 0
    shared_bridge_any_obs_count = 0

    for point_id, point in points3d.items():
        total_points += 1
        obs_image_ids = [int(v) for v in point.image_ids]
        point2d_idxs = [int(v) for v in point.point2D_idxs]
        obs_sources = [str(image_map[image_id]['group_source']) for image_id in obs_image_ids]
        is_shared = ('UAV' in obs_sources) and ('PHONE' in obs_sources)

        bridge_obs_total = 0
        bridge_obs_uav = 0
        bridge_obs_phone = 0
        names = []

        for image_id, point2d_idx in zip(obs_image_ids, point2d_idxs):
            image = images[image_id]
            name = image.name
            names.append(name)
            x, y = image.xys[point2d_idx]
            in_bridge = sample_mask(masks[name], float(x), float(y))
            if in_bridge:
                bridge_obs_total += 1
                if image_map[image_id]['group_source'] == 'UAV':
                    bridge_obs_uav += 1
                else:
                    bridge_obs_phone += 1

        track_length = len(obs_image_ids)
        bridge_obs_ratio = float(bridge_obs_total / track_length) if track_length else 0.0
        global_bridge_majority = bridge_obs_total * 2 >= track_length
        global_bridge_any_obs = bridge_obs_total > 0
        shared_bridge_strict = is_shared and bridge_obs_uav > 0 and bridge_obs_phone > 0
        shared_bridge_any_obs = is_shared and bridge_obs_total > 0

        if global_bridge_majority:
            global_bridge_majority_count += 1
        if global_bridge_any_obs:
            global_bridge_any_obs_count += 1
        if is_shared:
            shared_point_count += 1
        if shared_bridge_strict:
            shared_bridge_strict_count += 1
        if shared_bridge_any_obs:
            shared_bridge_any_obs_count += 1

        point_rows.append(
            {
                'point3D_id': int(point_id),
                'track_length': track_length,
                'category': 'shared' if is_shared else ('uav_only' if set(obs_sources) == {'UAV'} else 'phone_only'),
                'bridge_observation_count': bridge_obs_total,
                'bridge_observation_ratio': bridge_obs_ratio,
                'bridge_observation_uav_count': bridge_obs_uav,
                'bridge_observation_phone_count': bridge_obs_phone,
                'is_bridge_global_majority': int(global_bridge_majority),
                'is_bridge_global_any_obs': int(global_bridge_any_obs),
                'is_bridge_shared_strict': int(shared_bridge_strict),
                'is_bridge_shared_any_obs': int(shared_bridge_any_obs),
                'image_names': ';'.join(names),
            }
        )

    summary_df = pd.DataFrame(
        [
            {
                'metric': 'global_sparse_point_count',
                'value': int(total_points),
                'definition': 'Total sparse 3D point count.',
            },
            {
                'metric': 'bridge_global_sparse_count',
                'value': int(global_bridge_majority_count),
                'definition': 'Sparse 3D points with at least half of observations inside bridge masks.',
            },
            {
                'metric': 'bridge_global_sparse_ratio',
                'value': float(global_bridge_majority_count / total_points) if total_points else 0.0,
                'definition': 'Majority bridge-supported sparse 3D points divided by all sparse 3D points.',
            },
            {
                'metric': 'bridge_global_sparse_any_obs_ratio',
                'value': float(global_bridge_any_obs_count / total_points) if total_points else 0.0,
                'definition': 'Sparse 3D points with at least one bridge observation.',
            },
            {
                'metric': 'shared_point_count',
                'value': int(shared_point_count),
                'definition': 'Sparse 3D points observed by both UAV and PHONE.',
            },
            {
                'metric': 'bridge_shared_point_count',
                'value': int(shared_bridge_strict_count),
                'definition': 'Shared 3D points with at least one bridge observation from UAV and at least one from PHONE.',
            },
            {
                'metric': 'bridge_shared_point_ratio',
                'value': float(shared_bridge_strict_count / shared_point_count) if shared_point_count else 0.0,
                'definition': 'Strict bridge-supported shared 3D points divided by all shared 3D points.',
            },
            {
                'metric': 'bridge_shared_point_any_obs_ratio',
                'value': float(shared_bridge_any_obs_count / shared_point_count) if shared_point_count else 0.0,
                'definition': 'Shared 3D points with at least one bridge observation in either source.',
            },
        ]
    )
    point_df = pd.DataFrame(point_rows).sort_values(['category', 'bridge_observation_ratio', 'track_length'], ascending=[True, False, False])
    return summary_df, point_df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def analyze_dataset(dataset_name: str, root: Path, mask_dir: Path, out_root: Path) -> dict[str, float | int | str]:
    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = root / 'colmap' / 'database.db'
    sparse_dir = root / 'colmap' / 'sparse' / '0'

    image_map = load_image_rows(db_path)
    image_names = [str(v['name']) for v in image_map.values()]
    masks = load_masks(mask_dir, image_names)
    keypoints = read_keypoints(db_path)
    pair_rows = read_two_view_matches(db_path)

    match_summary_df, match_pair_df = analyze_cross_source_matches(image_map, keypoints, pair_rows, masks)
    sparse_summary_df, sparse_point_df = analyze_sparse_points(image_map, sparse_dir, masks)

    summary_lookup = {row['metric']: row['value'] for row in pd.concat([match_summary_df, sparse_summary_df]).to_dict('records')}
    summary_df = pd.DataFrame(
        [
            {
                'metric': 'dataset_name',
                'value': dataset_name,
                'definition': 'Method / dataset identifier.',
            },
            *match_summary_df.to_dict('records'),
            *sparse_summary_df.to_dict('records'),
        ]
    )

    write_csv(summary_df, out_dir / 'bridge_participation_summary.csv')
    write_csv(match_pair_df, out_dir / 'cross_source_match_bridge_breakdown.csv')
    write_csv(sparse_point_df, out_dir / 'sparse_point_bridge_breakdown.csv')

    with pd.ExcelWriter(out_dir / 'bridge_participation_analysis.xlsx', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        match_pair_df.to_excel(writer, sheet_name='cross_source_pairs', index=False)
        sparse_point_df.to_excel(writer, sheet_name='sparse_points', index=False)

    return {
        'dataset_name': dataset_name,
        'bridge_match_ratio': float(summary_lookup['bridge_match_ratio']),
        'bridge_match_both_pair_ratio': float(summary_lookup['bridge_match_both_pair_ratio']),
        'bridge_shared_point_ratio': float(summary_lookup['bridge_shared_point_ratio']),
        'bridge_global_sparse_ratio': float(summary_lookup['bridge_global_sparse_ratio']),
        'bridge_global_sparse_any_obs_ratio': float(summary_lookup['bridge_global_sparse_any_obs_ratio']),
        'cross_source_match_count': int(summary_lookup['cross_source_match_count']),
        'shared_point_count': int(summary_lookup['shared_point_count']),
        'global_sparse_point_count': int(summary_lookup['global_sparse_point_count']),
    }


def main() -> None:
    args = parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = DEFAULT_DATASETS.copy()
    if args.dataset:
        datasets = {}
        for spec in args.dataset:
            if '=' not in spec:
                raise ValueError(f'Invalid --dataset spec: {spec}')
            name, raw_path = spec.split('=', 1)
            datasets[name] = Path(raw_path).resolve()

    summary_rows = []
    for dataset_name, root in datasets.items():
        print(f'Analyzing bridge participation for {dataset_name} ...')
        summary_rows.append(analyze_dataset(dataset_name, root.resolve(), args.mask_dir.resolve(), out_root))

    compare_df = pd.DataFrame(summary_rows).sort_values('bridge_shared_point_ratio', ascending=False)
    write_csv(compare_df, out_root / 'bridge_participation_comparison.csv')
    with pd.ExcelWriter(out_root / 'bridge_participation_comparison.xlsx', engine='openpyxl') as writer:
        compare_df.to_excel(writer, sheet_name='comparison', index=False)

    print(f'Wrote bridge participation analysis to {out_root}')
    print(compare_df.to_string(index=False))


if __name__ == '__main__':
    main()
