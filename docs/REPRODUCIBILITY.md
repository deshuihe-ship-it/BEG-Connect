# Running the released code

Run commands from the repository root. Matching drivers require new outputs. Complete original standalone DL+E/DL+G scripts corresponding to the final experiments could not be identified in the retained materials; no standalone code for these two ablations is supplied. Their component modules remain in BEG. VT calls the official COLMAP vocabulary-tree implementation. Reconstructed references do not guarantee paper numbers; read [the method map](METHOD_MAP.md).

## Matching

```bash
python examples/check_release.py --data-dir data/compact_bridge --deep
python src/matching/compact_bridge/run_reference_ablation.py dl --data-dir data/compact_bridge --work-dir /path/to/work --lightglue-repo /path/to/LightGlue --colmap /path/to/colmap
python src/matching/compact_bridge/run_reference_ablation.py beg --data-dir data/compact_bridge --work-dir /path/to/work --baseline-dir /path/to/work/reference_dl --lightglue-repo /path/to/LightGlue --colmap /path/to/colmap
python src/matching/compact_bridge/run_reference_ablation.py vt --data-dir data/compact_bridge --work-dir /path/to/work --vocab-tree /path/to/vocabulary-tree.bin --colmap /path/to/colmap
```

For offline DINOv2 add `--dinov2-repo /path/to/dinov2` with cached weights. Add `--image-list examples/smoke_images.txt` for an eight-image execution check; apply the same list to dependent stages. Subsets are not full experiments.

A complete original BG script corresponding to the final experiment could not be identified in the retained materials. The released BG code is a reconstructed reference and does not guarantee the reported numerical results. It expects a SIFT baseline reconstructed from enhanced images:

```bash
python examples/build_enhanced_dataset.py --data-dir data/compact_bridge --output-dir /path/to/enhanced-data
python src/matching/compact_bridge/run_reference_ablation.py vt --data-dir /path/to/enhanced-data --work-dir /path/to/enhanced-work --vocab-tree /path/to/vocabulary-tree.bin --colmap /path/to/colmap
python src/matching/compact_bridge/run_reference_ablation.py bg --data-dir data/compact_bridge --work-dir /path/to/work --baseline-dir /path/to/enhanced-work/reference_vt --colmap /path/to/colmap
```

Do not pass a DL database to BG. Baseline image memberships must equal selected input memberships. Reports are under `reference_METHOD/reports/`. Inspect mapper logs when overlap is insufficient. Convert the selected sparse model into Nerfstudio data and confirm image basenames/source labels before training. The full-data `run_matching.py dl/beg` adapter remains available.

## Neural stages

```bash
bash examples/train_standard_nerfacto.sh /path/to/processed-scene /path/to/output beg_c
python src/training/beg_nerf/compact_bridge/prepare_beg_nerf_v2.py --workspace /path/to/workspace --processed-data /path/to/processed-scene --mask-dir data/compact_bridge/masks
python src/training/beg_nerf/compact_bridge/run_stage1.py --workspace /path/to/workspace --base-config /path/to/nerfacto-run/config.yml --additional-iterations 5000
python src/training/beg_nerf/compact_bridge/evaluate_stage1.py --workspace /path/to/workspace
python src/training/beg_nerf/compact_bridge/run_refiner_step.py cache --workspace /path/to/workspace
python src/training/beg_nerf/compact_bridge/run_refiner_step.py train --workspace /path/to/workspace --steps 12000
python src/training/beg_nerf/compact_bridge/run_refiner_step.py evaluate --workspace /path/to/workspace
```

Use equal-sized perspective images without a Nerfstudio foreground-mask field; provide bridge masks through the sampler manifest. Required inputs include `transforms.json` with per-frame `colmap_im_id`, same-basename masks and a user-trained config with `nerfstudio_models/step-000014999.ckpt`. Compact Bridge preparation expects all 600 registered images and generates the 525/75 interval split. Preparation automatically checks train, validation and test filename memberships against the released metadata lists and stops before writing manifests if any membership differs or filenames are duplicated. Use --split-dir to specify an alternative directory of retained lists; the released Compact Bridge lists are the default. Ordering alone does not cause a mismatch. The residual network reserves 30 validation views from training inputs separately from the 75 test views.

## Other scripts

- Crop evaluation: `--model-root`, `--mask-dir`, `--output-dir`, repeated `--experiment`.
- Geometry analysis: repeated `--dataset NAME=PATH`, `--mask-dir`, `--out-root`.
- Source-specific splits: `--processed-data`, `--output-dir`.
- Large-span training: explicit workspace/data/model arguments and authorized inputs.
- RoMa comparison: `BRIDGE_LARGE_ROOT`, `ROMA_REPO`, `ROMA_WEIGHTS`, `ROMA_DINO_WEIGHTS`, `COLMAP_BIN`; install RoMa/hloc first. BEG imports the adjacent released RoMa script.
- Common-set evaluation: `--workspace` and `--cases`, a JSON mapping `D-L`, `D-L-BEG`, `D-R`, `D-R-BEG` to `data`, `config`, `checkpoint`, `split`. Its `train` action performs training, not just evaluation.

No independent full retraining or exact numerical reproduction is claimed. `VALIDATION.json` records actual local checks.
