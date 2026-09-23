# Reproduction guide and current status

This guide separates commands that work with files in the candidate package from historical experiment steps requiring additional inputs. No command below has been validated as a full clean-room rerun of manuscript metrics.

## 1. Validate a separately obtained dataset

From the package root:

```bash
python examples/check_release.py --data-dir /path/to/authorized/compact_bridge
python examples/check_release.py --data-dir /path/to/authorized/compact_bridge --deep  # optional; requires Pillow
```

No dataset is bundled in the code candidate. The checker requires `images/`, `masks/`, and `metadata/image_source_mapping.csv`, with matching `frame_XXXXX` basenames. This validates structure, not segmentation accuracy or rights to redistribute the files.

## 2. Compact Bridge: required author-provided or independently generated inputs

The BEG-NeRF source code needs a Nerfstudio-processed image directory with `transforms.json` (including `colmap_im_id` for every frame), matching bridge masks for each processed frame, and a trained **standard Nerfacto** baseline. The current package does not provide camera poses/intrinsics or a baseline checkpoint. The exact paper split and metrics should not be claimed without the original run manifest and held-out frames.

Once BEG-Connect geometry has already been reconstructed and converted to Nerfstudio processed data, `bash examples/train_standard_nerfacto.sh /path/to/processed-scene /path/to/output beg_c` runs the **standard Nerfacto stage only**. It does not create BEG-Connect geometry. This is a convenience wrapper based on the historical 15,000-step training command, **not** a verified exact rerun of the paper's BEG-C result.

The historical Compact Bridge preparation script strictly expects 600 frames (331 UAV, 269 phone), a 525/75 train/test split, same-basename masks and a `colmap_im_id` for every processed frame. A different reconstruction/split needs code adaptation and must not be called the paper's original protocol. After preparing the required inputs in a compatible Nerfstudio environment, the scene-specific preparation and stage-one entry points are:

```bash
python src/training/beg_nerf/compact_bridge/prepare_beg_nerf_v2.py \
  --workspace /path/to/working-directory \
  --processed-data /path/to/nerfstudio-processed-scene \
  --mask-dir /path/to/masks-matching-processed-frame-names

python src/training/beg_nerf/compact_bridge/run_stage1.py \
  --workspace /path/to/working-directory \
  --base-config /path/to/standard-nerfacto-run/config.yml \
  --additional-iterations 5000
```

The base checkpoint expected by the historical stage-one script is `nerfstudio_models/step-000014999.ckpt` next to `config.yml`. Stage one uses four 32×32 patches per batch (bridge, bridge, boundary, global). The residual refinement is a separate 12,000-step component:

```bash
python src/training/beg_nerf/compact_bridge/run_refiner_step.py cache --workspace /path/to/working-directory
python src/training/beg_nerf/compact_bridge/run_refiner_step.py train --workspace /path/to/working-directory --steps 12000
python src/training/beg_nerf/compact_bridge/run_refiner_step.py evaluate --workspace /path/to/working-directory
```

These scripts expect compatible rendering results and CUDA; they have not been exercised here without the private reconstruction and checkpoint. `python src/training/beg_nerf/compact_bridge/evaluate_stage1.py --workspace /path/to/working-directory` evaluates stage one separately.

## 3. Large-span Bridge matching

The historical DL script takes `BRIDGE_LARGE_ROOT` (containing `images/`), `LIGHTGLUE_REPO` and optionally `COLMAP_BIN` as environment variables. The BEG-Connect script uses the same root and additionally requires `BRIDGE_YOLO_WEIGHTS` and the completed DL COLMAP database. Data/weights are not redistributed. The scripts create output beside the authorized input directory; inspect the destination and retain backups before running them. They are source provenance, not clean-room tested command examples.

## 4. Method and ablation coverage

`configs/ablation_settings.yaml` records which components each manuscript label turns on; it is **descriptive**, not an executable matrix. DL+E and DL+G final standalone drivers, and BG/VT final-run provenance, still require verification. BEG-C uses standard Nerfacto on BEG-Connect geometry. For the exact paper comparisons, retain the same train/test images, checkpoint step, mask/crop definition, seeds and per-image outputs for every method. The included RoMa common-50 evaluation source still refers to historical private runs.

## 5. Before public GitHub release

1. Confirm code rights and any institutional approval before activating the proposed MIT license. Data distribution is optional; if desired, separately confirm dataset rights before applying CC BY 4.0.
2. Pin dependency versions from the original working environment, rather than guessing them from present-day installations.
3. Replace remaining private defaults in legacy/evaluation scripts or move those scripts to a clearly labeled provenance archive.
4. Verify VT, BG and individual DL+E/DL+G final-run provenance; add only authentic implementations and run manifests.
5. Run an isolated, authorized end-to-end example and record its environment, inputs, split, outputs and deviations from manuscript numbers.
6. Scan the final Git staging area for restricted media, camera metadata, weights, secrets and absolute private paths.
