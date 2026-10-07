# Bridge-aware UAV–phone reconstruction

Code and Compact Bridge data accompanying the AIC manuscript on BEG-Connect and BEG-NeRF.

## Data

`data/compact_bridge/` contains 600 processed JPEG images at 960 × 540 pixels (331 UAV, 269 phone), bridge-region masks, source labels and retained split lists. These are processed research images, not original-resolution camera files. See [dataset notes](docs/DATASET.md) and `data/compact_bridge/DATA_TERMS.txt`.

```bash
python examples/check_release.py --data-dir data/compact_bridge --deep
```

Users reconstruct camera geometry with SfM and prepare their own Nerfstudio scene. Historical camera poses, intrinsics and COLMAP databases are not distributed. Large-span Bridge images and manual annotations remain restricted.

## Original code and reconstructed references

The release contains retained original implementations, portability changes and reconstructed reference drivers. Complete versions of the original BG, DL+E and DL+G scripts corresponding to the final experiments could not be identified in the retained materials. Standalone DL+E/DL+G code is not provided; their enhancement and guided-expansion modules remain in the BEG main-method implementation. BG is provided only as a reconstructed reference, not the original experimental program, and is not guaranteed to reproduce the paper's numerical results. VT uses COLMAP's official vocabulary-tree implementation. Complete original run manifests are not available for every ablation.

| Code                                                    | Origin and purpose                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `src/matching/paper_methods/`                           | Retained DL/BEG core, adapted for explicit paths and supplied masks.                                                                  |
| `src/matching/compact_bridge/run_matching.py`           | Compact Bridge adapter to retained DL/BEG code.                                                                                       |
| `src/matching/compact_bridge/run_reference_ablation.py` | DL/BEG adapters, an official COLMAP VT wrapper, and a reconstructed BG reference. No standalone DL+E/DL+G code.                       |
| `src/training/beg_nerf/`                                | Retained fine-tuning, residual network and evaluation code with portable workspace interfaces and included patch-sampler integration. |
| `src/matching/roma_comparison/`                         | Retained RoMa comparison code with configurable external paths.                                                                       |
| `configs/`, `results/`                                  | Parameter/summary records, not guarantees of historical reproduction.                                                                 |

See [method map](docs/METHOD_MAP.md), [running instructions](docs/REPRODUCIBILITY.md), [code origin](docs/CODE_OVERVIEW.md) and [validation scope](docs/VALIDATION.json). Smoke tests demonstrate execution on stated inputs, not full experimental reproduction.

## Models and dependencies

No scene-trained Nerfacto, BEG-NeRF, residual-network or YOLO weights are released. The historical YOLO model was trained for our scene and is not provided as a general bridge segmenter. For a new scene requiring learned masks, manually annotate representative images and train or fine-tune a segmentation model. Compact Bridge masks are supplied directly.

DINOv2, ALIKED and LightGlue use their authors' pretrained weights; see [download instructions](docs/PRETRAINED_MODELS.md) and [basic dependencies](environment/ENVIRONMENT.md). No locked historical environment is claimed.

Retained splits cover identified BEG-NeRF and repeatability runs, not every ablation. Nerfstudio validation/test memberships coincide in these records. The residual network holds validation images out of its training set, separately from test images.

Original project code is covered by `LICENSE`; third-party artifacts retain their own terms. The code license does not cover dataset materials. Updating this local directory does not publish it to GitHub.
