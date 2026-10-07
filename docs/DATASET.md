# Dataset availability

The author confirmed permission to release Compact Bridge materials on 6 October 2026. The local release includes `data/compact_bridge/`:
- `images/`: 600 JPEG frames at 960 × 540, not original camera-resolution files.
- `masks/`: 600 same-basename rule-based bridge-region PNG masks.
- `metadata/image_source_mapping.csv`: 331 UAV / 269 phone source labels.
- `metadata/retained_splits.json`: filename memberships, per-run provenance and original-manifest hashes.
- `metadata/train.txt`, `val.txt`, `test.txt`: memberships shared by the identified records.
- `metadata/SHA256SUMS.txt`: hashes of images, masks and metadata, excluding itself.

Identified retained Nerfstudio runs use 525 training images (289 UAV, 236 phone) and 75 held-out images (42 UAV, 33 phone). Nerfstudio validation and test memberships are identical. The residual network separately reserves 30 validation images from training inputs, leaving 495 training images and 75 independent test images. These records do not establish the protocol of every matching ablation. New SfM processing can change ordering: preserve filenames and check generated memberships against retained lists.

Run `python examples/check_release.py --data-dir data/compact_bridge --deep` from the package root. Masks approximate bridge regions; they are not independent geometry or defect ground truth.

The author confirmed public download permission. No CC BY or unrestricted reuse license has been selected; the code MIT license does not apply to data. See `data/compact_bridge/DATA_TERMS.txt`.

Large-span Bridge imagery and annotations remain restricted. Original videos, historical camera geometry, COLMAP databases and scene-trained weights are excluded. Users reconstruct geometry and train models themselves; exact manuscript numbers are not guaranteed.
