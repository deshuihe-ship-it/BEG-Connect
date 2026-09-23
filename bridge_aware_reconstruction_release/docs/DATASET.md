# Dataset availability and rights

## Current release candidate

The public code candidate contains **no Compact Bridge images, masks, or source-mapping metadata**. Those files are held locally outside the release directory because redistribution authority has not been confirmed. Large-span Bridge data and its manually annotated images, training data, and YOLO weights are also excluded under the external acquisition arrangement.

The local Compact Bridge set contains 600 retained images (331 UAV, 269 phone), 600 same-basename masks from the V3.1 rule-based workflow, and `metadata/image_source_mapping.csv`. A validation utility is available for a dataset that the user is authorized to use:

```bash
python examples/check_release.py --data-dir /path/to/compact_bridge
python examples/check_release.py --data-dir /path/to/compact_bridge --deep
```

The expected dataset directory has `images/`, `masks/`, and `metadata/image_source_mapping.csv`. The deep check requires Pillow and verifies image decoding, mask size, and nonempty masks; it does not assess mask accuracy or geometry.

## Before distributing data

The manuscript identifies the Compact Bridge as a campus site. Before public release, confirm who owns the images and masks, whether the site or institution imposes terms on recording/distribution, and whether identifiable people, vehicles, signs, or other third-party rights appear. Missing EXIF metadata alone does not answer those questions.

CC BY 4.0 is recorded only as a possible license choice in the local holdback. It is **not currently applied and grants no permission**. If the rights holders approve public and commercial reuse with attribution, replace the proposal note with the actual data license and add the data. Otherwise, keep the dataset private or omit it entirely. The code package by itself cannot reproduce the manuscript's SfM, Nerfacto, or BEG-NeRF metrics; it lacks poses/intrinsics, run splits/manifests, and baseline checkpoints.

Bridge masks are not independent survey ground truth or defect labels.
