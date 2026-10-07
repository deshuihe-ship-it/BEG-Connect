# Third-party pretrained weights

These models were pretrained by their upstream authors, not trained by this study. Obtain them from official sources; no weights are redistributed here.

## DINOv2

Official source: https://github.com/facebookresearch/dinov2

Use ViT-S/14 without registers. PyTorch Hub downloads its weights on first use:

```python
import torch
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval()
```

The upstream README also links the backbone checkpoint. Initial download requires internet access; offline use needs a populated cache/local checkout.

## ALIKED / LightGlue

Official integration: https://github.com/cvg/LightGlue
ALIKED upstream: https://github.com/Shiaoming/ALIKED

Install LightGlue following upstream instructions or supply its checkout via LIGHTGLUE_REPO / --lightglue-repo.

```python
from lightglue import ALIKED, LightGlue
extractor = ALIKED(max_num_keypoints=2048).eval()
matcher = LightGlue(features="aliked", filter_threshold=0.05).eval()
```

The retained implementation downloads aliked-n16 and aliked_lightglue (v0.1_arxiv) weights through PyTorch Hub. Confirm identifiers against the installed upstream version; future defaults may differ.

No scene-trained Nerfacto, BEG-NeRF, residual U-Net or YOLO weight is supplied. The YOLO segmenter was trained for our bridge scene and is not released as a general detector. For learned masks in another scene, manually annotate representative images and train or fine-tune a segmenter. Compact Bridge uses supplied bridge-region masks. Upstream artifact licenses are separate from the project code license.
