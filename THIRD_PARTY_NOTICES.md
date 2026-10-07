# Third-party software, model, and data notices

Updated local release scope: 6 October 2026. This inventory is not a historical version lock, SBOM, or legal opinion.

The project MIT license applies to original code/documentation only. Compact Bridge dataset terms are in data/compact_bridge/DATA_TERMS.txt; public download permission was confirmed by the author, but an unrestricted reuse license has not been selected.

External components are not bundled. Obtain their code and model artifacts upstream and retain applicable notices:

- DINOv2: https://github.com/facebookresearch/dinov2
- LightGlue, including its ALIKED integration: https://github.com/cvg/LightGlue
- ALIKED: https://github.com/Shiaoming/ALIKED
- Nerfstudio: https://github.com/nerfstudio-project/nerfstudio
- COLMAP: https://github.com/colmap/colmap
- RoMa (comparison only): https://github.com/Parskatt/RoMa
- hloc (comparison only): https://github.com/cvg/Hierarchical-Localization

See docs/PRETRAINED_MODELS.md for matching-weight download instructions. No third-party pretrained or scene-trained weights are redistributed. PyTorch/TorchMetrics metrics may download their own feature weights. Dependencies such as NumPy, OpenCV, Pillow, SciPy, pandas, h5py, PyYAML and jaxtyping retain their package-specific licenses.

The historical Large-span Bridge code imports Ultralytics for YOLO segmentation. This is an optional dependency for that workflow, not for supplied Compact Bridge masks. Obtain Ultralytics under its upstream terms (https://github.com/ultralytics/ultralytics); the project MIT license does not relicense it or its weights. The Large-span Bridge imagery, annotations and scene-specific YOLO weights are not supplied.

Check terms for the exact versions/artifacts obtained. This release does not bundle external repositories, native executables, historical camera geometry or scene-trained checkpoints.
