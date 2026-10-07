# Basic dependencies

Use a compatible Python/PyTorch/CUDA environment with Nerfstudio (Nerfacto), COLMAP, NumPy, OpenCV, Pillow and the libraries in requirements-minimal.txt. Install PyTorch/CUDA and Nerfstudio using their official compatibility instructions. Historical matching/training requires CUDA.

DINOv2 and LightGlue/ALIKED are external dependencies; see docs/PRETRAINED_MODELS.md. RoMa/hloc are only needed by comparison scripts. Ultralytics is only needed by the Large-span Bridge YOLO workflow, not the supplied Compact Bridge masks.

Paths are supplied through CLI arguments or documented environment variables. The released BEG patch sampler does not require a private Nerfstudio modification. This is a basic inventory, not a locked historical environment or a clean-room-tested installation recipe. No scene-trained weights are included. See docs/VALIDATION.json for actual execution checks and limits.
