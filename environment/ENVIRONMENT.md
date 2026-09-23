# Environment and third-party dependencies

`requirements-minimal.txt` is an inventory of dependencies found in the current scripts; it is **not a tested, pinned environment**. Historical experiments used Python, PyTorch/CUDA, Nerfstudio/Nerfacto, COLMAP, DINOv2, ALIKED, and LightGlue. The RoMa comparison also requires RoMa and hloc. Their repositories, pretrained weights, and COLMAP executable are not included; install them under their respective licenses.

The exact versions of every third-party package cannot be reliably inferred from the paper results. Before release, verify and record the versions or commit identifiers from the original experiment environment, including Python, CUDA, PyTorch, Nerfstudio, LightGlue, RoMa, and COLMAP. Do not present unverified versions as the experiment's locked environment.

`python examples/check_release.py --data-dir /path/to/authorized/compact_bridge` runs without the ML training environment, but requires a separately obtained dataset. Reproducing manuscript experiments also requires camera reconstructions, corresponding training data/configurations, and some unpublished weights. See the root README.
