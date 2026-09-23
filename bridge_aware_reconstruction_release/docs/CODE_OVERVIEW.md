# Code and manuscript methods

[METHOD_MAP.md](METHOD_MAP.md) provides the method-to-file mapping. DL uses DINOv2 retrieval and ALIKED/LightGlue matching. BEG or BEG-Connect adds bridge-region enhancement and conservative candidate expansion. DL+E and DL+G are component ablations, not separate learned models. BEG-C combines BEG-Connect geometry with standard Nerfacto; Complete also includes BEG-NeRF bridge-patch fine-tuning and residual image refinement.

`src/matching/paper_methods/` contains the main Large-span Bridge DL and BEG-Connect implementations. `src/training/beg_nerf/` contains the two BEG-NeRF stages and Compact Bridge entry points. `src/evaluation/` contains evaluation and diagnostic scripts. Three early `run_*training.py` variants and one experiment-workbook helper were moved to the sibling `bridge_aware_reconstruction_unpublished_archive/` directory and retained there; they are not the manuscript's final BEG-NeRF implementation.

Some legacy and comparison scripts still contain historical absolute paths and depend on the original workspace. The Compact Bridge BEG-NeRF entry points now call shared code within this package and accept workspace/configuration paths, but training still requires unpublished poses, split data, and a baseline checkpoint. Large-span Bridge matching requires restricted images and YOLO weights. See the README and reproduction guide for details.

Final standalone run entry points for VT, BG, DL+E, and DL+G have not been traced to completion. Similar exploratory scripts must not be relabeled as manuscript methods, and the package does not claim one-command reproduction of these table rows.
