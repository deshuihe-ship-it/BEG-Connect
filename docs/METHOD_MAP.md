# Method-to-code map

| Setting       | Released implementation                                                        | Origin                                                                                               |
| ------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| DL            | DINOv2 retrieval and ALIKED/LightGlue; CLI `dl`                                | Retained core, new packaging                                                                         |
| DL+E          | Enhancement module remains in BEG; no standalone experiment code supplied      | Complete final-experiment script not identified in retained materials                                |
| DL+G          | Guided-expansion module remains in BEG; no standalone experiment code supplied | Complete final-experiment script not identified in retained materials                                |
| BEG           | CLI `beg`: conservative expansion with matching on enhanced images             | Retained core, new packaging                                                                         |
| VT            | CLI `vt`: wrapper around official COLMAP SIFT/vocabulary-tree matching         | Official implementation; exact historical run configuration not recovered                            |
| BG            | CLI `bg`: reconstructed guided-SIFT reference                                  | Complete final-experiment script not identified in retained materials; no guarantee of paper results |
| BEG-C         | Standard Nerfacto on user-generated BEG geometry                               | External Nerfstudio baseline                                                                         |
| Complete      | Bridge-patch fine-tuning plus residual U-Net                                   | Retained model/loss code, portable sampler integration                                               |
| D-R / D-R-BEG | `src/matching/roma_comparison/`                                                | Retained comparison code; requires RoMa and hloc                                                     |

Complete versions of the original BG, DL+E and DL+G scripts corresponding to the final experiments could not be identified in the retained materials. DL+E/DL+G independent implementations are not supplied. Their enhancement and guided-expansion modules remain available in the BEG main method.

BG is a reconstructed reference based on the described procedure and retained materials, not the original experimental program. It expects an enhanced SIFT baseline and uses seeds with at least 30 verified cross-source inliers, temporal radius five and all cross-source pairs between the expanded sets. These are explicit reference choices; identical paper results are not guaranteed.

VT calls the [official COLMAP vocabulary-tree implementation](https://colmap.github.io/cli.html). This does not establish recovery of the exact historical version or configuration.
