# Code origin and changes

## Retained code

DL/BEG cores, Compact Bridge mask generation, BEG-NeRF fine-tuning/losses, the residual network, evaluation and RoMa comparison code were retained from the research workspace. Release edits replace local paths with arguments/environment variables, connect shared modules and protect matching outputs. These files should not be described as entirely rewritten or lost.

The patch-sampler integration is included in the release instead of relying on a private Nerfstudio `_bcat_sample_indices` method. Its equal-sized perspective-image input contract is checked explicitly.

## Reconstructed references

`src/matching/compact_bridge/run_reference_ablation.py` provides DL/BEG adapters, a wrapper around official COLMAP VT and a reconstructed BG reference. Complete versions of the original BG, DL+E and DL+G scripts corresponding to the final experiments could not be identified in the retained materials. No standalone DL+E/DL+G implementations are supplied; their enhancement and guided-expansion modules remain in the BEG main method. BG is reconstructed from the described procedure and retained materials, not a recovered original experiment.

References are provided for inspection, adaptation and new experiments. They have not been validated to reproduce the numerical results reported in the paper. Execution checks, original-artifact recounts and historical-model reevaluation are different evidence types; none is a substitute for full independent reproduction.

## User-provided inputs

COLMAP and a vocabulary tree for VT; official pretrained matching models; optional RoMa/hloc dependencies; new SfM geometry and Nerfstudio data; user-trained scene models. Learned masks for new scenes require new annotations and segmenter training/fine-tuning.
