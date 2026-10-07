#!/usr/bin/env python3
"""Run the shared BEG-NeRFv2 stage-one implementation on xiaoqiao."""

from __future__ import annotations

import importlib.util
import argparse
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "train_beg_nerf_v2.py"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if not args.base_config.is_file():
        parser.error(f"Base Nerfacto config not found: {args.base_config}")
    sampler = args.workspace / "config/bridge_patch_sampler.json"
    if not sampler.is_file():
        parser.error(f"Run compact_bridge/prepare_beg_nerf_v2.py first: {sampler}")
    spec = importlib.util.spec_from_file_location("shared_beg_stage1", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.WORK = args.workspace
    module.BASE_CONFIG = args.base_config
    module.BASE_CHECKPOINTS = args.base_config.parent / "nerfstudio_models"
    module.SAMPLER_CONFIG = sampler
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()
