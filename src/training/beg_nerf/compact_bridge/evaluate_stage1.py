#!/usr/bin/env python3
"""Evaluate xiaoqiao BEG-NeRFv2 stage one on its untouched 75-view test split."""

from __future__ import annotations

import importlib.util
import argparse
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "evaluate_beg_nerf_v2.py"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("shared_beg_eval", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.WORK = args.workspace
    module.MANIFEST = args.workspace / "config/exact_manifest.json"
    module.main()


if __name__ == "__main__":
    main()
