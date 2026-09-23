#!/usr/bin/env python3
"""Dispatch a shared render-refiner step with xiaoqiao workspace paths."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parents[1]

files = {
    "cache": "cache_render_refiner_pairs.py",
    "train": "train_render_refiner.py",
    "evaluate": "evaluate_render_refiner.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("cache", "train", "evaluate"))
    parser.add_argument("--workspace", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    source = SHARED / files[args.step]
    spec = importlib.util.spec_from_file_location(f"shared_refiner_{args.step}", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.WORK = args.workspace
    module.OUT = args.workspace / "render_refiner"
    if args.step == "cache":
        module.MANIFEST_PATH = args.workspace / "config/exact_manifest.json"
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()
