from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_of(name):
    return "uav" if int(Path(name).stem.rsplit("_", 1)[1]) <= 331 else "phone1"


def main():
    parser = argparse.ArgumentParser(description="Packaged historical Compact Bridge DL/BEG adaptation; not a verified paper rerun")
    parser.add_argument("method", choices=["dl", "beg"])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--lightglue-repo", type=Path, required=True)
    parser.add_argument("--colmap", required=True)
    args = parser.parse_args()
    data = args.data_dir.resolve()
    work = args.work_dir.resolve()
    if len(list((data / "images").glob("*.jpg"))) != 600:
        parser.error("Expected 600 Compact Bridge JPEG images; run the dataset checker first")
    if args.method == "beg" and not (work / "all_dinov2_lightglue/colmap/database.db").is_file():
        parser.error("Run DL first in the same work directory")
    output = work / ("all_dinov2_lightglue" if args.method == "dl" else "all_dinov2_lightglue_bridge_enhance_expand")
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    os.environ["BRIDGE_LARGE_ROOT"] = str(work)
    os.environ["LIGHTGLUE_REPO"] = str(args.lightglue_repo.resolve())
    os.environ["COLMAP_BIN"] = args.colmap
    if args.method == "beg":
        if len(list((data / "masks").glob("*.png"))) != 600:
            parser.error("Expected 600 supplied Compact Bridge masks")
        os.environ["COMPACT_BRIDGE_MASK_DIR"] = str(data / "masks")
    scripts = Path(__file__).resolve().parents[1] / "paper_methods"
    dl = load_module("compact_dl", scripts / "dl_dinov2_lightglue.py")
    dl.IMAGE_DIR = data / "images"
    dl.source_of = source_of
    if args.method == "dl":
        dl.main()
        return
    beg = load_module("compact_beg", scripts / "beg_connect_conservative_expansion.py")
    beg.dl = dl
    beg.IMAGE_DIR = data / "images"
    beg.source_of = source_of

    beg.main()


if __name__ == "__main__":
    main()
