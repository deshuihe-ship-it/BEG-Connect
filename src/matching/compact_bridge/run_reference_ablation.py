"""Reference experiment drivers assembled where complete final-experiment scripts could not be identified."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

from run_matching import load_module


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def configure(data, work, lightglue, colmap):
    os.environ["BRIDGE_LARGE_ROOT"] = str(work)
    os.environ["COLMAP_BIN"] = colmap
    if lightglue:
        os.environ["LIGHTGLUE_REPO"] = str(lightglue.resolve())
    scripts = Path(__file__).resolve().parents[1] / "paper_methods"
    dl = load_module("reference_dl", scripts / "dl_dinov2_lightglue.py")
    beg = load_module("reference_beg", scripts / "beg_connect_conservative_expansion.py")
    with (data / "metadata/image_source_mapping.csv").open() as handle:
        sources = {row["name"]: "uav" if row["group_source"] == "UAV" else "phone1" for row in csv.DictReader(handle)}
    dl.source_of = lambda name: sources[Path(name).name]
    beg.source_of = dl.source_of
    dl.IMAGE_DIR = data / "images"
    beg.IMAGE_DIR = dl.IMAGE_DIR
    beg.dl = dl
    return dl, beg


def set_dl_output(dl, output):
    dl.OUT_DIR = output
    dl.DB_DIR = output / "colmap"
    dl.DB_PATH = dl.DB_DIR / "database.db"
    dl.REPORT_DIR = output / "reports"
    dl.LOG_DIR = output / "logs"
    dl.SPARSE_DIR = dl.DB_DIR / "sparse"
    dl.PAIR_FILE = dl.REPORT_DIR / "all_candidate_pairs.txt"


def enhance_images(beg, data, destination):
    destination.mkdir(parents=True)
    beg.build_enhanced_images(data / "masks", destination)


def run_colmap(colmap, arguments, log_path):
    with log_path.open("w") as handle:
        subprocess.run([colmap, *arguments], check=True, stdout=handle, stderr=subprocess.STDOUT)


def matching_option(colmap, name):
    result = subprocess.run([colmap, 'matches_importer', '-h'], capture_output=True, text=True, check=True)
    prefix = 'FeatureMatching' if f'--FeatureMatching.{name}' in result.stdout + result.stderr else 'SiftMatching'
    return f'--{prefix}.{name}'


def map_and_report(dl, output, image_dir):
    sparse = output / "colmap/sparse"
    dl.run_mapper(output / "colmap/database.db", image_dir, sparse, output / "logs/mapper.log")
    models = sorted(path for path in sparse.iterdir() if path.is_dir() and path.name.isdigit())
    if not models:
        raise RuntimeError("COLMAP produced no sparse model; inspect mapper.log and image overlap")
    summaries = [(model, dl.registered_summary(model, output / "reports" / f"registered_{model.name}.json")) for model in models]
    selected, summary = max(summaries, key=lambda item: item[1]["registered_total"])
    write_json(output / "reports/registered_summary.json", summary)
    (output / "reports/selected_model.txt").write_text(str(selected) + "\n")


def guided_sift_pairs(beg, baseline, names, radius):
    seeds, _ = beg.seed_pairs(baseline)
    by_source = {source: set() for source in ("uav", "phone1")}
    available = {source: {beg.frame_index(name): name for name in names if beg.source_of(name) == source} for source in by_source}
    for uav, phone, _ in seeds:
        for name in (uav, phone):
            source = beg.source_of(name)
            for index in range(beg.frame_index(name) - radius, beg.frame_index(name) + radius + 1):
                if index in available[source]:
                    by_source[source].add(available[source][index])
    return [(uav, phone) for uav in sorted(by_source["uav"]) for phone in sorted(by_source["phone1"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("dl", "beg", "vt", "bg"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--lightglue-repo", type=Path)
    parser.add_argument("--dinov2-repo", type=Path)
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--vocab-tree", type=Path)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--neighbor-radius", type=int, default=5)
    args = parser.parse_args()
    data, work = args.data_dir.resolve(), args.work_dir.resolve()
    colmap = shutil.which(args.colmap)
    if not colmap:
        parser.error("COLMAP executable not found")
    if args.method not in ("vt", "bg") and (not args.lightglue_repo or not args.lightglue_repo.is_dir()):
        parser.error("Supply --lightglue-repo for DL/LightGlue methods")
    if args.method == "vt" and (not args.vocab_tree or not args.vocab_tree.is_file()):
        parser.error("VT requires a COLMAP vocabulary tree via --vocab-tree")
    if args.method in ("beg", "bg") and (not args.baseline_dir or not (args.baseline_dir / "colmap/database.db").is_file()):
        parser.error("Supply an existing baseline via --baseline-dir (DL for BEG; enhanced VT/SIFT for BG)")
    if not (data / "metadata/image_source_mapping.csv").is_file() or not (data / "images").is_dir():
        parser.error("Expected released Compact Bridge dataset structure")
    output = work / ("reference_" + args.method)
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    work.mkdir(parents=True, exist_ok=True)
    dl, beg = configure(data, work, args.lightglue_repo, colmap)
    if args.dinov2_repo:
        import torch
        dl.load_dinov2 = lambda device: torch.hub.load(str(args.dinov2_repo.resolve()), "dinov2_vits14", source="local").to(device).eval()
    if args.image_list:
        names = args.image_list.read_text().splitlines()
        if not names or len(names) != len(set(names)):
            parser.error("Image list must contain unique basenames")
        subset = work / ("inputs_" + args.method)
        subset.mkdir()
        for name in names:
            if Path(name).name != name or not (data / "images" / name).is_file():
                parser.error(f"Invalid image basename: {name}")
            (subset / name).symlink_to(data / "images" / name)
        dl.IMAGE_DIR = subset
        beg.IMAGE_DIR = subset
    names = sorted(path.name for path in dl.IMAGE_DIR.glob("*.jpg"))
    if len(names) < 2:
        parser.error("At least two images are required")
    if args.baseline_dir:
        with sqlite3.connect(f"file:{(args.baseline_dir / 'colmap/database.db').resolve()}?mode=ro", uri=True) as connection:
            baseline_names = {row[0] for row in connection.execute("SELECT name FROM images")}
        if baseline_names != set(names):
            parser.error("Baseline image membership must equal the selected input membership")
    if args.method == "dl":
        set_dl_output(dl, output)
        dl.main()
    elif args.method == "beg":
        beg.BASE = args.baseline_dir.resolve()
        beg.OUT = output
        os.environ["COMPACT_BRIDGE_MASK_DIR"] = str(data / "masks")
        beg.main()
    else:
        for folder in ("colmap", "reports", "logs"):
            (output / folder).mkdir(parents=True)
        database = output / "colmap/database.db"
        if args.method == "vt":
            dl.run_feature_extractor(database, dl.IMAGE_DIR, output / "logs/feature_extractor.log")
            run_colmap(colmap, ["vocab_tree_matcher", "--database_path", str(database), "--VocabTreeMatching.vocab_tree_path", str(args.vocab_tree.resolve())], output / "logs/matcher.log")
        else:
            shutil.copy2(args.baseline_dir / "colmap/database.db", database)
            pairs = guided_sift_pairs(beg, database, names, args.neighbor_radius)
            pair_file = output / "reports/guided_pairs.txt"
            pair_file.write_text("".join(f"{uav} {phone}\n" for uav, phone in pairs))
            if pairs:
                run_colmap(colmap, ["matches_importer", "--database_path", str(database), "--match_list_path", str(pair_file), "--match_type", "pairs", matching_option(colmap, 'guided_matching'), "1", matching_option(colmap, 'max_num_matches'), '65536', "--SiftMatching.max_ratio", "0.9", "--SiftMatching.max_distance", "0.8"], output / "logs/matcher.log")
            write_json(output / "reports/guidance.json", {"candidate_pairs": len(pairs), "neighbor_radius": args.neighbor_radius, "matcher": "guided SIFT"})
        map_and_report(dl, output, dl.IMAGE_DIR)
    write_json(output / "reports/reference_implementation.json", {
        "method": args.method, "implementation": "retained_core_with_new_driver" if args.method in ("dl", "beg") else "reconstructed_reference",
        "paper_numerical_reproduction_validated": False, "input_images": names,
        "baseline": str(args.baseline_dir.resolve()) if args.baseline_dir else None,
        "scope": "Reference implementation; not a recovered original ablation run or a guarantee of paper results",
    })


if __name__ == "__main__":
    main()
