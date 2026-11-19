#!/usr/bin/env python3
"""Rebuild an installable wheel from an already-installed package tree.

Only stdlib is used so the offline bootstrap does not depend on pip/wheel.
Usage: repack_wheel.py <site-packages-dir> <distribution-name> <out-wheel-path>
"""
import io
import os
import sys
import zipfile


def find_dist_info(site_dir: str, dist_name: str) -> str:
    prefix = dist_name.lower().replace("-", "_")
    for entry in sorted(os.listdir(site_dir)):
        stem = entry[: -len(".dist-info")].lower().replace("-", "_")
        if entry.endswith(".dist-info") and (stem == prefix or stem.startswith(prefix + "_") or stem.startswith(prefix + "-")):
            return os.path.join(site_dir, entry)
    raise SystemExit(f"dist-info for {dist_name} not found in {site_dir}")


def parse_record(dist_info: str):
    record_path = os.path.join(dist_info, "RECORD")
    entries = []
    with open(record_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            path = line.split(",", 1)[0]
            entries.append(path)
    return entries


def wheel_name(site_dir: str, dist_info: str) -> str:
    meta = {}
    with open(os.path.join(dist_info, "METADATA"), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Name:"):
                meta["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                meta["version"] = line.split(":", 1)[1].strip()
            elif line.strip() == "":
                break
    tags = []
    pure = True
    with open(os.path.join(dist_info, "WHEEL"), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("Tag:"):
                tags.append(line.split(":", 1)[1].strip())
            elif line.startswith("Root-Is-Purelib:"):
                pure = line.split(":", 1)[1].strip() == "true"
    if not tags:
        tags = ["py3-none-any"]
    tag = tags[0]
    name_norm = meta["name"].replace("-", "_")
    return f"{name_norm}-{meta['version']}-{tag}.whl"


def main() -> None:
    site_dir, dist_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    dist_info = find_dist_info(site_dir, dist_name)
    recorded = parse_record(dist_info)
    files = []
    for rel in recorded:
        if rel.endswith(".dist-info/RECORD"):
            continue
        normalized = os.path.normpath(os.path.join(site_dir, rel))
        if os.path.commonpath([normalized, os.path.abspath(site_dir)]) != os.path.abspath(site_dir):
            continue
        files.append(rel)
    files.append(os.path.relpath(os.path.join(dist_info, "RECORD"), site_dir))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for rel in sorted(set(files)):
            abs_path = os.path.join(site_dir, rel)
            if not os.path.isfile(abs_path):
                continue
            archive.write(abs_path, rel)
    print(wheel_name(site_dir, dist_info), "->", out_path)


if __name__ == "__main__":
    main()
