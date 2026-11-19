"""Deterministic procedural sample scene and interoperability fixtures."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

from .binary_format import OLD_FORMAT_VERSION, encode_binary
from .model import (
    CURRENT_SCHEMA_VERSION,
    DEFAULT_STROKE_SAMPLE,
    DEFAULT_STYLE,
    Contour,
    Project,
    Segment,
    Shape,
    Style,
    StrokeSample,
    canonical_json,
    f32,
    project_from_dict,
)
from .png import encode_png
from .render import RenderConfig, RenderEngine


def _pt(x: float, y: float) -> tuple[float, float]:
    return (f32(x), f32(y))


def _ellipse(center: tuple[float, float], rx: float, ry: float, closed: bool = True) -> Contour:
    cx, cy = center
    kappa = 0.5522847498307936
    p0 = _pt(cx + rx, cy)
    p1 = _pt(cx, cy + ry)
    p2 = _pt(cx - rx, cy)
    p3 = _pt(cx, cy - ry)
    segments = [
        Segment(
            "cubic",
            [
                p0,
                _pt(cx + rx, cy + ry * kappa),
                _pt(cx + rx * kappa, cy + ry),
                p1,
            ],
        ),
        Segment(
            "cubic",
            [
                p1,
                _pt(cx - rx * kappa, cy + ry),
                _pt(cx - rx, cy + ry * kappa),
                p2,
            ],
        ),
        Segment(
            "cubic",
            [
                p2,
                _pt(cx - rx, cy - ry * kappa),
                _pt(cx - rx * kappa, cy - ry),
                p3,
            ],
        ),
        Segment(
            "cubic",
            [
                p3,
                _pt(cx + rx * kappa, cy - ry),
                _pt(cx + rx, cy - ry * kappa),
                p0,
            ],
        ),
    ]
    return Contour(closed=closed, segments=segments)


def _hill_contour() -> Contour:
    p0 = _pt(18, 128)
    p1 = _pt(217, 128)
    peak = _pt(119, 76)
    segments = [
        Segment("line", [p0, p1]),
        Segment("quadratic", [p1, _pt(171, 88), peak]),
        Segment("quadratic", [peak, _pt(63, 87), p0]),
    ]
    return Contour(closed=True, segments=segments)


def _tree_trunk() -> Contour:
    points = [_pt(116, 109), _pt(121, 109), _pt(120, 137), _pt(117, 137)]
    return Contour(
        closed=True,
        segments=[
            Segment("line", [points[0], points[1]]),
            Segment("line", [points[1], points[2]]),
            Segment("line", [points[2], points[3]]),
            Segment("line", [points[3], points[0]]),
        ],
    )


def _branch_contour() -> Contour:
    return Contour(
        closed=False,
        segments=[
            Segment("cubic", [_pt(119, 103), _pt(103, 92), _pt(100, 78), _pt(88, 70)]),
            Segment("quadratic", [_pt(88, 70), _pt(81, 65), _pt(74, 67)]),
        ],
    )


def _bird_contour() -> Contour:
    return Contour(
        closed=False,
        segments=[
            Segment("quadratic", [_pt(53, 41), _pt(62, 32), _pt(72, 39)]),
            Segment("quadratic", [_pt(72, 39), _pt(82, 31), _pt(91, 39)]),
        ],
    )


def _cloud_contours() -> list[Contour]:
    return [
        _ellipse((65, 45), 17, 9),
        _ellipse((83, 42), 20, 11),
        _ellipse((101, 47), 16, 8),
    ]


def build_sample_project() -> Project:
    shapes = [
        Shape(
            "sky_sun",
            "background",
            f32(2.0),
            ["accent"],
            [_ellipse((193, 38), 17, 17)],
        ),
        Shape(
            "cloud_bank",
            "background",
            f32(1.4),
            ["neutral"],
            _cloud_contours(),
        ),
        Shape(
            "hill_mass",
            "midground",
            f32(2.25),
            ["leaf"],
            [_hill_contour()],
        ),
        Shape(
            "tree_crown",
            "midground",
            f32(2.4),
            ["leaf"],
            [_ellipse((119, 89), 31, 25)],
        ),
        Shape(
            "tree_trunk",
            "midground",
            f32(1.8),
            ["neutral"],
            [_tree_trunk()],
        ),
        Shape(
            "branch",
            "ink",
            f32(1.15),
            ["neutral"],
            [_branch_contour()],
        ),
        Shape(
            "bird",
            "ink",
            f32(0.95),
            [],
            [_bird_contour()],
        ),
    ]
    return Project(
        project_id="npr-demo-scene",
        version_stamp="000000000001",
        shapes=shapes,
        style=Style(),
        stroke_sample=StrokeSample(),
        layer_order=["background", "midground", "ink"],
        schema_version=CURRENT_SCHEMA_VERSION,
        metadata={"name": "Deterministic NPR demo", "seed": 20261119},
    )


def legacy_v1_document(project: Project) -> dict[str, Any]:
    document = project.to_dict()
    document["schema_version"] = 1
    document.pop("style")
    document.pop("stroke_sample")
    document.pop("layer_order")
    return document


def legacy_v2_document(project: Project) -> dict[str, Any]:
    document = project.to_dict()
    document["schema_version"] = 2
    document.pop("stroke_sample")
    document.pop("layer_order")
    return document


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bad_index(binary: bytes) -> bytes:
    # v2, little endian, 32-bit indices: overwrite the final index with 65535.
    corrupted = bytearray(binary)
    struct.pack_into("<I", corrupted, len(corrupted) - 4, 0x0000FFFF)
    return bytes(corrupted)


def _index_count_mismatch(binary: bytes) -> bytes:
    # Header index_count lives at byte 18 for the documented fixed header.
    corrupted = bytearray(binary)
    declared = struct.unpack_from("<I", corrupted, 18)[0]
    struct.pack_into("<I", corrupted, 18, declared + 1)
    return bytes(corrupted)


def write_samples(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    project = build_sample_project()
    current = project.to_dict()
    v1 = legacy_v1_document(project)
    v2 = legacy_v2_document(project)

    _write_json(out / "current_v3.json", current)
    _write_json(out / "legacy_v1.json", v1)
    _write_json(out / "legacy_v2.json", v2)
    invalid_sampling = copy.deepcopy(current)
    invalid_sampling["stroke_sample"]["curve_steps"] = 0
    _write_json(out / "invalid_stroke_sample.json", invalid_sampling)

    binary_files: dict[str, Path] = {}
    for byte_order in ("little", "big"):
        for index_width in (2, 4):
            encoded = encode_binary(
                project,
                byte_order=byte_order,
                index_width=index_width,
                format_version=2,
            )
            path = out / f"skeleton_v2_{byte_order[0]}{index_width * 8}.npbs"
            path.write_bytes(encoded.data)
            binary_files[path.name] = path

    for byte_order in ("little", "big"):
        encoded = encode_binary(
            project,
            byte_order=byte_order,
            index_width=4 if byte_order == "little" else 2,
            format_version=OLD_FORMAT_VERSION,
        )
        path = out / f"skeleton_v1_{byte_order[0]}{4 if byte_order == 'little' else 2}.npbs"
        path.write_bytes(encoded.data)
        binary_files[path.name] = path

    no_meta = encode_binary(
        project,
        byte_order="little",
        index_width=2,
        format_version=2,
        include_project_metadata=False,
    )
    no_meta_path = out / "skeleton_v2_no_metadata_le16.npbs"
    no_meta_path.write_bytes(no_meta.data)
    binary_files[no_meta_path.name] = no_meta_path

    valid_le32 = binary_files["skeleton_v2_l32.npbs"].read_bytes()
    bad_endian = bytearray(valid_le32)
    bad_endian[4:8] = b"\xde\xad\xbe\xef"
    (out / "invalid_endian.npbs").write_bytes(bytes(bad_endian))
    (out / "invalid_index.npbs").write_bytes(_bad_index(valid_le32))
    (out / "invalid_index_count.npbs").write_bytes(_index_count_mismatch(valid_le32))

    engine = RenderEngine(project, RenderConfig())
    images = {}
    for mode in ("lineart", "sketch", "toon"):
        result = engine.render(mode)  # type: ignore[arg-type]
        path = out / f"preview_{mode}.png"
        path.write_bytes(encode_png(result.image))
        images[mode] = {
            "file": path.name,
            "pixel_sha256": result.pixel_sha256,
            "png_sha256": result.png_sha256,
            "shape_render_hashes": result.shape_hashes,
            "geometry_hashes": result.geometry_hashes,
        }

    manifest = {
        "format": "npr-sample-manifest-v1",
        "project_id": project.project_id,
        "project_hash": project.content_hash(),
        "geometry_hash": project.geometry_hash(),
        "generator": "npr_editor.samples.build_sample_project",
        "files": {},
        "previews": images,
        "binary_variants": sorted(binary_files),
    }
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _write_json(out / "manifest.json", manifest)
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate deterministic NPR samples")
    parser.add_argument("--out", default="samples")
    args = parser.parse_args()
    manifest = write_samples(args.out)
    print(canonical_json({"project_hash": manifest["project_hash"], "files": len(manifest["files"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
