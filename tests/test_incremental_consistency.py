from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from .conftest import Workspace


def _replace_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (256, 192), (242, 241, 236))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 24, 110, 88), fill=color, outline=(20, 20, 20), width=2)
    draw.ellipse((145, 84, 226, 165), fill=(60, 120, 210), outline=(20, 20, 20), width=2)
    image.save(path, format="PNG")


def _edit_assertion(path: Path) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    current = document["assertions"][0]["value"]
    document["assertions"][0]["value"] = "circle" if current != "circle" else "rectangle"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def test_arbitrary_edit_sequence_matches_clean_full_rebuild(
    workspace: Workspace, tmp_path: Path
) -> None:
    workspace.run(mode="full", output_name="incremental-output")

    _replace_image(workspace.samples / "sample-01" / "image.png", (200, 40, 60))
    workspace.run(mode="incremental", output_name="incremental-output")

    _edit_assertion(workspace.samples / "sample-02" / "text.json")
    workspace.run(mode="incremental", output_name="incremental-output")

    workspace.edit_config(visual_weight=0.5, text_weight=0.5)
    workspace.run(mode="incremental", output_name="incremental-output")

    workspace.edit_config(chunk_size=16)
    workspace.run(mode="incremental", output_name="incremental-output")

    workspace.edit_config(quantization_level=2)
    incremental = workspace.run(mode="incremental", output_name="incremental-output")

    clean_root = tmp_path / "clean"
    shutil.copytree(workspace.samples, clean_root / "samples")
    shutil.copytree(workspace.config_dir, clean_root / "config")
    clean = Workspace(
        root=clean_root,
        samples=clean_root / "samples",
        config_dir=clean_root / "config",
        config_path=clean_root / "config" / "alignment.json",
        injections_path=clean_root / "config" / "injections.json",
    )
    full = clean.run(mode="full", output_name="full-output")

    assert incremental["reproducibility_sha256"] == full["reproducibility_sha256"]
    assert incremental["retrieval_index"]["queries"] == full["retrieval_index"]["queries"]
    assert incremental["failures"] == full["failures"]
    assert incremental["boundary_diagnostics"] == full["boundary_diagnostics"]
    assert incremental["retrieval_quality"] == full["retrieval_quality"]
    for left, right in zip(incremental["samples"], full["samples"], strict=True):
        assert left["visual_cache_key"] == right["visual_cache_key"]
        assert left["text_cache_key"] == right["text_cache_key"]
        assert left["alignment_cache_key"] == right["alignment_cache_key"]
        assert left["self_alignment"] == right["self_alignment"]


def test_stage_selective_cache_reuse(workspace: Workspace) -> None:
    baseline = workspace.run(mode="full", output_name="selective-output")
    assert baseline["reuse_ledger"]["visual"]["recomputed_count"] == 8

    second = workspace.run(mode="incremental", output_name="selective-output")
    assert second["reproducibility_sha256"] == baseline["reproducibility_sha256"]
    for stage in ("visual", "text", "alignment", "retrieval"):
        assert second["reuse_ledger"][stage]["recomputed_count"] == 0
        assert second["reuse_ledger"][stage]["reused_count"] >= 1

    workspace.edit_config(visual_weight=0.5, text_weight=0.5)
    weight_changed = workspace.run(mode="incremental", output_name="selective-output")
    assert weight_changed["reuse_ledger"]["visual"]["recomputed_count"] == 0
    assert weight_changed["reuse_ledger"]["text"]["recomputed_count"] == 0
    assert weight_changed["reuse_ledger"]["alignment"]["recomputed_count"] == 8
    assert weight_changed["reuse_ledger"]["retrieval"]["recomputed_count"] == 1
    assert "E_SNAPSHOT_STALE" in weight_changed["business_codes"]

    _replace_image(workspace.samples / "sample-01" / "image.png", (20, 160, 90))
    image_changed = workspace.run(mode="incremental", output_name="selective-output")
    assert image_changed["reuse_ledger"]["visual"]["recomputed_count"] == 1
    assert image_changed["reuse_ledger"]["text"]["recomputed_count"] == 0
    assert image_changed["reuse_ledger"]["alignment"]["recomputed_count"] == 1
    assert image_changed["reuse_ledger"]["retrieval"]["recomputed_count"] == 1

    _edit_assertion(workspace.samples / "sample-02" / "text.json")
    assertion_changed = workspace.run(mode="incremental", output_name="selective-output")
    assert assertion_changed["reuse_ledger"]["visual"]["recomputed_count"] == 0
    assert assertion_changed["reuse_ledger"]["text"]["recomputed_count"] == 1
    assert assertion_changed["reuse_ledger"]["alignment"]["recomputed_count"] == 1
    assert assertion_changed["reuse_ledger"]["retrieval"]["recomputed_count"] == 1
