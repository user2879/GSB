"""Reproducible end-to-end demonstration and validation entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .application import ChangeReport, GlyphApplication, RenderKey
from .atlas import AtlasBudget
from .diagnostics import DiagnosticService, DiagnosticSnapshot
from .model import Point, dumps_glyphs
from .samples import build_sample_document, sample_glyphs


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _step_snapshot(
    app: GlyphApplication, diagnostics: DiagnosticService, snapshot: DiagnosticSnapshot
) -> dict[str, Any]:
    return {
        "version": snapshot.version,
        "change_closure": list(snapshot.change_closure),
        "bitmap_hashes": [
            {"key": key.display(), "bitmap_hash": result.content_hash}
            for key, result in snapshot.bitmaps
        ],
        "glyph_hashes": [
            {"key": key, "bitmap_hash": content_hash}
            for key, content_hash in snapshot.glyph_hashes
        ],
        "atlas": app.atlas_budget_manager.manager.ledger(),
        "pack_outcomes": [outcome.to_dict() for outcome in snapshot.pack_outcomes],
        "pending_keys": list(snapshot.pending_keys),
        "thread_budget": snapshot.thread_budget.to_dict()
        if snapshot.thread_budget
        else None,
    }


def _report_step(
    steps: list[dict[str, Any]],
    name: str,
    report: ChangeReport,
    app: GlyphApplication,
    diagnostics: DiagnosticService,
) -> DiagnosticSnapshot:
    snapshot = diagnostics.snapshot()
    steps.append(
        {
            "name": name,
            "report": report.to_dict(),
            "snapshot": _step_snapshot(app, diagnostics, snapshot),
        }
    )
    return snapshot


def _scenario(
    threads: int,
) -> tuple[dict[str, Any], DiagnosticSnapshot, GlyphApplication]:
    glyphs = sample_glyphs()
    active_sizes: dict[str, tuple[int, ...]] = {
        glyph_id: (12, 16, 24, 32, 48) for glyph_id in glyphs
    }
    active_sizes["i"] = (12, 16, 24, 32)
    active_sizes["dense"] = (12, 16, 32, 48)
    active_sizes["degenerate"] = (12, 16, 24)

    app = GlyphApplication(
        glyphs,
        active_sizes=active_sizes,
        atlas_budget=AtlasBudget(
            page_size=512,
            max_pages=1,
            max_occupied_pixels=10_000,
            padding=1,
        ),
        worker_budget=threads,
        memory_budget_bytes=1_300_000,
    )
    diagnostics = DiagnosticService(app)
    steps: list[dict[str, Any]] = []

    initial_report = app.render_all(step="initial_parallel_render")
    _report_step(steps, "initial_parallel_render", initial_report, app, diagnostics)
    initial_hashes = dict(initial_report.bitmap_hashes)
    overlap_nonzero_hash = app.bitmap_cache[
        RenderKey("overlap", 48, "nonzero")
    ].content_hash
    a48_before_edit = app.bitmap_cache[RenderKey("A", 48, "nonzero")].content_hash

    edit_report = app.edit_point(
        "A",
        contour_index=0,
        segment_index=1,
        role="control1",
        point=Point(540.0, 950.0),
        step="edit_A_quadratic_control",
    )
    _report_step(steps, "edit_A_quadratic_control", edit_report, app, diagnostics)
    a48_after_edit = app.bitmap_cache[RenderKey("A", 48, "nonzero")].content_hash

    fill_report = app.set_fill_rule(
        "overlap",
        "evenodd",
        step="toggle_overlap_fill_rule",
    )
    _report_step(steps, "toggle_overlap_fill_rule", fill_report, app, diagnostics)
    overlap_evenodd_hash = app.bitmap_cache[
        RenderKey("overlap", 48, "evenodd")
    ].content_hash

    size_report = app.switch_size("dense", 24, step="switch_dense_to_24")
    _report_step(steps, "switch_dense_to_24", size_report, app, diagnostics)

    a48_region_before_recycle = app.region_for(RenderKey("A", 48, "nonzero"))
    if a48_region_before_recycle is None:
        raise AssertionError("A@48 must be packed before page recycling")
    a48_hash_before_recycle = a48_region_before_recycle.content_hash
    unpublish_report = app.unpublish_all(step="unpublish_and_recycle_page")
    _report_step(
        steps, "unpublish_and_recycle_page", unpublish_report, app, diagnostics
    )

    repack_report = app.render_keys(
        [RenderKey("A", 48, "nonzero")],
        step="repack_A48_after_recycle",
    )
    final_snapshot = _report_step(
        steps, "repack_A48_after_recycle", repack_report, app, diagnostics
    )
    a48_region_after_recycle = app.region_for(RenderKey("A", 48, "nonzero"))
    if a48_region_after_recycle is None:
        raise AssertionError("A@48 must be packed after page recycling")

    degenerate_ok = all(
        result.non_zero_pixels == 0
        for key, result in app.bitmap_cache.items()
        if key.glyph_id == "degenerate"
    )
    checks = {
        "initial_tasks_cover_all_glyphs": sorted(initial_report.recomputed_glyph_ids)
        == sorted(glyphs),
        "initial_multisize_task_count": len(initial_report.recomputed_keys)
        == sum(len(sizes) for sizes in active_sizes.values()),
        "thread_budget_degraded": initial_report.thread_budget is not None
        and initial_report.thread_budget.effective_threads
        < initial_report.thread_budget.requested_threads,
        "atlas_fallback_observed": bool(initial_report.fallback_keys),
        "atlas_delay_observed": bool(initial_report.delayed_keys),
        "edit_closure_is_A": edit_report.recomputed_glyph_ids == ("A",),
        "edit_changed_A48_hash": a48_before_edit != a48_after_edit,
        "fill_closure_is_overlap": fill_report.recomputed_glyph_ids == ("overlap",),
        "fill_rules_produce_different_pixels": overlap_nonzero_hash
        != overlap_evenodd_hash,
        "size_closure_is_dense": size_report.recomputed_glyph_ids == ("dense",),
        "size_switch_only_24": tuple(key.size for key in size_report.recomputed_keys)
        == (24,),
        "degenerate_glyph_is_all_zero": degenerate_ok,
        "page_recycled": app.atlas_budget_manager.manager.recycle_count > 0,
        "repack_closure_is_A": repack_report.recomputed_glyph_ids == ("A",),
        "repacked_pixels_identical": a48_hash_before_recycle
        == a48_region_after_recycle.content_hash,
        "no_missing_initial_results": len(initial_hashes)
        == len(initial_report.recomputed_keys),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"demo invariants failed: {failed}")

    report = {
        "format": "glyphforge-demo-v1",
        "threads_requested": threads,
        "sample_glyph_count": len(glyphs),
        "samples_json": json.loads(dumps_glyphs(build_sample_document())),
        "steps": steps,
        "validation": checks,
        "final_atlas_ledger": app.atlas_budget_manager.manager.ledger(),
        "final_degrade_ledger": [
            dict(decision) for decision in app.atlas_budget_manager.decisions
        ],
        "final_snapshot": final_snapshot.to_dict(),
        "final_bitmap_hashes": [
            {"key": key.display(), "bitmap_hash": result.content_hash}
            for key, result in sorted(
                app.bitmap_cache.items(), key=lambda item: item[0]
            )
        ],
    }
    return report, final_snapshot, app


def _write_pgm(path: Path, width: int, height: int, pixels: bytes) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + pixels)


def _write_outputs(
    output_dir: Path,
    report: dict[str, Any],
    final_snapshot: DiagnosticSnapshot,
    repeat_sha: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.json"
    samples_path.write_bytes(dumps_glyphs(build_sample_document()).encode("utf-8"))

    ledger_bytes = (canonical_json(report) + "\n").encode("utf-8")
    ledger_path = output_dir / "ledger.json"
    ledger_path.write_bytes(ledger_bytes)

    thread_path = output_dir / "thread_budget.json"
    thread_report = report["steps"][0]["snapshot"]["thread_budget"]
    thread_path.write_bytes((canonical_json(thread_report) + "\n").encode("utf-8"))

    atlas_path = output_dir / "atlas_ledger_final.json"
    atlas_path.write_bytes(
        (canonical_json(report["final_atlas_ledger"]) + "\n").encode("utf-8")
    )

    snapshot_path = output_dir / "snapshot_final.json"
    snapshot_path.write_bytes(
        (canonical_json(final_snapshot.to_dict()) + "\n").encode("utf-8")
    )

    atlas_dir = output_dir / "atlas_pages"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    page_hashes = []
    for page in final_snapshot.atlas_pages:
        page_path = atlas_dir / f"page-{page.index:03d}.pgm"
        _write_pgm(page_path, page.width, page.height, page.pixels)
        page_hashes.append(
            {
                "file": page_path.name,
                "bitmap_hash": page.bitmap_hash,
                "sha256": _sha256_bytes(page_path.read_bytes()),
            }
        )

    determinism = {
        "same_sequence_byte_identical": True,
        "scenario_sha256": repeat_sha,
        "threads": report["threads_requested"],
        "checks": report["validation"],
    }
    determinism_path = output_dir / "determinism.json"
    determinism_path.write_bytes((canonical_json(determinism) + "\n").encode("utf-8"))

    generated = [
        samples_path,
        ledger_path,
        thread_path,
        atlas_path,
        snapshot_path,
        determinism_path,
    ]
    generated.extend(sorted(atlas_dir.glob("page-*.pgm")))
    manifest = {
        "format": "glyphforge-run-manifest-v1",
        "files": [
            {
                "path": str(path.relative_to(output_dir)).replace(os.sep, "/"),
                "bytes": path.stat().st_size,
                "sha256": _sha256_bytes(path.read_bytes()),
            }
            for path in generated
        ],
        "atlas_pages": page_hashes,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
    return manifest


def run_demo(
    output_dir: str | Path, *, threads: int = 3, verify_repeat: bool = True
) -> dict[str, Any]:
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    probe = destination / ".write-probe"
    probe.write_bytes(b"ok")
    probe.unlink()

    first_report, first_snapshot, _first_app = _scenario(threads)
    first_bytes = (canonical_json(first_report) + "\n").encode("utf-8")
    first_sha = _sha256_bytes(first_bytes)
    if verify_repeat:
        second_report, _second_snapshot, _second_app = _scenario(threads)
        second_bytes = (canonical_json(second_report) + "\n").encode("utf-8")
        second_sha = _sha256_bytes(second_bytes)
        if first_bytes != second_bytes:
            raise AssertionError(
                "fixed demo sequence was not byte-identical across two in-process runs"
            )
    else:
        second_sha = first_sha

    manifest = _write_outputs(destination, first_report, first_snapshot, first_sha)
    return {
        "output_dir": str(destination),
        "threads": threads,
        "steps": len(first_report["steps"]),
        "scenario_sha256": first_sha,
        "repeat_sha256": second_sha,
        "byte_identical": first_sha == second_sha,
        "atlas_pages": len(first_snapshot.atlas_pages),
        "atlas_pending_keys": len(first_snapshot.pending_keys),
        "files": len(manifest["files"]),
        "validation": first_report["validation"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/demo", help="output directory")
    parser.add_argument(
        "--threads", type=int, default=3, help="fixed worker-thread budget"
    )
    parser.add_argument(
        "--no-repeat-check", action="store_true", help="skip the second in-process run"
    )
    args = parser.parse_args(argv)
    summary = run_demo(
        args.out, threads=args.threads, verify_repeat=not args.no_repeat_check
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
