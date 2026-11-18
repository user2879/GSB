from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from .engine import RemixEngine, compare_signatures, render_signature
from .fingerprint import digest_json
from .injections import Injection


def _clip_index(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(clip["clip_id"]): clip for clip in config["clips"]}


def apply_edit(
    config: dict[str, Any],
    edit: dict[str, Any],
    *,
    samples_root: str | Path,
    sample_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated = copy.deepcopy(config)
    clips = _clip_index(updated)
    op = str(edit["op"])
    clip_id = str(edit["clip_id"])
    if clip_id not in clips:
        raise ValueError(f"edit {edit.get('id')} references unknown clip {clip_id}")
    clip = clips[clip_id]
    before: dict[str, Any] = {"op": op, "clip_id": clip_id}
    if op == "set_gain":
        track = str(edit["track"])
        if track not in clip["tracks"]:
            raise ValueError(f"edit {edit.get('id')} references unknown track {track}")
        before["gain"] = clip["tracks"][track]["gain"]
        clip["tracks"][track]["gain"] = edit["gain"]
        before["new_gain"] = edit["gain"]
    elif op == "set_mute":
        track = str(edit["track"])
        if track not in clip["tracks"]:
            raise ValueError(f"edit {edit.get('id')} references unknown track {track}")
        before["mute"] = clip["tracks"][track]["mute"]
        before["new_mute"] = bool(edit["mute"])
        clip["tracks"][track]["mute"] = bool(edit["mute"])
    elif op == "set_trim":
        target = str(edit.get("target", "clip"))
        if target == "track":
            track = str(edit["track"])
            settings = clip["tracks"][track]
            before["track"] = track
            before["start_ms"] = settings["start_ms"]
            before["end_ms"] = settings["end_ms"]
            if "start_ms" in edit:
                settings["start_ms"] = float(edit["start_ms"])
            if "end_ms" in edit:
                settings["end_ms"] = (
                    None if edit["end_ms"] is None else float(edit["end_ms"])
                )
            before["new_start_ms"] = settings["start_ms"]
            before["new_end_ms"] = settings["end_ms"]
        else:
            before["start_ms"] = clip["start_ms"]
            before["end_ms"] = clip["end_ms"]
            if "start_ms" in edit:
                clip["start_ms"] = float(edit["start_ms"])
            if "end_ms" in edit:
                clip["end_ms"] = None if edit["end_ms"] is None else float(edit["end_ms"])
            before["new_start_ms"] = clip["start_ms"]
            before["new_end_ms"] = clip["end_ms"]
    elif op == "replace_track":
        track = str(edit["track"])
        if track not in clip["tracks"]:
            raise ValueError(f"edit {edit.get('id')} references unknown track {track}")
        wav = str(edit["wav"])
        path = Path(wav)
        if not path.is_absolute():
            path = Path(samples_root) / path
        if not path.is_file():
            raise ValueError(f"edit {edit.get('id')} replacement does not exist: {path}")
        before["track"] = track
        before["replacement"] = wav
        updated.setdefault("track_overrides", {}).setdefault(clip_id, {})[track] = wav
    elif op == "set_format_version":
        version = str(edit["format_version"])
        if version not in {"v1", "v2", "v3"}:
            raise ValueError(f"edit {edit.get('id')} has invalid format version {version}")
        manifests = {
            item["clip_id"]: item for item in sample_manifest.get("clips", [])
        }
        package = (
            manifests.get(clip_id, {}).get("packages", {}).get(version)
            or f"{clip_id}/packages/{version}"
        )
        before["format_version"] = clip["source_format"]
        before["source_package"] = clip["source_package"]
        before["new_format_version"] = version
        before["new_source_package"] = package
        clip["source_format"] = version
        clip["source_package"] = package
    else:
        raise ValueError(f"unsupported edit operation {op!r}")
    before["edit_id"] = str(edit.get("id", op))
    return updated, before


def run_edit_sequence(
    *,
    samples_root: str | Path,
    base_config: dict[str, Any],
    edits: list[dict[str, Any]],
    workdir: str | Path,
    sample_manifest: dict[str, Any],
    injections: list[Injection] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(base_config)
    audit: list[dict[str, Any]] = []
    last_incremental = None
    last_full = None
    current_config_hash = digest_json(config)
    for edit in edits:
        edit_id = str(edit.get("id", edit["op"]))
        config, change = apply_edit(
            config,
            edit,
            samples_root=samples_root,
            sample_manifest=sample_manifest,
        )
        incremental = RemixEngine(
            samples_root=samples_root,
            config=config,
            workdir=workdir,
            injections=injections or [],
            quiet=quiet,
        ).execute()
        if incremental.stopped:
            return {
                "stopped": True,
                "audit": audit,
                "edit_id": edit_id,
                "config": config,
            }
        assert incremental.report is not None
        with tempfile.TemporaryDirectory(
            prefix=f"full_{edit_id}_", dir=str(workdir)
        ) as full_dir:
            full = RemixEngine(
                samples_root=samples_root,
                config=config,
                workdir=full_dir,
                injections=injections or [],
                force_workers=1,
                quiet=quiet,
            ).execute()
            if full.stopped or full.report is None:
                raise RuntimeError(f"full reference render stopped for edit {edit_id}")
            comparison = compare_signatures(
                render_signature(incremental.report, incremental.workdir),
                render_signature(full.report, full.workdir),
            )
            comparison.update(
                {
                    "edit_id": edit_id,
                    "incremental_workers": incremental.report["thread_pool_size"],
                    "full_workers": 1,
                    "cache_event_order_matches": (
                        incremental.report["cache_event_order"]
                        == full.report["cache_event_order"]
                    ),
                    "failure_order_matches": (
                        incremental.report["failure_order"]
                        == full.report["failure_order"]
                    ),
                    "incremental_cache_summary": incremental.report["cache_summary"],
                    "full_cache_summary": full.report["cache_summary"],
                }
            )
        audit_path = workdir / "edit_reports" / f"{edit_id}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_record = {
            "edit_id": edit_id,
            "change": change,
            "config_hash": digest_json(config),
            "incremental_report": str(workdir / "report.json"),
            "full_report_signature": digest_json(render_signature(full.report, full.workdir)),
            "incremental_report_signature": digest_json(
                render_signature(incremental.report, incremental.workdir)
            ),
            "comparison": comparison,
            "cache_summary": incremental.report["cache_summary"],
            "snapshot_version": incremental.report["snapshots"]["global_version"],
        }
        audit_path.write_text(
            json.dumps(audit_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit.append(audit_record)
        last_incremental = incremental.report
        last_full = full.report
        current_config_hash = digest_json(config)
        if not comparison["matches"]:
            raise AssertionError(
                f"incremental/full render mismatch after edit {edit_id}: {comparison}"
            )
    if last_incremental is None or last_full is None:
        raise ValueError("edit sequence is empty")
    comparison = compare_signatures(
        render_signature(last_incremental, workdir),
        render_signature(last_full, workdir),
    )
    comparison.update(
        {
            "edit_id": audit[-1]["edit_id"],
            "incremental_workers": last_incremental["thread_pool_size"],
            "full_workers": 1,
            "cache_event_order_matches": (
                last_incremental["cache_event_order"]
                == last_full["cache_event_order"]
            ),
            "failure_order_matches": (
                last_incremental["failure_order"] == last_full["failure_order"]
            ),
            "incremental_cache_summary": last_incremental["cache_summary"],
            "full_cache_summary": last_full["cache_summary"],
        }
    )
    final_report = dict(last_incremental)
    final_report["run_mode"] = "edit_sequence"
    final_report["config_hash"] = current_config_hash
    final_report["edit_sequence"] = {
        "base_config_hash": digest_json(base_config),
        "edits": audit,
        "edit_count": len(audit),
        "all_edits_match_full": all(
            item["comparison"]["matches"] for item in audit
        ),
        "all_event_orders_match_serial": all(
            item["comparison"]["cache_event_order_matches"] for item in audit
        ),
        "all_failure_orders_match_serial": all(
            item["comparison"]["failure_order_matches"] for item in audit
        ),
        "final_config": config,
    }
    final_report["concurrency_comparison"] = {
        **comparison,
        "mode": "configured thread pool vs blank-workspace single-thread full render",
    }
    final_report["final_incremental_vs_full"] = comparison
    final_report["render_signature_sha256"] = digest_json(
        render_signature(final_report, workdir)
    )
    return {
        "stopped": False,
        "audit": audit,
        "report": final_report,
        "config": config,
        "comparison": comparison,
    }

