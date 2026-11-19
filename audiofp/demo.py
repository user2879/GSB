"""One-command deterministic acceptance demonstration and report writer."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audio import sha256_file
from .config import DEFAULT_CONFIG, LEGACY_CONFIG, FingerprintConfig
from .errors import AudioWorkbenchError
from .fixtures import generate_fixtures, load_manifest
from .library import Library


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _step(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value["content_hash"] = canonical_hash(
        {key: item for key, item in value.items() if key != "content_hash"}
    )
    return value


def _tree_hashes(root: str | Path) -> dict[str, str]:
    base = Path(root)
    return {
        path.relative_to(base).as_posix(): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
    }


def _tree_digest(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(hashes.items()):
        digest.update(f"{path}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def _format_paths(manifest: dict[str, Any], root: Path) -> list[tuple[str, str, Path]]:
    values: list[tuple[str, str, Path]] = []
    for track in manifest["tracks"]:
        for entry in track["formats"]:
            values.append(
                (
                    track["truth_id"],
                    entry["label"],
                    root / entry["relative_path"],
                )
            )
    return sorted(values, key=lambda item: (item[0], item[1]))


def _group_truth_ids(
    groups: list[dict[str, Any]],
    truth_by_path: dict[str, str],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for group in groups:
        mapping[group["group_id"]] = sorted(
            {
                truth_by_path[source["relative_path"]]
                for source in group["sources"]
                if source["relative_path"] in truth_by_path
            }
        )
    return mapping


def _find_group_for_truth(groups: list[dict[str, Any]], truth_id: str) -> str | None:
    for group in groups:
        labels = {source["relative_path"].split("/")[-1].split("_")[0] for source in group["sources"]}
        if truth_id in labels:
            return group["group_id"]
    return None


def _error_code(callable_obj: Any) -> tuple[str | None, dict[str, Any] | None]:
    try:
        callable_obj()
    except AudioWorkbenchError as exc:
        return exc.code, exc.to_dict()
    return None, None


def _import_all(library: Library, manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    imported: list[dict[str, Any]] = []
    for truth_id, label, path in _format_paths(manifest, root):
        display_label = next(
            track["title"] for track in manifest["tracks"] if track["truth_id"] == truth_id
        )
        item = library.import_file(path, root=root, display_label=display_label)
        imported.append(
            {
                "truth_id": truth_id,
                "format_label": label,
                **item.to_dict(),
            }
        )
    return imported


def _query_signature(result: dict[str, Any], display_label_key: str = "display_label") -> list[dict[str, Any]]:
    return [
        {
            "label": item[display_label_key],
            "score": item["score"],
            "votes": item["votes"],
            "alignment_start_seconds": item["alignment_start_seconds"],
            "alignment_duration_seconds": item["alignment_duration_seconds"],
            "format_signatures": sorted(
                source["format_signature"] for source in item["formats"]
            ),
        }
        for item in result["results"]
    ]


def _fingerprint_row_set(store: Any) -> list[tuple[Any, ...]]:
    rows = store.connection.execute(
        "SELECT s.relative_path, f.fingerprint_hash, f.anchor_frame, f.target_frame, "
        "f.band1, f.band2, f.delta_frames FROM fingerprints f "
        "JOIN sources s ON s.source_id=f.source_id "
        "ORDER BY s.relative_path, f.anchor_frame, f.target_frame, f.band1, f.band2, "
        "f.fingerprint_hash"
    ).fetchall()
    return [tuple(row) for row in rows]


def _run_pipeline(output_dir: str | Path, seed: int) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    fixtures = output / "fixtures"
    indexes = output / "index"
    indexes.mkdir(parents=True)
    reports = output / "reports"
    reports.mkdir()
    report: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "config": DEFAULT_CONFIG.to_dict(),
        "config_digest": DEFAULT_CONFIG.digest(),
        "ok": False,
        "steps": {},
        "artifacts": {},
    }

    manifest = generate_fixtures(fixtures, seed=seed, clean=True)
    report["steps"]["fixtures"] = _step(
        {
            "ok": True,
            "manifest_relative_path": "fixtures/manifest.json",
            "manifest_sha256": sha256_file(fixtures / "manifest.json"),
            "track_count": len(manifest["tracks"]),
            "format_count": sum(len(track["formats"]) for track in manifest["tracks"]),
            "broken_sample_count": len(manifest["broken_samples"]),
            "variant_count": len(manifest["variants"]),
        }
    )

    main_index = indexes / "current.db"
    imported: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    group_truth: dict[str, list[str]] = {}
    dedup_report: dict[str, Any] = {}
    ranking_reports: dict[str, Any] = {}
    segment_reports: dict[str, Any] = {}
    variant_reports: list[dict[str, Any]] = []
    broken_reports: list[dict[str, Any]] = []

    with Library(main_index, DEFAULT_CONFIG) as library:
        for track in manifest["tracks"]:
            items = [
                (entry["label"], fixtures / entry["relative_path"])
                for entry in track["formats"]
            ]
            pair_report = library.consistency_report(items)
            consistency.append(
                {
                    "truth_id": track["truth_id"],
                    "threshold": pair_report["threshold"],
                    "passed": pair_report["passed"],
                    "pairs": pair_report["pairs"],
                }
            )

        imported = _import_all(library, manifest, fixtures)
        groups = library.store.list_groups()
        group_truth = _group_truth_ids(
            groups,
            {
                path: truth_id
                for path, truth_id in manifest["truth_by_relative_path"].items()
                if truth_id in {track["truth_id"] for track in manifest["tracks"]}
            },
        )
        dedup_report = library.validate_dedup_truth(
            {
                path: truth_id
                for path, truth_id in manifest["truth_by_relative_path"].items()
                if truth_id in {track["truth_id"] for track in manifest["tracks"]}
            }
        )

        for track in manifest["tracks"]:
            queries = [
                (entry["label"], fixtures / entry["relative_path"])
                for entry in track["formats"]
            ]
            ranking = library.compare_query_rankings(queries, top_k=5)
            top_truth = None
            if ranking["runs"] and ranking["runs"][0]["results"]:
                top_group = ranking["runs"][0]["results"][0]["group_id"]
                top_truth = group_truth.get(top_group)
            ranking_reports[track["truth_id"]] = {
                "ok": bool(ranking["ok"]) and top_truth == [track["truth_id"]],
                "format_rank_consistent": ranking["ok"],
                "first_rank_difference": ranking["failure"],
                "top_group_truth_ids": top_truth,
                "runs": [
                    {
                        "label": run["label"],
                        "candidate_order": run["candidate_order"],
                        "results": _query_signature(run),
                    }
                    for run in ranking["runs"]
                ],
            }

        query_entries = manifest["query"]["formats"]
        segment_ranking = library.compare_query_rankings(
            [(entry["label"], fixtures / entry["relative_path"]) for entry in query_entries],
            top_k=5,
        )
        segment_details: list[dict[str, Any]] = []
        truth_starts = manifest["query"]["expected_alignment_starts_seconds"]
        for run in segment_ranking["runs"]:
            if not run["results"]:
                segment_details.append({"label": run["label"], "ok": False, "reason": "no_result"})
                continue
            top = run["results"][0]
            top_truth = group_truth.get(top["group_id"], [])
            aligned_start = float(top["alignment_start_seconds"])
            alignment_error = min(abs(aligned_start - start) for start in truth_starts)
            segment_details.append(
                {
                    "label": run["label"],
                    "ok": top_truth == ["aurora"] and alignment_error <= 0.08,
                    "top_group_truth_ids": top_truth,
                    "alignment_start_seconds": aligned_start,
                    "alignment_error_seconds": round(alignment_error, 6),
                    "alignment_duration_seconds": top["alignment_duration_seconds"],
                    "score": top["score"],
                    "candidate_order": run["candidate_order"],
                }
            )
        segment_reports = {
            "ok": bool(segment_ranking["ok"])
            and all(item.get("ok", False) for item in segment_details),
            "threshold_seconds": 0.08,
            "format_rank_consistent": segment_ranking["ok"],
            "first_rank_difference": segment_ranking["failure"],
            "runs": segment_details,
        }

        for variant in manifest["variants"]:
            result = library.query_file(fixtures / variant["relative_path"], top_k=3)
            top_truth = group_truth.get(result["results"][0]["group_id"], []) if result["results"] else []
            top = result["results"][0] if result["results"] else None
            variant_reports.append(
                {
                    "relative_path": variant["relative_path"],
                    "kind": variant["kind"],
                    "truth_group": variant["variant_of"],
                    "ok": top_truth == [variant["variant_of"]] and top is not None,
                    "top_group_truth_ids": top_truth,
                    "top_score": top["score"] if top else 0.0,
                    "transform": top.get("transform") if top else None,
                    "variant_search": result.get("variant_search"),
                }
            )

        for broken in manifest["broken_samples"]:
            code, error = _error_code(
                lambda path=fixtures / broken["relative_path"]: library.query_file(path)
            )
            broken_reports.append(
                {
                    "relative_path": broken["relative_path"],
                    "expected_error": broken["expected_error"],
                    "actual_error": code,
                    "ok": code == broken["expected_error"],
                    "details": error,
                }
            )

        index_stats = library.store.stats()
        index_meta = library.store.metadata()

    report["steps"]["decode_extract_import"] = _step(
        {
            "ok": len(imported) == sum(len(track["formats"]) for track in manifest["tracks"]),
            "imported": imported,
            "index": index_stats,
            "groups": groups,
        }
    )
    report["steps"]["format_consistency"] = _step(
        {
            "ok": bool(consistency) and all(item["passed"] for item in consistency),
            "threshold": DEFAULT_CONFIG.format_hit_threshold,
            "tracks": consistency,
        }
    )
    report["steps"]["dedup"] = _step(
        {
            "ok": bool(dedup_report.get("ok")),
            "summary": dedup_report,
            "group_truth_ids": group_truth,
        }
    )
    report["steps"]["query_ranking"] = _step(
        {
            "ok": all(item["ok"] for item in ranking_reports.values()),
            "tracks": ranking_reports,
        }
    )
    report["steps"]["segment_alignment"] = _step(segment_reports)
    report["steps"]["variant_retrieval"] = _step(
        {
            "ok": all(item["ok"] for item in variant_reports),
            "variants": variant_reports,
        }
    )
    report["steps"]["broken_samples"] = _step(
        {
            "ok": all(item["ok"] for item in broken_reports),
            "items": broken_reports,
        }
    )

    # Controlled negative contract: distinct truth labels inside one group must
    # produce E_DEDUP_MERGE with fingerprint evidence.
    with Library(main_index, DEFAULT_CONFIG) as library:
        aurora_group = _find_group_for_truth(library.store.list_groups(), "aurora")
        fake_truth = dict(
            {
                path: truth_id
                for path, truth_id in manifest["truth_by_relative_path"].items()
                if truth_id in {track["truth_id"] for track in manifest["tracks"]}
            }
        )
        if aurora_group is not None:
            group = library.store.group_details(aurora_group)
            if group:
                fake_truth[group["sources"][1]["relative_path"]] = "false_merge_probe"
        dedup_code, dedup_error = _error_code(lambda: library.validate_dedup_truth(fake_truth))

    wrong_pair_items = [
        ("aurora_wav", fixtures / "audio" / "aurora_44k_16b_mono.wav"),
        ("ember_wav", fixtures / "audio" / "ember_44k_16b_mono.wav"),
    ]
    with Library(main_index, DEFAULT_CONFIG) as library:
        drift_code, drift_error = _error_code(
            lambda: library.consistency_report(wrong_pair_items)
        )

    report["steps"]["negative_error_contracts"] = _step(
        {
            "ok": dedup_code == "E_DEDUP_MERGE" and drift_code == "E_FORMAT_DRIFT",
            "dedup_merge_probe": {"code": dedup_code, "error": dedup_error},
            "format_drift_probe": {"code": drift_code, "error": drift_error},
        }
    )

    # Build a legacy index, make an unmigratable copy, then migrate and compare
    # against a fresh fp-v2 rebuild.
    legacy_index = indexes / "legacy.db"
    with Library(legacy_index, LEGACY_CONFIG) as legacy:
        _import_all(legacy, manifest, fixtures)
        legacy_stats_before = legacy.store.stats()
    unmigratable_index = indexes / "unmigratable.db"
    shutil.copy2(legacy_index, unmigratable_index)

    with Library(legacy_index, DEFAULT_CONFIG, auto_migrate=True) as migrated:
        migrated_stats = migrated.store.stats()
        migrated_meta = migrated.store.metadata()
        migrated_rows = _fingerprint_row_set(migrated.store)
        migrated_queries = [
            migrated.query_file(fixtures / entry["relative_path"], top_k=5)
            for entry in query_entries
        ]
    with Library(legacy_index, DEFAULT_CONFIG, auto_migrate=True) as migrated_again:
        idempotent_stats = migrated_again.store.stats()
        idempotent_meta = migrated_again.store.metadata()

    rebuilt_index = indexes / "rebuilt.db"
    with Library(rebuilt_index, DEFAULT_CONFIG) as rebuilt:
        _import_all(rebuilt, manifest, fixtures)
        rebuilt_stats = rebuilt.store.stats()
        rebuilt_rows = _fingerprint_row_set(rebuilt.store)
        rebuilt_queries = [
            rebuilt.query_file(fixtures / entry["relative_path"], top_k=5)
            for entry in query_entries
        ]

    migrated_signatures = [_query_signature(result) for result in migrated_queries]
    rebuilt_signatures = [_query_signature(result) for result in rebuilt_queries]
    row_set_equal = sorted(migrated_rows) == sorted(rebuilt_rows)
    retrieval_equal = migrated_signatures == rebuilt_signatures
    migrated_content_hash_equal = (
        migrated_stats["content_hash"] == rebuilt_stats["content_hash"]
    )
    incompatible = replace(
        LEGACY_CONFIG,
        version="fp-v9-incompatible",
        n_fft=1024,
        encoding_space="incompatible-stft",
    )
    incompatible_code, incompatible_error = _error_code(
        lambda: Library(unmigratable_index, incompatible, auto_migrate=True).close()
    )
    report["steps"]["index_migration"] = _step(
        {
            "ok": bool(
                row_set_equal
                and retrieval_equal
                and migrated_content_hash_equal
                and incompatible_code == "E_INDEX_VERSION"
            )
            and idempotent_stats["revision"] == migrated_stats["revision"]
            and idempotent_stats["content_hash"] == migrated_stats["content_hash"],
            "legacy_stats": legacy_stats_before,
            "migrated_stats": migrated_stats,
            "migrated_fingerprint_version": migrated_meta["fingerprint_version"],
            "idempotent_stats": idempotent_stats,
            "idempotent_content_hash_equal": idempotent_stats["content_hash"]
            == migrated_stats["content_hash"],
            "rebuilt_stats": rebuilt_stats,
            "fingerprint_row_set_equal": row_set_equal,
            "retrieval_equal": retrieval_equal,
            "logical_content_hash_equal": migrated_content_hash_equal,
            "migrated_query_signatures": migrated_signatures,
            "rebuilt_query_signatures": rebuilt_signatures,
            "incompatible_open": {"code": incompatible_code, "error": incompatible_error},
        }
    )

    # Incremental import and deletion on a separate database.
    incremental_index = indexes / "incremental.db"
    with Library(incremental_index, DEFAULT_CONFIG) as incremental:
        aurora_paths = [
            fixtures / entry["relative_path"]
            for track in manifest["tracks"]
            if track["truth_id"] == "aurora"
            for entry in track["formats"]
        ]
        for path in aurora_paths:
            incremental.import_file(path, root=fixtures, display_label="Aurora Circuit")
        before = incremental.query_file(
            fixtures / query_entries[1]["relative_path"], top_k=5, search_variants=False
        )
        cinder_paths = [
            fixtures / entry["relative_path"]
            for track in manifest["tracks"]
            if track["truth_id"] == "cinder"
            for entry in track["formats"]
        ]
        for path in cinder_paths:
            incremental.import_file(path, root=fixtures, display_label="Cinder Atlas")
        after_import = incremental.query_file(
            fixtures / query_entries[1]["relative_path"], top_k=5, search_variants=False
        )
        cinder_group = _find_group_for_truth(incremental.store.list_groups(), "cinder")
        before_delete_stats = incremental.store.stats()
        deletion = incremental.store.delete_group(cinder_group or "")
        after_delete = incremental.query_file(
            fixtures / query_entries[1]["relative_path"], top_k=5, search_variants=False
        )
        after_delete_stats = incremental.store.stats()
        deleted_count = incremental.store.connection.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE group_id=?", (cinder_group,)
        ).fetchone()[0]
        after_import_group_fingerprint_count = int(
            sum(group["fingerprint_count"] for group in incremental.store.list_groups())
        )

    def _aurora_item(result: dict[str, Any]) -> dict[str, Any] | None:
        return next(
            (item for item in result["results"] if group_truth.get(item["group_id"]) == ["aurora"]),
            None,
        )

    before_item = _aurora_item(before)
    after_import_item = _aurora_item(after_import)
    after_delete_item = _aurora_item(after_delete)
    incremental_ok = (
        before_item is not None
        and after_import_item is not None
        and after_delete_item is not None
        and before_item["votes"] == after_import_item["votes"]
        and before_item["alignment_start_seconds"] == after_import_item["alignment_start_seconds"]
        and before_item["votes"] == after_delete_item["votes"]
        and before_item["alignment_start_seconds"] == after_delete_item["alignment_start_seconds"]
        and deletion["deleted_sources"] == 3
        and deleted_count == 0
    )
    report["steps"]["incremental_maintenance"] = _step(
        {
            "ok": bool(incremental_ok),
            "before_import_stats": {"groups": 1, "sources": 3},
            "before_import_query": before_item,
            "after_import_query": after_import_item,
            "after_import_stats": {
                "groups": 2,
                "sources": 6,
                "fingerprints": after_import_group_fingerprint_count,
            },
            "deletion": deletion,
            "deleted_group_fingerprint_rows": int(deleted_count),
            "after_delete_query": after_delete_item,
            "after_delete_stats": after_delete_stats,
        }
    )

    # Crash recovery: a child commits nothing and exits abruptly; SQLite must
    # roll back, and an injected orphan temp file must be reclaimed.
    crash_index = indexes / "crash.db"
    shutil.copy2(main_index, crash_index)
    with Library(crash_index, DEFAULT_CONFIG) as crash_probe:
        crash_before = crash_probe.store.stats()
    orphan = Path(str(crash_index) + ".tmp-orphan-crash-probe")
    orphan.write_text("orphan", encoding="utf-8")
    child_script = r"""
import os, sqlite3, sys
path = sys.argv[1]
conn = sqlite3.connect(path)
conn.execute("PRAGMA foreign_keys=ON")
conn.execute("BEGIN IMMEDIATE")
conn.execute("INSERT INTO work_groups(group_id, display_label, fingerprint_count, duration) VALUES ('grp_crash','crash',1,1.0)")
conn.execute("INSERT INTO sources(source_id, group_id, relative_path, content_sha256, fingerprint_content_hash, container, codec, sample_rate, bit_depth, channels, declared_frames, decoded_frames, duration, format_signature, fingerprint_count, import_sequence, format_json) VALUES ('src_crash','grp_crash','crash.wav','x','x','WAV','PCM_16',44100,16,1,1,1,1.0,'wav:pcm_16:44100:16:1',1,999,'{}')")
conn.execute("INSERT INTO fingerprints(fingerprint_hash, source_id, group_id, anchor_frame, target_frame, band1, band2, delta_frames) VALUES ('crashed', 'src_crash', 'grp_crash', 0, 1, 1, 1, 1)")
os._exit(42)
"""
    child = subprocess.run(
        [sys.executable, "-c", child_script, str(crash_index)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    with Library(crash_index, DEFAULT_CONFIG) as recovered:
        crash_after = recovered.store.stats()
        crashed_rows = recovered.store.connection.execute(
            "SELECT COUNT(*) FROM fingerprints WHERE fingerprint_hash='crashed'"
        ).fetchone()[0]
    report["steps"]["crash_recovery"] = _step(
        {
            "ok": child.returncode == 42
            and crash_after["groups"] == crash_before["groups"]
            and crash_after["sources"] == crash_before["sources"]
            and crash_after["fingerprints"] == crash_before["fingerprints"]
            and crash_after["content_hash"] == crash_before["content_hash"]
            and int(crashed_rows) == 0
            and bool(crash_after["recovered_orphans"]),
            "child_returncode": child.returncode,
            "before": crash_before,
            "after": crash_after,
            "uncommitted_rows_visible": int(crashed_rows),
            "injected_orphan": orphan.name,
        }
    )

    # Record stable artifact hashes before writing the report itself.
    artifact_hashes = _tree_hashes(output)
    report["artifacts"] = artifact_hashes
    report["ok"] = all(bool(step.get("ok")) for step in report["steps"].values())
    report["index_metadata"] = index_meta
    report["generated_index_file_sha256"] = sha256_file(main_index)
    report_path = reports / "demo_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown_report(report, reports / "demo_report.md")
    return report


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Audio Fingerprint Workbench Validation",
        "",
        f"- Overall: **{'PASS' if report['ok'] else 'FAIL'}**",
        f"- Seed: `{report['seed']}`",
        f"- Fingerprint version: `{report['config']['version']}`",
        f"- Config digest: `{report['config_digest']}`",
        "",
        "## Steps",
        "",
        "| Step | Result | Content hash |",
        "| --- | --- | --- |",
    ]
    for name, step in report["steps"].items():
        lines.append(
            f"| `{name}` | {'PASS' if step.get('ok') else 'FAIL'} | `{step.get('content_hash', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Key measurements",
            "",
        ]
    )
    consistency = report["steps"]["format_consistency"]
    for track in consistency["tracks"]:
        rates = ", ".join(
            f"{pair['pair']}={pair['hit_rate']:.4f}" for pair in track["pairs"]
        )
        lines.append(f"- Format consistency `{track['truth_id']}`: {rates}")
    segment = report["steps"]["segment_alignment"]
    for run in segment["runs"]:
        lines.append(
            f"- Segment query `{run['label']}`: score={run['score']:.4f}, "
            f"aligned={run['alignment_start_seconds']:.4f}s, error={run['alignment_error_seconds']:.4f}s"
        )
    migration = report["steps"]["index_migration"]
    lines.append(
        f"- Migration: row_set_equal={migration['fingerprint_row_set_equal']}, "
        f"retrieval_equal={migration['retrieval_equal']}, "
        f"idempotent={migration['idempotent_content_hash_equal']}"
    )
    incremental = report["steps"]["incremental_maintenance"]
    lines.append(
        f"- Incremental delete: sources={incremental['deletion']['deleted_sources']}, "
        f"remaining fingerprint rows={incremental['deleted_group_fingerprint_rows']}"
    )
    crash = report["steps"]["crash_recovery"]
    lines.append(
        f"- Crash recovery: child_rc={crash['child_returncode']}, "
        f"uncommitted_rows={crash['uncommitted_rows_visible']}, "
        f"orphans={len(crash['after']['recovered_orphans'])}"
    )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The CLI verifies a second independent run byte-for-byte and writes",
            "`reports/repeatability.json`. No network or real audio assets are used.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_demo(
    output_dir: str | Path,
    *,
    seed: int = 20261119,
    verify_repeat: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    report = _run_pipeline(output, seed)
    repeatability: dict[str, Any] = {
        "requested": verify_repeat,
        "ok": None,
        "first_tree_digest": None,
        "second_tree_digest": None,
        "differing_files": [],
    }
    if verify_repeat:
        first_hashes = _tree_hashes(output)
        with tempfile.TemporaryDirectory(prefix="audiofp-repeat-") as temporary:
            _run_pipeline(temporary, seed)
            second_hashes = _tree_hashes(temporary)
        all_paths = sorted(set(first_hashes) | set(second_hashes))
        differing = [
            path
            for path in all_paths
            if first_hashes.get(path) != second_hashes.get(path)
        ]
        repeatability = {
            "requested": True,
            "ok": not differing,
            "first_tree_digest": _tree_digest(first_hashes),
            "second_tree_digest": _tree_digest(second_hashes),
            "differing_files": differing,
            "file_count": len(all_paths),
        }
        repeat_path = output / "reports" / "repeatability.json"
        repeat_path.write_text(
            json.dumps(repeatability, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    report["repeatability"] = repeatability
    return report
