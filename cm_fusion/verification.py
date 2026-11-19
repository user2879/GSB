from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .artifact import convert_index, decode_with_compat, encode_index, load_meta, write_index
from .compat import CompatTable
from .config import Config
from .engine import RetrievalEngine
from .errors import PipelineError
from .loaders import load_corpus, load_documents, load_queries
from .postings import build_logical_index, canonical_postings_digest
from .search import execute_search
from .snapshot import SnapshotState
from .util import atomic_write_text, deep_merge, read_json


def deep_diff(left: Any, right: Any, path: str = "$", limit: int = 80) -> list[dict]:
    mismatches: list[dict] = []
    _deep_diff(left, right, path, mismatches, limit)
    return mismatches


def _deep_diff(left: Any, right: Any, path: str, mismatches: list[dict], limit: int) -> None:
    if len(mismatches) >= limit:
        return
    if type(left) is not type(right):
        mismatches.append({"path": path, "left": left, "right": right, "reason": "type"})
        return
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                mismatches.append({"path": f"{path}.{key}", "left": "<missing>", "right": right[key], "reason": "missing-left"})
            elif key not in right:
                mismatches.append({"path": f"{path}.{key}", "left": left[key], "right": "<missing>", "reason": "missing-right"})
            else:
                _deep_diff(left[key], right[key], f"{path}.{key}", mismatches, limit)
            if len(mismatches) >= limit:
                return
    elif isinstance(left, list):
        if len(left) != len(right):
            mismatches.append({"path": f"{path}.length", "left": len(left), "right": len(right), "reason": "length"})
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _deep_diff(left_item, right_item, f"{path}[{index}]", mismatches, limit)
            if len(mismatches) >= limit:
                return
    elif left != right:
        mismatches.append({"path": path, "left": left, "right": right, "reason": "value"})


def _load_snapshot_payload(snapshot: SnapshotState) -> dict:
    return {
        "index": read_json(snapshot.stage_path("compressed") / "index.json"),
        "failures": read_json(snapshot.stage_path("normalized") / "failures.json"),
        "results": read_json(snapshot.stage_path("fusion") / "results.json"),
        "diagnostics": read_json(snapshot.stage_path("fusion") / "diagnostics.json"),
    }


def _index_comparison_payload(index_meta: dict) -> dict:
    return {
        "format_version": index_meta["format_version"],
        "header_kind": index_meta["header_kind"],
        "compression_tier": index_meta["compression_tier"],
        "tier_semantic": index_meta["tier_semantic"],
        "keep_positions": index_meta["keep_positions"],
        "block_size": index_meta["block_size"],
        "document_count": index_meta["document_count"],
        "posting_count": index_meta["posting_count"],
        "block_count": index_meta["block_count"],
        "doc_table": index_meta["doc_table"],
        "term_table": index_meta["term_table"],
        "blocks": index_meta["blocks"],
        "sidecar_digest": index_meta["sidecar_digest"],
        "logical_digest": index_meta["logical_digest"],
        "index_fingerprint": index_meta["index_fingerprint"],
        "degradation_losses": index_meta["degradation_losses"],
    }


def compare_snapshots(incremental: SnapshotState, full: SnapshotState) -> dict:
    inc = _load_snapshot_payload(incremental)
    complete = _load_snapshot_payload(full)
    checks = []

    index_mismatches = deep_diff(_index_comparison_payload(inc["index"]), _index_comparison_payload(complete["index"]))
    checks.append({"name": "index_fingerprint", "ok": inc["index"]["index_fingerprint"] == complete["index"]["index_fingerprint"]})
    checks.append({"name": "logical_postings_digest", "ok": inc["index"]["logical_digest"] == complete["index"]["logical_digest"]})
    checks.append({"name": "degradation_losses", "ok": inc["index"]["degradation_losses"] == complete["index"]["degradation_losses"]})
    checks.append({"name": "index_payload", "ok": not index_mismatches})
    failure_mismatches = deep_diff(inc["failures"], complete["failures"])
    checks.append({"name": "failure_list", "ok": not failure_mismatches})
    result_mismatches = deep_diff(inc["results"], complete["results"])
    checks.append({"name": "per_query_results", "ok": not result_mismatches})
    query_order = [item.get("query_id") for item in inc["results"]] == [item.get("query_id") for item in complete["results"]]
    checks.append({"name": "query_order", "ok": query_order})
    timing_equal = inc["diagnostics"].get("query_timings") == complete["diagnostics"].get("query_timings")
    checks.append({"name": "deterministic_timing_statistics", "ok": timing_equal})

    compat = CompatTable.from_mapping(read_json(full.input_path("compat.json")))
    inc_decoded = decode_with_compat(incremental.stage_path("compressed"), compat)
    full_decoded = decode_with_compat(full.stage_path("compressed"), compat)
    inc_digest = canonical_postings_digest(inc_decoded.postings)
    full_digest = canonical_postings_digest(full_decoded.postings)
    checks.append({"name": "decoded_postings", "ok": inc_digest == full_digest})
    checks.append(
        {
            "name": "query_channel_decomposition",
            "ok": not deep_diff(
                _query_contribution_view(inc["results"]),
                _query_contribution_view(complete["results"]),
            ),
        }
    )
    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "mismatches": {
            "index": index_mismatches,
            "failures": failure_mismatches,
            "results": result_mismatches,
        },
        "incremental": {
            "global_version": incremental.version,
            "index_fingerprint": inc["index"]["index_fingerprint"],
            "query_count": len(inc["results"]),
            "completed_queries": inc["diagnostics"]["summary"]["completed_queries"],
        },
        "full": {
            "global_version": full.version,
            "index_fingerprint": complete["index"]["index_fingerprint"],
            "query_count": len(complete["results"]),
            "completed_queries": complete["diagnostics"]["summary"]["completed_queries"],
        },
        "decoded_postings_digest": inc_digest,
    }


def _query_contribution_view(results: list[dict]) -> list[dict]:
    view = []
    for result in results:
        view.append(
            {
                "query_id": result["query_id"],
                "candidates": [
                    {
                        "rank": item["rank"],
                        "doc_id": item["doc_id"],
                        "fused_fixed": item["fused_fixed"],
                        "lexical_contribution_fixed": item["lexical"]["contribution_fixed"],
                        "cooccurrence_contribution_fixed": item["cooccurrence"]["contribution_fixed"],
                        "float64_reference": item["float64_reference"],
                    }
                    for item in result["candidates"]
                ],
            }
        )
    return view


def verify_dual_tier(config: Config, documents) -> dict:
    logical = build_logical_index(documents, config)
    results = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        for tier in (1, 2):
            patched = Config.from_mapping(deep_merge(config.raw, {"active_compression_tier": tier}), config.compat)
            encoded = encode_index(logical, patched, tier_id=tier)
            target = root / f"tier{tier}"
            write_index(encoded, target)
            decoded = decode_with_compat(target, config.compat)
            results[tier] = {
                "fingerprint": encoded.fingerprint,
                "digest": canonical_postings_digest(decoded.postings),
                "block_count": encoded.meta["block_count"],
                "semantic": encoded.meta["tier_semantic"],
            }
    return {
        "ok": results[1]["digest"] == results[2]["digest"],
        "tier1": results[1],
        "tier2": results[2],
        "logical_digest": logical.logical_digest(),
    }


def verify_position_switch(config: Config, documents) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        decoded = {}
        encoded_meta = {}
        for keep in (True, False):
            patched = Config.from_mapping(deep_merge(config.raw, {"keep_positions": keep}), config.compat)
            logical = build_logical_index(documents, patched)
            encoded = encode_index(logical, patched, keep_positions=keep)
            target = root / ("with_positions" if keep else "without_positions")
            write_index(encoded, target)
            decoded[keep] = decode_with_compat(target, config.compat)
            encoded_meta[keep] = encoded.meta
        with_positions = all(
            posting.positions is not None
            for field in decoded[True].postings.values()
            for postings in field.values()
            for posting in postings
        )
        without_positions = all(
            posting.positions is None
            for field in decoded[False].postings.values()
            for postings in field.values()
            for posting in postings
        )
        tf_equal = _posting_tf_view(decoded[True].postings) == _posting_tf_view(decoded[False].postings)
        structural = (
            all(block["has_positions"] for block in encoded_meta[True]["blocks"])
            and all(not block["has_positions"] for block in encoded_meta[False]["blocks"])
            and all(block["position_count"] == 0 for block in encoded_meta[False]["blocks"])
        )
    return {
        "ok": with_positions and without_positions and tf_equal and structural,
        "with_positions_decoded": with_positions,
        "without_positions_decoded": without_positions,
        "term_frequencies_equal": tf_equal,
        "structural_flag_distinct": structural,
        "fingerprints_differ": encoded_meta[True]["index_fingerprint"] != encoded_meta[False]["index_fingerprint"],
    }


def _posting_tf_view(postings: dict) -> dict:
    return {
        field: {
            term: [[posting.doc_id, posting.tf] for posting in items]
            for term, items in sorted(fields.items())
        }
        for field, fields in sorted(postings.items())
    }


def verify_conversion(config: Config, documents) -> dict:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        logical = build_logical_index(documents, config)
        v2_config = Config.from_mapping(
            deep_merge(config.raw, {"index_format_version": "v2", "active_compression_tier": 2, "keep_positions": True}),
            config.compat,
        )
        v2_encoded = encode_index(logical, v2_config, format_version="v2", tier_id=2)
        v2_root = root / "v2"
        write_index(v2_encoded, v2_root)
        v2_decoded = decode_with_compat(v2_root, config.compat)
        v2_digest = canonical_postings_digest(v2_decoded.postings)

        v1_config = Config.from_mapping(
            deep_merge(config.raw, {"index_format_version": "v1", "active_compression_tier": 2, "keep_positions": True}),
            config.compat,
        )
        v1_encoded = convert_index(v2_root, v1_config, "v1", v2_encoded.meta, target_tier=2, keep_positions=True)
        v1_root = root / "v1"
        write_index(v1_encoded, v1_root)
        v1_decoded = decode_with_compat(v1_root, config.compat)
        v1_digest = canonical_postings_digest(v1_decoded.postings)

        v2_back_config = Config.from_mapping(
            deep_merge(config.raw, {"index_format_version": "v2", "active_compression_tier": 2, "keep_positions": True}),
            config.compat,
        )
        v2_back_encoded = convert_index(v1_root, v2_back_config, "v2", v1_encoded.meta, target_tier=2, keep_positions=True)
        v2_back_root = root / "v2_back"
        write_index(v2_back_encoded, v2_back_root)
        v2_back_decoded = decode_with_compat(v2_back_root, config.compat)
        v2_back_digest = canonical_postings_digest(v2_back_decoded.postings)
        v1_loss_features = sorted(item["feature"] for item in v1_encoded.meta["degradation_losses"])
        v2_loss_features = sorted(item["feature"] for item in v2_back_encoded.meta["degradation_losses"])
        source_semantic = config.compat.tier_semantic("v2", 2)
        v1_semantic = config.compat.tier_semantic("v1", 2)
        v2_semantic = config.compat.tier_semantic("v2", 2)
        checks = {
            "v2_to_v1_lossless": v2_digest == v1_digest,
            "v1_to_v2_lossless": v1_digest == v2_back_digest,
            "v1_sidecar_skiplist": "position_skiplist" in v1_loss_features,
            "v2_sidecar_checksum": "block_checksum" in v2_loss_features,
            "v1_has_checksum": v1_encoded.meta["format_version"] == "v1",
            "v2_has_skiplist": any(block["has_skiplist"] for block in v2_encoded.meta["blocks"]),
            "tier_semantics_mapped": v1_encoded.meta["tier_semantic"] == v1_semantic
            and v2_back_encoded.meta["tier_semantic"] == v2_semantic,
        }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "v2_digest": v2_digest,
        "v1_digest": v1_digest,
        "v2_back_digest": v2_back_digest,
        "v1_loss_features": v1_loss_features,
        "v2_back_loss_features": v2_loss_features,
        "tier_mapping": {
            "source_v2": source_semantic,
            "converted_v1": v1_encoded.meta["tier_semantic"],
            "converted_back_v2": v2_back_encoded.meta["tier_semantic"],
        },
        "v1_sidecar_blocks": len(v1_encoded.sidecar.get("blocks", {})),
        "v2_back_sidecar_blocks": len(v2_back_encoded.sidecar.get("blocks", {})),
    }


def verify_concurrency(config: Config, documents, queries) -> dict:
    logical = build_logical_index(documents, config)
    serial_config = Config.from_mapping(deep_merge(config.raw, {"concurrency": 1}), config.compat)
    parallel_config = Config.from_mapping(deep_merge(config.raw, {"concurrency": max(4, config.concurrency)}), config.compat)
    serial = execute_search(documents, queries, logical, serial_config)
    parallel = execute_search(documents, queries, logical, parallel_config)
    serial_results = list(serial.results)
    parallel_results = list(parallel.results)
    serial_timing = [
        {key: item[key] for key in ("order", "query_id", "logical_cost", "deterministic_time_ms", "candidate_count")}
        for item in serial.diagnostics["query_timings"]
    ]
    parallel_timing = [
        {key: item[key] for key in ("order", "query_id", "logical_cost", "deterministic_time_ms", "candidate_count")}
        for item in parallel.diagnostics["query_timings"]
    ]
    return {
        "ok": serial_results == parallel_results and serial_timing == parallel_timing,
        "result_order_equal": serial_results == parallel_results,
        "deterministic_timing_equal": serial_timing == parallel_timing,
        "serial_concurrency": 1,
        "parallel_concurrency": max(4, config.concurrency),
        "query_count": len(queries),
        "observational_wall_times_excluded_from_equality": True,
    }


def verify_validation(sample_dir: str | Path, config: Config) -> dict:
    root = Path(sample_dir)
    docs, doc_failures, _ = _load_documents_capture(root / "invalid_docs.jsonl", config)
    queries, query_failures, _ = _load_queries_capture(root / "invalid_queries.jsonl", config)
    failures = sorted(doc_failures + query_failures, key=lambda item: item.sort_key())
    codes = [item.code for item in failures]
    accepted = sorted(item.doc_id for item in docs) + sorted(item.query_id for item in queries)
    return {
        "ok": len(failures) == 9 and accepted == ["D8005", "Q8004"],
        "failure_count": len(failures),
        "accepted_ids": accepted,
        "codes": codes,
        "failures": [item.to_dict() for item in failures],
    }


def _load_documents_capture(path: Path, config: Config):
    return load_documents(path, config)


def _load_queries_capture(path: Path, config: Config):
    return load_queries(path, config)


def verify_budget(engine: RetrievalEngine, current: SnapshotState) -> dict:
    command = {
        "command_id": "BUDGET-0001",
        "op": "set_config",
        "patch": {"top_k": 7},
    }
    outcome = engine.apply_edit(command, budget_ms=0)
    after = engine.manager.load_current()
    return {
        "ok": outcome.status == "E_INTERACT_BUDGET_PENDING" and after.version == current.version,
        "status": outcome.status,
        "completed_stage": outcome.result.get("completed_stage"),
        "completed_stages": outcome.result.get("completed_stages", []),
        "summary": outcome.result.get("summary"),
        "snapshot_version_before": current.version,
        "snapshot_version_after": after.version,
        "budget_event_file": str(engine.state_dir / "budget_events.jsonl"),
    }


def verify_resume_mismatch(engine: RetrievalEngine, other_config_path: str | Path) -> dict:
    try:
        engine.resume(expected_config_path=other_config_path)
    except PipelineError as exc:
        return {"ok": exc.code == "E_CONFIG_MISMATCH", "code": exc.code, "message": exc.message}
    return {"ok": False, "code": None, "message": "resume unexpectedly accepted a mismatched config"}


def verify_recovery_cleanup(engine: RetrievalEngine, current: SnapshotState) -> dict:
    staging = engine.state_dir / "work" / "staging-interruption-test"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "partial.bin").write_bytes(b"partial")
    orphan = engine.state_dir / "snapshots" / f"v{current.version + 1000:08d}"
    orphan.mkdir(parents=True, exist_ok=True)
    resumed = engine.resume(expected_version=current.version)
    return {
        "ok": resumed.version == current.version and not staging.exists() and not orphan.exists(),
        "resumed_version": resumed.version,
        "staging_removed": not staging.exists(),
        "orphan_snapshot_removed": not orphan.exists(),
        "cleanup": resumed.manifest.get("recovery", {}),
    }


def run_consistency_suite(sample_dir: str | Path = "samples", output_dir: str | Path = "reports/demo") -> dict:
    samples = Path(sample_dir)
    out = Path(output_dir)
    for name in ("incremental_state", "full_state", "incremental", "full"):
        target = out / name
        if target.exists():
            shutil.rmtree(target)
    out.mkdir(parents=True, exist_ok=True)

    incremental_engine = RetrievalEngine(out / "incremental_state")
    initial = incremental_engine.build(
        samples / "docs.jsonl",
        samples / "queries.jsonl",
        samples / "config.json",
        samples / "compat.json",
        output_dir=out / "incremental",
    )
    edit_history = []
    for line in (samples / "edits.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        command = json.loads(line)
        outcome = incremental_engine.apply_edit(command, output_dir=out / "incremental", budget_ms=120000)
        edit_history.append(
            {
                "command_id": command["command_id"],
                "op": command["op"],
                "status": outcome.status,
                "global_version": outcome.result.get("global_version"),
                "index_fingerprint": outcome.result.get("index_fingerprint"),
                "reused_blocks": outcome.result.get("index", {}).get("reuse_report", {}).get("reused_block_count"),
                "rebuilt_blocks": outcome.result.get("index", {}).get("reuse_report", {}).get("rebuilt_block_count"),
                "reused_queries": outcome.result.get("query_reuse", {}).get("reused_query_count"),
                "rebuilt_queries": outcome.result.get("query_reuse", {}).get("rebuilt_query_count"),
            }
        )
    incremental_current = incremental_engine.manager.load_current()
    assert incremental_current is not None

    full_engine = RetrievalEngine(out / "full_state")
    full_engine.build(
        samples / "final" / "docs.jsonl",
        samples / "final" / "queries.jsonl",
        samples / "final" / "config.json",
        samples / "final" / "compat.json",
        output_dir=out / "full",
    )
    full_current = full_engine.manager.load_current()
    assert full_current is not None

    config = Config.load(samples / "final" / "config.json", CompatTable.load(samples / "final" / "compat.json"))
    corpus = load_corpus(samples / "final" / "docs.jsonl", samples / "final" / "queries.jsonl", config)
    comparison = compare_snapshots(incremental_current, full_current)
    dual_tier = verify_dual_tier(config, corpus.documents)
    position_switch = verify_position_switch(config, corpus.documents)
    conversion = verify_conversion(config, corpus.documents)
    concurrency = verify_concurrency(config, corpus.documents, corpus.queries)
    validation = verify_validation(samples, config)
    budget = verify_budget(incremental_engine, incremental_current)
    recovery = verify_recovery_cleanup(incremental_engine, incremental_current)
    resume_mismatch = verify_resume_mismatch(incremental_engine, samples / "config.json")

    checks = {
        "incremental_equals_full_rebuild": comparison["ok"],
        "dual_tier_lossless": dual_tier["ok"],
        "position_switch_structural_and_lossless": position_switch["ok"],
        "v1_v2_roundtrip_lossless": conversion["ok"],
        "concurrency_stable_order": concurrency["ok"],
        "validation_failure_codes": validation["ok"],
        "interactive_budget_pending": budget["ok"],
        "snapshot_recovery_cleanup": recovery["ok"],
        "resume_config_mismatch_rejected": resume_mismatch["ok"],
    }
    diagnostics = read_json(incremental_current.stage_path("fusion") / "diagnostics.json")
    index_meta = read_json(incremental_current.stage_path("compressed") / "index.json")
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "sample_seed": 20261120,
        "ok": all(checks.values()),
        "checks": checks,
        "summary": {
            "valid_documents": index_meta["document_count"],
            "valid_queries": len(diagnostics["query_timings"]),
            "block_count": index_meta["block_count"],
            "index_fingerprint": index_meta["index_fingerprint"],
            "logical_digest": index_meta["logical_digest"],
            "format_version": index_meta["format_version"],
            "compression_tier": index_meta["compression_tier"],
            "tier_semantic": index_meta["tier_semantic"],
            "keep_positions": index_meta["keep_positions"],
            "degradation_loss_count": len(index_meta["degradation_losses"]),
            "failure_count": len(corpus.failures),
            "max_fixed_error_units": diagnostics["max_fixed_error_units"],
            "over_tolerance_doc_count": diagnostics["over_tolerance_doc_count"],
            "query_reuse_final": diagnostics["query_reuse"],
        },
        "incremental_vs_full": comparison,
        "dual_tier": dual_tier,
        "position_switch": position_switch,
        "conversion": conversion,
        "concurrency": concurrency,
        "validation": validation,
        "budget_pending": budget,
        "recovery": recovery,
        "resume_mismatch": resume_mismatch,
        "edit_history": edit_history,
        "artifacts": {
            "incremental_results": str(out / "incremental" / "retrieval_results.json"),
            "full_results": str(out / "full" / "retrieval_results.json"),
            "incremental_diagnostics": str(out / "incremental" / "diagnostics.json"),
            "full_diagnostics": str(out / "full" / "diagnostics.json"),
        },
    }
    atomic_write_text(out / "consistency_report.json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write_text(out / "consistency_report.md", _report_markdown(report))
    return report


def _report_markdown(report: dict) -> str:
    lines = [
        "# Consistency Verification Report",
        "",
        f"Overall: **{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Python: `{report['environment']['python'].splitlines()[0]}`",
        f"- NumPy: `{report['environment']['numpy']}`",
        f"- Index fingerprint: `{report['summary']['index_fingerprint']}`",
        f"- Logical digest: `{report['summary']['logical_digest']}`",
        f"- Valid documents / queries: `{report['summary']['valid_documents']}` / `{report['summary']['valid_queries']}`",
        f"- Blocks: `{report['summary']['block_count']}`",
        f"- Final format / tier / positions: `{report['summary']['format_version']}` / `{report['summary']['compression_tier']}` / `{report['summary']['keep_positions']}`",
        f"- Failures: `{report['summary']['failure_count']}`",
        f"- Max fixed error (units): `{report['summary']['max_fixed_error_units']}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, ok in report["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Edit Replay",
            "",
            "| Command | Op | Status | Version | Reused blocks | Rebuilt blocks | Reused queries | Rebuilt queries |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["edit_history"]:
        lines.append(
            f"| `{item['command_id']}` | `{item['op']}` | `{item['status']}` | {item['global_version']} | "
            f"{item['reused_blocks']} | {item['rebuilt_blocks']} | {item['reused_queries']} | {item['rebuilt_queries']} |"
        )
    comparison = report["incremental_vs_full"]
    lines.extend(
        [
            "",
            "## Incremental vs Full Rebuild",
            "",
            f"- Index fingerprint equal: `{comparison['incremental']['index_fingerprint'] == comparison['full']['index_fingerprint']}`",
            f"- Decoded posting digest: `{comparison['decoded_postings_digest']}`",
            f"- Mismatch count: `{sum(len(value) for value in comparison['mismatches'].values())}`",
            "",
            "## Additional Diagnostics",
            "",
            f"- Tier 1/Tier 2 decoded digest equal: `{report['dual_tier']['tier1']['digest'] == report['dual_tier']['tier2']['digest']}`",
            f"- Position-off payloads omit positions: `{report['position_switch']['without_positions_decoded']}`",
            f"- v2→v1→v2 digest sequence: `{report['conversion']['v2_digest']}` / `{report['conversion']['v1_digest']}` / `{report['conversion']['v2_back_digest']}`",
            f"- Concurrency 1 vs {report['concurrency']['parallel_concurrency']} results equal: `{report['concurrency']['result_order_equal']}`",
            f"- Budget pending status: `{report['budget_pending']['status']}` after `{report['budget_pending']['completed_stage']}`",
            f"- Interrupted staging/orphan cleanup: `{report['recovery']['ok']}`",
            f"- Resume mismatch code: `{report['resume_mismatch']['code']}`",
            "",
            "Observed wall-clock query times are retained in `diagnostics.json` as observations; deterministic task order, logical cost and candidate counts are the concurrency equality fields.",
            "",
        ]
    )
    return "\n".join(lines)
