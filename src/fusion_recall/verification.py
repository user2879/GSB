from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from .commands import parse_commands
from .diagnostics import diagnose_query_flip
from .doctor import environment_report
from .engine import Engine, full_rebuild_engine, prepare_config
from .generator import build_config, build_vocab
from .hashing import canonical_hash
from .index import build_index
from .loading import load_documents, load_queries, validate_document
from .snapshot_recovery_check import simulated_recovery
from .vocab import Vocabulary, load_vocabulary


def _reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sample_bundle(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    vocab = load_vocabulary(root / "configs" / "vocab.json")
    import json

    raw_config = json.loads((root / "configs" / "pipeline.json").read_text(encoding="utf-8"))
    config = prepare_config(raw_config, vocab)
    documents = load_documents(config, vocab, path=root / "samples" / "documents.jsonl")
    queries = load_queries(config, vocab, path=root / "samples" / "queries.jsonl")
    commands = parse_commands(config, vocab, path=root / "samples" / "commands.jsonl")
    return {
        "vocab": vocab,
        "config": config,
        "documents": documents,
        "queries": queries,
        "commands": commands,
        "failures": documents["failures"] + queries["failures"] + commands["failures"],
    }


def _synthetic_doc(config, vocab, doc_id, version, terms, view):
    raw = {
        "doc_id": doc_id,
        "version": version,
        "views": {
            "original": {"language": "zh-CN", "fields": {"title": [], "body": []}},
            "translation": {"language": "en-US", "fields": {"title": [], "body": []}},
        },
    }
    raw["views"][view]["fields"]["body"] = list(terms)
    doc, failure = validate_document(raw, config, vocab)
    if failure is not None:
        raise AssertionError(failure)
    return doc


def synthetic_flip_checks() -> dict[str, Any]:
    vocab = Vocabulary(build_vocab())
    tf_config = prepare_config(build_config(), vocab)
    tf_config["top_k"] = 1
    tf_config["saturation_k"] = 1
    tf_docs = [
        _synthetic_doc(tf_config, vocab, "D001", 1, ["C05"] * 15, "original"),
        _synthetic_doc(tf_config, vocab, "D002", 1, ["C05"] * 16, "original"),
    ]
    tf_index = build_index(tf_docs, tf_config, vocab)
    tf_query = {
        "query_id": "SYN_TF",
        "terms": ["C05"],
        "query_fingerprint": "syn_tf",
        "language_preference_rational": {name: [1, 1] for name in tf_config["view_names"]},
    }
    tf_diagnosis = diagnose_query_flip(
        tf_query,
        tf_index,
        tf_config,
        vocab,
        {"frac_bits": 16, "quant_tier": "floor"},
        {"frac_bits": 8, "quant_tier": "floor"},
    )

    lang_config = prepare_config(build_config(), vocab)
    lang_config["top_k"] = 1
    lang_config["language_weight_rational"]["original"] = [1001, 1000]
    lang_config["language_weight_rational"]["translation"] = [1, 1]
    lang_docs = [
        _synthetic_doc(lang_config, vocab, "D002", 1, ["C06"], "original"),
        _synthetic_doc(lang_config, vocab, "D001", 1, ["C06"], "translation"),
    ]
    lang_index = build_index(lang_docs, lang_config, vocab)
    lang_query = {
        "query_id": "SYN_LANG",
        "terms": ["C06"],
        "query_fingerprint": "syn_lang",
        "language_preference_rational": {name: [1, 1] for name in lang_config["view_names"]},
    }
    lang_diagnosis = diagnose_query_flip(
        lang_query,
        lang_index,
        lang_config,
        vocab,
        {"frac_bits": 64, "quant_tier": "nearest"},
        {"frac_bits": 8, "quant_tier": "floor"},
    )
    return {
        "tf": {
            "flipped": tf_diagnosis["flipped"],
            "primary_cause": tf_diagnosis["primary_cause"],
            "tf_delta": tf_diagnosis["tf_quantization_changes_hybrid_ranking"],
            "tie_context": tf_diagnosis["tie_resolution_decides_boundary"],
            "selected_a": tf_diagnosis["selected_a"],
            "selected_b": tf_diagnosis["selected_b"],
        },
        "language": {
            "flipped": lang_diagnosis["flipped"],
            "primary_cause": lang_diagnosis["primary_cause"],
            "language_delta": lang_diagnosis["language_rounding_changes_hybrid_ranking"],
            "selected_a": lang_diagnosis["selected_a"],
            "selected_b": lang_diagnosis["selected_b"],
        },
    }


def run_verification(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    bundle = load_sample_bundle(root)
    vocab = bundle["vocab"]
    base_config = bundle["config"]
    docs = bundle["documents"]["documents"]
    queries = bundle["queries"]["queries"]
    commands = bundle["commands"]["commands"]
    failures = bundle["failures"]

    serial_root = _reset_dir(root / "workspace" / "verify-serial")
    serial = Engine(
        deepcopy(base_config),
        vocab,
        queries,
        root=serial_root,
        input_failures=deepcopy(failures),
    )
    serial.cold_build(deepcopy(docs))
    serial.apply_commands(deepcopy(commands))
    serial_report = serial.run_batch(unlimited=True, workers=1)
    serial.write_snapshot(serial_report)

    full_root = _reset_dir(root / "workspace" / "verify-full")
    full = full_rebuild_engine(
        deepcopy(serial.config),
        vocab,
        deepcopy(queries),
        list(serial.documents.values()),
        root=full_root,
        input_failures=deepcopy(failures),
        global_version=serial.global_version,
        workers=1,
    )
    full_report = full.write_snapshot()

    concurrent_root = _reset_dir(root / "workspace" / "verify-concurrent")
    concurrent = Engine(
        deepcopy(base_config),
        vocab,
        deepcopy(queries),
        root=concurrent_root,
        input_failures=deepcopy(failures),
    )
    concurrent.cold_build(deepcopy(docs))
    concurrent.apply_commands(deepcopy(commands))
    concurrent_report = concurrent.run_batch(unlimited=True, workers=4)
    concurrent.write_snapshot(concurrent_report)

    serial_material = serial.parity_material()
    full_material = full.parity_material()
    concurrent_material = concurrent.parity_material()
    serial_material.pop("command_history")
    full_material.pop("command_history")
    concurrent_material.pop("command_history")
    serial_hash = canonical_hash(serial_material)
    full_hash = canonical_hash(full_material)
    concurrent_hash = canonical_hash(concurrent_material)

    cache_root = _reset_dir(root / "workspace" / "verify-cache")
    cache_engine = Engine(
        deepcopy(base_config), vocab, deepcopy(queries), root=cache_root
    )
    cache_engine.cold_build(deepcopy(docs))
    cache_engine.run_batch(unlimited=True)
    replace_command = next(
        command
        for command in commands
        if command["op"] == "replace_document" and command["document"]["doc_id"] == "D004"
    )
    replace_scope = cache_engine.apply_command(deepcopy(replace_command))["scope"]
    selective_replace = cache_engine.run_batch(unlimited=True)
    selective_replace_summary = {
        "buckets_recomputed": selective_replace["last_run"]["buckets_recomputed"],
        "buckets_reused_count": len(selective_replace["last_run"]["buckets_reused"]),
        "fusion_recomputed": selective_replace["last_run"]["fusion_recomputed"],
        "fusion_reused": selective_replace["last_run"]["fusion_reused"],
        "rankings_reused": selective_replace["last_run"]["rankings_reused"],
    }
    language_command = {
        "op": "set_language_weight",
        "view": "translation",
        "weight": "1.40",
        "line": 999,
    }
    language_outcome = cache_engine.apply_command(language_command)
    after_language = cache_engine.run_batch(unlimited=True)
    language_run = {
        "buckets_reused": language_outcome["scope"]["buckets_reused"],
        "buckets_recomputed": language_outcome["scope"]["buckets_recomputed"],
        "fusion_recomputed": after_language["last_run"]["fusion_recomputed"],
        "fusion_reused": after_language["last_run"]["fusion_reused"],
    }
    unchanged_rerun = cache_engine.run_batch(unlimited=True, fresh_cycle=True)
    rerun_summary = {
        "fusion_reused": unchanged_rerun["last_run"]["fusion_reused"],
        "fusion_recomputed": unchanged_rerun["last_run"]["fusion_recomputed"],
        "rankings_reused": unchanged_rerun["last_run"]["rankings_reused"],
        "rankings_recomputed": unchanged_rerun["last_run"]["rankings_recomputed"],
    }

    budget_root = _reset_dir(root / "workspace" / "verify-budget")
    budget = Engine(deepcopy(base_config), vocab, deepcopy(queries), root=budget_root)
    budget.cold_build(deepcopy(docs), global_version=1)
    first_batch = budget.run_batch(batch_points=40, query_quota_points=40)
    budget_pointer = budget.write_snapshot(first_batch)
    first_summary = {
        "ready": first_batch["last_run"]["ready"],
        "quota_failures": first_batch["last_run"]["quota_failures"],
        "pending": first_batch["last_run"]["pending"],
        "todo": budget.scheduler_state["todo"],
        "batch_consumed": budget.scheduler_state["batch_consumed"],
        "batch_remaining": budget.scheduler_state["batch_remaining"],
    }
    second_batch = budget.run_batch(batch_points=24, query_quota_points=40)
    second_summary = {
        "ready": second_batch["last_run"]["ready"],
        "pending": second_batch["last_run"]["pending"],
        "todo": budget.scheduler_state["todo"],
    }

    recovery = simulated_recovery(root / "workspace" / "verify-recovery")
    environment = environment_report(root)
    flip_checks = synthetic_flip_checks()

    result = {
        "ok": serial_hash == full_hash == concurrent_hash,
        "environment": environment,
        "parity": {
            "serial_hash": serial_hash,
            "full_rebuild_hash": full_hash,
            "concurrent_hash": concurrent_hash,
            "incremental_equals_full": serial_hash == full_hash,
            "concurrent_equals_serial": serial_hash == concurrent_hash,
        },
        "sample_counts": {
            "valid_documents": len(docs),
            "valid_queries": len(queries),
            "valid_commands": len(commands),
            "failures": len(failures),
            "term_buckets": serial_report["counts"]["term_buckets"],
            "boundary_entries": serial_report["counts"]["boundary_entries"],
        },
        "replacement_d004_scope": replace_scope,
        "selective_replace_after_warm_cache": selective_replace_summary,
        "language_weight_run": {
            "bucket_reused_count": len(language_run["buckets_reused"]),
            "bucket_recomputed_count": len(language_run["buckets_recomputed"]),
            "fusion_recomputed": language_run["fusion_recomputed"],
            "fusion_reused": language_run["fusion_reused"],
        },
        "unchanged_rerun": rerun_summary,
        "budget_first_batch": first_summary,
        "budget_second_batch": second_summary,
        "budget_snapshot_pointer": budget_pointer,
        "recovery": recovery,
        "synthetic_flip_checks": flip_checks,
        "serial_report_summary": {
            "counts": serial_report["counts"],
            "business_codes": serial_report["business_codes"],
            "candidate_counts_by_view": serial_report["candidate_counts_by_view"],
            "shutdown": serial_report["shutdown"],
        },
    }
    return result


def write_verification_report(root: str | Path = ".") -> Path:
    root = Path(root)
    result = run_verification(root)
    output = root / "runs" / "verification_report.json"
    from .store import atomic_write_json

    atomic_write_json(output, result)
    return output
