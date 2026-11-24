"""端到端运行管线：全量重建、增量编辑、并发对照与结果落盘。"""

import copy
import os

from .commands import load_commands
from .config import load_config
from .corpus import load_corpus
from .engine import ExtractionEngine
from .model import load_model
from .sample_data import generate
from .serialization import write_json
from .state import RuntimeState


def prepare_workspace(data_dir):
    info = generate(data_dir)
    return info


def build_runtime(data_dir, overrides=None):
    config = load_config(os.path.join(data_dir, "config.yaml"))
    if overrides:
        for key, value in overrides.items():
            object.__setattr__(config, key, value)
    model = load_model(os.path.join(data_dir, "model_package.json"), config)
    records, load_failures = load_corpus(
        os.path.join(data_dir, "sentences.jsonl"), model.vocabulary, config
    )
    state = RuntimeState(
        sentences={record.sentence_id: record for record in records},
        model=model,
        config=config,
    )
    return state, load_failures


def full_rebuild(data_dir, thread_count=None, overrides=None):
    state, load_failures = build_runtime(data_dir, overrides=overrides)
    if thread_count is not None:
        object.__setattr__(state.config, "thread_count", thread_count)
    engine = ExtractionEngine(state)
    report = engine.run()
    report.failures = load_failures + report.failures
    report.failures.sort(key=lambda item: item.sort_key(item))
    return engine, report


def incremental_run(data_dir, thread_count=None):
    state, load_failures = build_runtime(data_dir)
    if thread_count is not None:
        object.__setattr__(state.config, "thread_count", thread_count)
    engine = ExtractionEngine(state)
    baseline = engine.run()
    commands = load_commands(os.path.join(data_dir, "commands.json"))
    command_reports = []
    for command in commands:
        report = engine.run(command=command)
        payload_dict = report.to_dict()
        payload_dict["failures"] = sorted(
            [item.to_dict() for item in load_failures] + payload_dict["failures"],
            key=lambda row: (row["sentence_id"] or "", row["stage"], row["code"], row["message"]),
        )
        command_reports.append({
            "seq": command.seq,
            "kind": command.kind,
            "payload": command.payload,
            "report": payload_dict,
        })
    return engine, state, baseline, command_reports, load_failures


def rebuild_from_state(state):
    """按当前最终句子集合与模板参数从空白全量重建，用于增量一致性对照。"""
    import json as _json
    import tempfile
    import shutil
    data_dir = tempfile.mkdtemp(prefix="relx_rebuild_")
    with open(os.path.join(data_dir, "sentences.jsonl"), "w",
              encoding="utf-8", newline="\n") as handle:
        for sid in state.sentence_ids_sorted():
            handle.write(_json.dumps(state.sentences[sid].to_dict(),
                                     ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    with open(os.path.join(data_dir, "config.yaml"), "w",
              encoding="utf-8", newline="\n") as handle:
        from . import mini_yaml
        handle.write(mini_yaml.dumps({
            "entity_types": list(state.config.entity_types),
            "relation_types": list(state.config.relation_types),
            "max_term_length": state.config.max_term_length,
            "score_threshold": state.config.score_threshold,
            "fixedpoint_width": state.config.fixedpoint_width,
            "quant_level": state.config.quant_level,
            "thread_count": state.config.thread_count,
            "feature_budget_per_batch": state.config.feature_budget_per_batch,
            "snapshot_commit_batches": state.config.snapshot_commit_batches,
        }))
    from .fingerprint import fnv1a64
    model_dict = {
        "model_version": state.model.model_version,
        "feature_dim": state.model.feature_dim,
        "hash_buckets": state.model.hash_buckets,
        "term_hash": {
            term: int(fnv1a64(term.encode("utf-8")), 16) % state.model.hash_buckets
            for term in state.model.vocabulary
        },
        "entity_lexicon": {
            etype: list(terms) for etype, terms in state.model.entity_lexicon.items()
        },
        "templates": [
            {
                "template_id": tpl.template_id,
                "relation_type": tpl.relation_type,
                "trigger_terms": list(tpl.trigger_terms),
                "head_type": tpl.head_type,
                "tail_type": tpl.tail_type,
                "weight": tpl.weight,
                "version": 1,
            }
            for tpl in state.model.templates
        ],
    }
    with open(os.path.join(data_dir, "model_package.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        _json.dump(model_dict, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _engine, report = full_rebuild(data_dir)
    shutil.rmtree(data_dir, ignore_errors=True)
    return report


BUDGET_CODES = {"E_FEATURE_BUDGET_PENDING", "E_FEATURE_BUDGET_OVERSIZE"}


def normalize_report(report_dict):
    """剥离复用/重算与预算批次过程信息，仅保留最终抽取产物用于一致性比对。"""
    cleaned = copy.deepcopy(report_dict)
    for sentence in cleaned["sentences"]:
        sentence.pop("reuse_recompute", None)
    cleaned.pop("failures", None)
    cleaned.pop("budget", None)
    cleaned.pop("reuse", None)
    cleaned.pop("recompute", None)
    cleaned.pop("invalidations", None)
    return cleaned


def compare_artifacts(left, right):
    left_clean = normalize_report(left)
    right_clean = normalize_report(right)
    return left_clean == right_clean, left_clean, right_clean


def run_demo(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    info = prepare_workspace(data_dir)

    engine_full, report_full = full_rebuild(data_dir)
    write_json(os.path.join(output_dir, "full_rebuild.json"), report_full.to_dict())

    engine_inc, final_state, baseline, command_reports, load_failures = incremental_run(data_dir)
    write_json(os.path.join(output_dir, "incremental_baseline.json"), baseline.to_dict())
    write_json(os.path.join(output_dir, "incremental_commands.json"), command_reports)
    write_json(os.path.join(output_dir, "incremental_final.json"),
               command_reports[-1]["report"])

    final_report = rebuild_from_state(final_state)
    consistent, left_clean, right_clean = compare_artifacts(
        final_report.to_dict(), command_reports[-1]["report"]
    )
    write_json(os.path.join(output_dir, "consistency_check.json"), {
        "artifacts_identical": consistent,
        "full_rebuild": left_clean,
        "incremental_final": right_clean,
    })

    engine_serial, serial_report = full_rebuild(data_dir, thread_count=1)
    engine_pool, pool_report = full_rebuild(data_dir, thread_count=8)
    concurrency_ok, serial_clean, pool_clean = compare_artifacts(
        serial_report.to_dict(), pool_report.to_dict()
    )
    write_json(os.path.join(output_dir, "concurrency_check.json"), {
        "serial_threads": 1,
        "pool_threads": 8,
        "artifacts_identical": concurrency_ok,
        "serial": serial_clean,
        "pool": pool_clean,
    })

    quant_variants = {}
    for width, quant in ((8, 2), (16, 4), (32, 8)):
        _, qreport = full_rebuild(
            data_dir, overrides={"fixedpoint_width": width, "quant_level": quant}
        )
        quant_variants[f"w{width}_q{quant}"] = qreport.to_dict()
    write_json(os.path.join(output_dir, "quant_variants.json"), quant_variants)

    summary = {
        "sample_info": info,
        "load_failures": [item.to_dict() for item in load_failures],
        "full_rebuild": {
            "sentence_results": len(report_full.results),
            "failures": len(report_full.failures),
            "recompute_entries": len(report_full.recompute),
            "reuse_entries": len(report_full.reuse),
            "budget_batches": report_full.batches,
            "pending": report_full.pending,
            "ledger_consistent": report_full.ledger_consistent,
        },
        "incremental_final": {
            "sentence_results": len(command_reports[-1]["report"]["sentences"]),
            "failures": len(command_reports[-1]["report"]["failures"]),
            "budget_batches": command_reports[-1]["report"]["budget"]["batches"],
        },
        "incremental_vs_full_identical": consistent,
        "concurrency_identical": concurrency_ok,
        "cache_audit": engine_inc.cache.audit_rows(),
    }
    write_json(os.path.join(output_dir, "summary.json"), summary)
    return summary
